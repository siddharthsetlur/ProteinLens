"""Classifiers that relate geometric descriptors to SAE node activations.

Contains:

* **kabsch_align / compute_rmsd** -- Kabsch superposition and RMSD.
* **superpose_fragments** -- Align top-K activated Ca fragments to build a
  motif template.
* **collect_node_fragments** -- Gather activated/background fragments across
  all proteins for a given SAE node.
* **train_motif_classifier** -- Train GBM + DT classifiers to separate
  activated vs background residue positions from local geometry.
* **compute_concordance_metrics** -- Quantify agreement between SAE
  activation and geometry-predicted probability at residue level.
* **fit_lasso_single_node** -- Protein-level LassoCV regression for a single
  SAE node.
* **format_monomial** -- Human-readable monomial string from Lasso weights.

Adapted from ``protein_results/kabsch_top_alignment.py``,
``protein_results/build_residue_motifs.py``, and
``protein_results/build_activation_multiset.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from proteinlens.analysis.geometry.residue_features import (
    extract_local_feature_vector,
    select_features,
)


# ===================================================================
# Fragment superposition (Kabsch)
# ===================================================================


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimally rotate + translate *mobile* onto *target* (Kabsch algorithm).

    Both inputs are (N, 3) Ca coordinate arrays. When the two backbones have
    different lengths, the shorter length is used (aligned from the
    N-terminus). The rotation is applied to the FULL *mobile* array.

    Parameters
    ----------
    mobile : np.ndarray
        Coordinates to be aligned, shape ``(N, 3)``.
    target : np.ndarray
        Reference coordinates, shape ``(M, 3)``.

    Returns
    -------
    np.ndarray
        Transformed *mobile* coordinates (same shape as input *mobile*).
    """
    n = min(len(mobile), len(target))
    P = mobile[:n].copy()
    Q = target[:n].copy()

    # Centre both coordinate sets
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P -= p_mean
    Q -= q_mean

    # Cross-covariance matrix
    H = P.T @ Q  # (3, 3)
    U, _S, Vt = np.linalg.svd(H)

    # Correct for possible reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T  # optimal rotation

    # Apply rotation + translation to ALL of mobile
    aligned = (mobile - p_mean) @ R.T + q_mean
    return aligned


def compute_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square deviation between two (N, 3) coordinate sets.

    Uses min(len(a), len(b)) residues, aligned from the N-terminus.

    Parameters
    ----------
    a, b : np.ndarray
        Coordinate arrays, each shape ``(N, 3)``.

    Returns
    -------
    float
        RMSD in Angstroms.
    """
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    return float(np.sqrt((diff * diff).sum() / n))


# ===================================================================
# Fragment collection
# ===================================================================


def collect_node_fragments(
    protein_data: list[dict],
    node_idx: int,
    half_w: int = 10,
    act_quantile: float = 0.80,
    max_fragments: int = 100,
    bg_ratio: int = 3,
) -> dict:
    """Collect activated and background Ca fragments for a SAE node.

    Scans all proteins in *protein_data*, identifies residue positions where
    SAE node *node_idx* fires above a quantile threshold, and extracts local
    Ca backbone fragments plus feature vectors. Background fragments are a
    mix of hard negatives (low-but-nonzero activation) and true zeros.

    Parameters
    ----------
    protein_data : list[dict]
        Each dict must have keys: ``act_matrix`` (N, D), ``ca`` (N, 3),
        ``profiles`` (from compute_residue_profiles), ``n_residues`` (int),
        ``accession`` (str), and optionally ``sequence`` (str).
    node_idx : int
        Index of the SAE node to analyse.
    half_w : int
        Half-window size for fragment extraction.
    act_quantile : float
        Quantile of nonzero activations used to set the activation threshold.
    max_fragments : int
        Maximum fragments for superposition. Up to ``5 * max_fragments``
        activated fragments are retained for classifier training; only
        ``max_fragments`` are used for Kabsch superposition.
    bg_ratio : int
        Target ratio of background to activated fragments.

    Returns
    -------
    dict
        Keys: ``activated`` (list[dict]), ``background`` (list[dict]),
        ``threshold`` (float), ``n_total_active`` (int),
        ``n_hard_negatives`` (int). Each fragment dict contains:
        ``accession``, ``position``, ``fragment`` (Ca coords),
        ``features`` (full feature vector), ``category`` (int),
        ``activation`` (float).
    """
    # First pass: gather all nonzero activations across proteins to set threshold
    all_nonzero: list[np.ndarray] = []
    for pdata in protein_data:
        col = pdata["act_matrix"][:, node_idx]
        nz = col[col > 0]
        if len(nz) > 0:
            all_nonzero.append(nz)

    if not all_nonzero:
        return {
            "activated": [], "background": [], "threshold": 0.0,
            "n_total_active": 0, "n_hard_negatives": 0,
        }

    all_nonzero_arr = np.concatenate(all_nonzero)
    n_total_active = len(all_nonzero_arr)

    # Not enough activated positions to build a meaningful classifier
    if n_total_active < 20:
        return {
            "activated": [], "background": [], "threshold": 0.0,
            "n_total_active": n_total_active, "n_hard_negatives": 0,
        }

    threshold = float(np.quantile(all_nonzero_arr, act_quantile))
    if threshold <= 0:
        threshold = float(np.median(all_nonzero_arr))

    # Second pass: categorise positions by activation value (NO feature
    # extraction yet — deferred to after sampling to avoid computing
    # features for ~140k positions when only ~2k are kept).
    # Each entry is (protein_index, position, activation_value).
    act_positions: list[tuple[int, int, float]] = []
    hneg_positions: list[tuple[int, int, float]] = []
    bg_positions: list[tuple[int, int, float]] = []

    for pi, pdata in enumerate(protein_data):
        col = pdata["act_matrix"][:, node_idx]
        n = pdata["n_residues"]

        for pos in range(half_w, n - half_w):
            val = float(col[pos])
            if val >= threshold:
                act_positions.append((pi, pos, val))
            elif val > 0:
                hneg_positions.append((pi, pos, val))
            else:
                bg_positions.append((pi, pos, val))

    # Sort activated by activation strength (descending)
    act_positions.sort(key=lambda x: -x[2])

    # Build combined background: ~50% hard negatives + ~50% true zeros
    rng = np.random.default_rng(42)
    n_bg_total = min(
        len(bg_positions) + len(hneg_positions),
        len(act_positions) * bg_ratio,
    )
    n_hard = min(len(hneg_positions), n_bg_total // 2)
    n_zero = min(len(bg_positions), n_bg_total - n_hard)

    if n_hard > 0 and len(hneg_positions) > n_hard:
        idx = rng.choice(len(hneg_positions), size=n_hard, replace=False)
        hneg_positions = [hneg_positions[i] for i in idx]
    else:
        hneg_positions = hneg_positions[:n_hard]

    if n_zero > 0 and len(bg_positions) > n_zero:
        idx = rng.choice(len(bg_positions), size=n_zero, replace=False)
        bg_positions = [bg_positions[i] for i in idx]
    else:
        bg_positions = bg_positions[:n_zero]

    # Now compute features + fragments ONLY for the kept positions (~2k)
    def _build_entry(pi: int, pos: int, val: float) -> dict:
        pdata = protein_data[pi]
        feat_vec = extract_local_feature_vector(
            pdata["profiles"], pdata["ca"], pos, half_w,
            sequence=pdata.get("sequence", None),
        )
        return {
            "accession": pdata["accession"],
            "position": pos,
            "fragment": pdata["ca"][pos - half_w: pos + half_w + 1].copy(),
            "features": feat_vec,
            "category": int(pdata["profiles"]["categories"][pos]),
            "activation": val,
        }

    activated = [_build_entry(*t) for t in act_positions[:max_fragments * 5]]
    combined_background = (
        [_build_entry(*t) for t in hneg_positions]
        + [_build_entry(*t) for t in bg_positions]
    )

    return {
        "activated": activated,
        "background": combined_background,
        "threshold": threshold,
        "n_total_active": n_total_active,
        "n_hard_negatives": len(hneg_positions),
    }


# ===================================================================
# Fragment superposition
# ===================================================================


def superpose_fragments(
    activated: list[dict],
    top_k: int = 100,
) -> dict:
    """Kabsch-align top-K activated fragments and compute a motif template.

    Performs two rounds of alignment: first to the strongest fragment, then
    to the resulting mean structure (for a more accurate template).

    Parameters
    ----------
    activated : list[dict]
        Fragment dicts from :func:`collect_node_fragments`, sorted by
        activation descending. Each must have a ``fragment`` key with
        shape ``(W, 3)`` Ca coordinates.
    top_k : int
        Maximum number of fragments to use.

    Returns
    -------
    dict
        Keys: ``mean_structure`` (W, 3) or None, ``per_pos_std`` (W,),
        ``mean_rmsd``, ``std_rmsd``, ``median_rmsd`` (floats),
        ``rmsds`` (list[float]), ``n_fragments`` (int),
        ``aligned_fragments`` (K, W, 3) array.
    """
    if len(activated) < 3:
        return {
            "mean_structure": None,
            "mean_rmsd": float("inf"),
            "std_rmsd": 0.0,
            "rmsds": [],
            "n_fragments": len(activated),
            "per_pos_std": None,
            "aligned_fragments": None,
        }

    frags = [a["fragment"] for a in activated[:top_k]]
    ref = frags[0]
    w = ref.shape[0]

    # First pass: align all to the strongest (reference) fragment
    aligned = [ref.copy()]
    for frag in frags[1:]:
        if frag.shape[0] != w:
            continue
        aln = kabsch_align(frag, ref)
        aligned.append(aln)

    aligned_arr = np.array(aligned)
    mean_structure = aligned_arr.mean(axis=0)

    # Second pass: align to the mean for a more accurate template
    aligned2 = []
    rmsds2 = []
    for frag in frags:
        if frag.shape[0] != w:
            continue
        aln = kabsch_align(frag, mean_structure)
        aligned2.append(aln)
        rmsds2.append(compute_rmsd(aln, mean_structure))

    aligned2_arr = np.array(aligned2)
    mean_structure2 = aligned2_arr.mean(axis=0)
    per_pos_std2 = aligned2_arr.std(axis=0).mean(axis=1)  # (W,)

    return {
        "mean_structure": mean_structure2,
        "per_pos_std": per_pos_std2,
        "mean_rmsd": float(np.mean(rmsds2)) if rmsds2 else float("inf"),
        "std_rmsd": float(np.std(rmsds2)) if rmsds2 else 0.0,
        "median_rmsd": float(np.median(rmsds2)) if rmsds2 else float("inf"),
        "rmsds": [float(r) for r in rmsds2],
        "n_fragments": len(aligned2),
        "aligned_fragments": aligned2_arr,
    }


# ===================================================================
# Motif classifier (GBM + DT + RF)
# ===================================================================


def train_motif_classifier(
    activated: list[dict],
    background: list[dict],
    feature_names: list[str],
    max_depth: int = 4,
    cv_folds: int = 5,
) -> dict:
    """Train classifiers to separate activated vs background positions.

    Uses three models:

    1. **Decision tree** (shallow) -- for human-readable rules.
    2. **Gradient-boosted ensemble (GBM)** -- primary model used for
       ``predict_proba`` and concordance analysis.
    3. **Random forest** -- for comparison AUC.

    Cross-validation uses leave-proteins-out (GroupKFold) when enough
    proteins are available, otherwise StratifiedKFold. The optimal
    probability threshold is found via precision-recall analysis.

    Parameters
    ----------
    activated : list[dict]
        Fragment dicts with ``features`` (full-length vector) and
        ``accession`` keys.
    background : list[dict]
        Same structure as *activated*, for negative samples.
    feature_names : list[str]
        Names for the active feature columns (used in decision tree text
        and importance dict). Length must match the feature vectors after
        :func:`select_features`.
    max_depth : int
        Maximum depth for the interpretable decision tree.
    cv_folds : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Keys: ``tree`` (GBM model or None), ``decision_tree`` (DT model),
        ``rules`` (str), ``f1_cv``, ``auc_cv``, ``gbm_auc_cv``,
        ``rf_auc_cv``, ``lpo_auc`` (floats), ``n_unique_proteins`` (int),
        ``feature_importances`` (dict), ``optimal_threshold`` (float),
        ``n_pos``, ``n_neg`` (int).
    """
    # Bail out early if not enough data
    if len(activated) < 20 or len(background) < 20:
        return {
            "tree": None,
            "decision_tree": None,
            "rules": "Insufficient data",
            "f1_cv": 0.0,
            "auc_cv": 0.0,
            "gbm_auc_cv": 0.0,
            "rf_auc_cv": 0.0,
            "lpo_auc": 0.0,
            "feature_importances": {},
            "optimal_threshold": 0.5,
            "n_pos": len(activated),
            "n_neg": len(background),
            "n_unique_proteins": 0,
        }

    # Build feature matrix using active feature subset
    X_pos = np.array([select_features(a["features"]) for a in activated])
    X_neg = np.array([select_features(b["features"]) for b in background])
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    # Remove rows with NaN/inf
    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y = y[valid]

    if len(X) < 40 or y.sum() < 10:
        return {
            "tree": None,
            "decision_tree": None,
            "rules": "Insufficient valid data after filtering",
            "f1_cv": 0.0,
            "auc_cv": 0.0,
            "gbm_auc_cv": 0.0,
            "rf_auc_cv": 0.0,
            "lpo_auc": 0.0,
            "feature_importances": {},
            "optimal_threshold": 0.5,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_unique_proteins": 0,
        }

    # -- 1. Decision tree for interpretable rules --
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(5, int(0.02 * len(X))),
        class_weight="balanced",
    )
    tree.fit(X, y)
    rules = export_text(tree, feature_names=feature_names, decimals=4)

    # -- 2. GBM for precise probability calibration --
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())

    gbm = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=max(5, int(0.02 * len(X))),
        random_state=42,
    )
    gbm.fit(X, y)

    # -- 3. Cross-validated metrics + threshold calibration --
    # Build protein group labels for leave-proteins-out CV
    acc_pos = [a["accession"] for a in activated]
    acc_neg = [b["accession"] for b in background]
    all_accessions_arr = np.array(acc_pos + acc_neg)[valid]
    unique_proteins = np.unique(all_accessions_arr)
    n_unique = len(unique_proteins)

    n_cv = min(cv_folds, n_unique)
    optimal_threshold = 0.5
    f1_cv = 0.0
    auc_cv = 0.0
    gbm_auc_cv = 0.0
    rf_auc_cv = 0.0
    lpo_auc = 0.0

    if n_cv >= 2:
        # Prefer GroupKFold (leave-proteins-out) if enough proteins
        if n_unique >= 5:
            cv = GroupKFold(n_splits=min(n_cv, n_unique))
            cv_split_args = (X, y, all_accessions_arr)
            use_groups = True
        else:
            cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
            cv_split_args = (X, y)
            use_groups = False

        # DT cross-val metrics
        try:
            f1_scores = cross_val_score(
                tree, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="f1",
            )
            f1_cv = float(np.mean(f1_scores))
        except Exception:
            f1_cv = 0.0
        try:
            auc_scores = cross_val_score(
                tree, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="roc_auc",
            )
            auc_cv = float(np.mean(auc_scores))
        except Exception:
            auc_cv = 0.0

        # GBM cross-val AUC
        try:
            gbm_auc_scores = cross_val_score(
                gbm, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="roc_auc",
            )
            gbm_auc_cv = float(np.mean(gbm_auc_scores))
        except Exception:
            gbm_auc_cv = 0.0

        # Optimal threshold via CV precision-recall on held-out GBM probs
        all_probs = np.zeros(len(y))
        for train_idx, val_idx in cv.split(*cv_split_args):
            if len(np.unique(y[train_idx])) < 2:
                all_probs[val_idx] = 0.5
                continue
            gbm_cv = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=max(5, int(0.02 * len(X))),
                random_state=42,
            )
            gbm_cv.fit(X[train_idx], y[train_idx])
            probs = gbm_cv.predict_proba(X[val_idx])
            all_probs[val_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        try:
            precision, recall, thresholds = precision_recall_curve(y, all_probs)
            f1_arr = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
            best_idx = np.argmax(f1_arr[:-1])
            optimal_threshold = float(thresholds[best_idx])
            optimal_threshold = max(0.3, min(0.95, optimal_threshold))
        except Exception:
            optimal_threshold = 0.5

        # Leave-proteins-out AUC from held-out GBM probs
        try:
            lpo_auc = float(roc_auc_score(y, all_probs))
        except Exception:
            lpo_auc = 0.0

        # RF for comparison
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced",
            random_state=42, n_jobs=1,
        )
        try:
            rf_auc = cross_val_score(
                rf, X, y, cv=cv,
                groups=all_accessions_arr if use_groups else None,
                scoring="roc_auc",
            )
            rf_auc_cv = float(np.mean(rf_auc))
        except Exception:
            rf_auc_cv = 0.0

    # Feature importances from GBM
    importances = {
        feature_names[i]: float(gbm.feature_importances_[i])
        for i in range(len(feature_names))
        if gbm.feature_importances_[i] > 0.005
    }
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    return {
        "tree": gbm,  # GBM is the primary model for predict_proba
        "decision_tree": tree,
        "rules": rules,
        "f1_cv": f1_cv,
        "auc_cv": auc_cv,
        "gbm_auc_cv": gbm_auc_cv,
        "rf_auc_cv": rf_auc_cv,
        "lpo_auc": lpo_auc,
        "n_unique_proteins": n_unique,
        "feature_importances": importances,
        "optimal_threshold": optimal_threshold,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ===================================================================
# Concordance metrics
# ===================================================================


def compute_concordance_metrics(
    protein_data: list[dict],
    node_idx: int,
    tree,
    threshold: float,
    geom_threshold: float,
    half_w: int,
    feat_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> dict:
    """Quantify concordance between SAE activation and geometry prediction.

    For every residue in every protein (where the node fires at least once),
    compares the raw SAE activation against the GBM's predicted P(active).

    Parameters
    ----------
    protein_data : list[dict]
        Same format as :func:`collect_node_fragments`.
    node_idx : int
        SAE node index.
    tree : sklearn classifier
        Trained GBM model with ``predict_proba`` (from
        :func:`train_motif_classifier`).
    threshold : float
        SAE activation threshold (from fragment collection).
    geom_threshold : float
        GBM probability threshold for "geometry predicts active".
    half_w : int
        Half-window for feature extraction.

    Returns
    -------
    dict
        Continuous metrics: ``spearman_r``, ``spearman_p``,
        ``residue_auroc``, ``avg_precision``, ``cosine_sim``.
        Binary metrics: ``precision``, ``recall``, ``f1``, ``iou``.
        Counts: ``tp``, ``fp``, ``fn``, ``tn``, ``n_residues``,
        ``n_proteins``.
    """
    empty = {
        "spearman_r": 0.0, "spearman_p": 1.0,
        "residue_auroc": 0.0, "avg_precision": 0.0, "cosine_sim": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "n_residues": 0, "n_proteins": 0,
    }
    if tree is None:
        return empty

    all_sae_act: list[float] = []
    all_geom_prob: list[float] = []
    n_proteins_used = 0

    for pdata in protein_data:
        col = pdata["act_matrix"][:, node_idx]
        ca = pdata["ca"]
        profiles = pdata["profiles"]
        seq = pdata.get("sequence", None)
        n = pdata["n_residues"]

        # Only include proteins where the node fires at least once
        if not np.any(col >= threshold):
            continue

        n_proteins_used += 1

        # Collect feature vectors for this protein in batch
        positions = list(range(half_w, n - half_w))
        sae_vals = [float(col[pos]) for pos in positions]
        batch_fvs: list[np.ndarray] = []
        batch_indices: list[int] = []  # index into positions
        geom_probs = [0.0] * len(positions)

        acc = pdata["accession"]
        for i, pos in enumerate(positions):
            feat_vec = None
            if feat_cache is not None:
                feat_vec = feat_cache.get((acc, pos))
            if feat_vec is None:
                feat_vec = extract_local_feature_vector(
                    profiles, ca, pos, half_w, sequence=seq,
                )
            if feat_vec is not None and np.all(np.isfinite(feat_vec)):
                batch_fvs.append(select_features(feat_vec))
                batch_indices.append(i)

        # Batch predict_proba (one call per protein instead of per-position)
        if batch_fvs:
            X_batch = np.array(batch_fvs)
            probs_batch = tree.predict_proba(X_batch)
            for j, idx in enumerate(batch_indices):
                p = probs_batch[j]
                geom_probs[idx] = float(p[1] if len(p) > 1 else p[0])

        all_sae_act.extend(sae_vals)
        all_geom_prob.extend(geom_probs)

    total = len(all_sae_act)
    if total < 10 or n_proteins_used == 0:
        return empty

    sae_arr = np.array(all_sae_act)
    geom_arr = np.array(all_geom_prob)
    sae_binary = (sae_arr >= threshold).astype(int)
    geom_binary = (geom_arr >= geom_threshold).astype(int)

    # Binary confusion matrix
    tp = int(np.sum((sae_binary == 1) & (geom_binary == 1)))
    fp = int(np.sum((sae_binary == 0) & (geom_binary == 1)))
    fn = int(np.sum((sae_binary == 1) & (geom_binary == 0)))
    tn = int(np.sum((sae_binary == 0) & (geom_binary == 0)))

    # -- Continuous metrics --
    try:
        sp = stats.spearmanr(sae_arr, geom_arr)
        spearman_r, spearman_p = float(sp.statistic), float(sp.pvalue)
    except Exception:
        spearman_r, spearman_p = 0.0, 1.0

    try:
        res_auroc = float(roc_auc_score(sae_binary, geom_arr))
    except Exception:
        res_auroc = 0.0

    try:
        ap = float(average_precision_score(sae_binary, geom_arr))
    except Exception:
        ap = 0.0

    dot = np.dot(sae_arr, geom_arr)
    norms = np.linalg.norm(sae_arr) * np.linalg.norm(geom_arr)
    cos_sim = float(dot / max(norms, 1e-12))

    # -- Binary metrics (at calibrated threshold) --
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    iou = tp / max(1, tp + fp + fn)

    return {
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
        "residue_auroc": round(res_auroc, 4),
        "avg_precision": round(ap, 4),
        "cosine_sim": round(cos_sim, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "iou": round(float(iou), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_residues": int(total),
        "n_proteins": int(n_proteins_used),
    }


# ===================================================================
# Protein-level Lasso regression (single node)
# ===================================================================


def fit_lasso_single_node(
    X: np.ndarray,
    y: np.ndarray,
    geom_names: list[str],
    cv_folds: int = 5,
) -> dict | None:
    """Fit a LassoCV regression for a single SAE node.

    Given a geometry matrix *X* and target activations *y* (both restricted
    to proteins that fire on this node), fits a Lasso regression with
    automatic alpha selection via cross-validation.

    The Lasso drives irrelevant weights to exactly zero, yielding a sparse
    monomial that generalises instead of memorising.

    Parameters
    ----------
    X : np.ndarray
        Geometry feature matrix, shape ``(n_samples, n_features)``.
        All rows should be finite.
    y : np.ndarray
        Target activation values, shape ``(n_samples,)``. Should be > 0
        (pre-filtered to active proteins only).
    geom_names : list[str]
        Feature names corresponding to columns of *X*.
    cv_folds : int
        Number of cross-validation folds.

    Returns
    -------
    dict or None
        ``None`` if no signal detected. Otherwise a dict with keys:
        ``r2``, ``r2_adj``, ``r2_cv``, ``pearson_r``, ``n_samples``,
        ``n_nonzero``, ``alpha_chosen``, ``monomial``, ``top_features``
        (list of dicts), ``weights_raw`` (list), ``intercept_raw`` (float).
    """
    n = len(y)
    if n < 10:
        return None

    if y.std() < 1e-12:
        return None

    # Standardise features so Lasso weights are comparable across features
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    n_cv = min(cv_folds, n)

    model = LassoCV(
        cv=n_cv,
        max_iter=5000,
        n_alphas=30,
        tol=1e-3,
    )
    model.fit(X_sc, y)

    y_pred = model.predict(X_sc)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-8))
    p = n_nonzero
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)

    # R^2_cv from LassoCV's internal cross-validation
    best_alpha_idx = np.argmin(model.mse_path_.mean(axis=1))
    mse_cv = model.mse_path_[best_alpha_idx].mean()
    r2_cv = 1.0 - mse_cv / y.var() if y.var() > 0 else 0.0

    # Skip nodes where cross-validated performance is negative (no signal)
    if r2_cv < 0.0:
        return None

    # Pearson correlation between actual and predicted
    pr, _ = stats.pearsonr(y, y_pred)

    # Identify top features by absolute standardised weight
    w = model.coef_
    order = np.argsort(np.abs(w))[::-1]
    top_feats = [
        {
            "feature": geom_names[int(i)],
            "weight": float(w[i]),
            "abs_weight": float(abs(w[i])),
        }
        for i in order
        if abs(w[i]) > 1e-8
    ][:10]

    # Convert standardised weights to raw (un-standardised) scale:
    # y_pred = X_raw @ w_raw + b_raw
    w_raw = w / scaler.scale_
    b_raw = float(model.intercept_ - (w * scaler.mean_ / scaler.scale_).sum())

    monomial = format_monomial(
        [float(x) for x in w_raw], b_raw, geom_names
    )

    return {
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "r2_cv": float(r2_cv),
        "pearson_r": float(pr),
        "n_samples": n,
        "n_nonzero": n_nonzero,
        "alpha_chosen": float(model.alpha_),
        "monomial": monomial,
        "top_features": top_feats,
        "weights_raw": [float(x) for x in w_raw],
        "intercept_raw": b_raw,
    }


def format_monomial(
    weights_raw: list[float],
    intercept_raw: float,
    geom_names: list[str],
    threshold: float = 1e-6,
    max_terms: int = 10,
) -> str:
    """Format linear model weights as a human-readable monomial string.

    Produces output like: ``y_hat = 0.342*hairpin_score - 0.187*avg_curvature + 0.003``

    Only includes terms with ``|weight| > threshold`` (important for Lasso
    where most weights are exactly zero). Sorted by ``|weight|`` descending.

    Parameters
    ----------
    weights_raw : list[float]
        Un-standardised weights (one per feature).
    intercept_raw : float
        Un-standardised intercept.
    geom_names : list[str]
        Feature names corresponding to *weights_raw*.
    threshold : float
        Minimum absolute weight to include a term.
    max_terms : int
        Maximum number of terms to show.

    Returns
    -------
    str
        Human-readable monomial string.
    """
    pairs = [
        (geom_names[i], w)
        for i, w in enumerate(weights_raw)
        if abs(w) > threshold
    ]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    pairs = pairs[:max_terms]

    if not pairs:
        return f"y_hat = {intercept_raw:.4g}"

    terms = []
    for idx, (name, w) in enumerate(pairs):
        if idx == 0:
            terms.append(f"{w:.4g}*{name}")
        elif w < 0:
            terms.append(f"- {abs(w):.4g}*{name}")
        else:
            terms.append(f"+ {w:.4g}*{name}")

    if abs(intercept_raw) > threshold:
        if intercept_raw < 0:
            terms.append(f"- {abs(intercept_raw):.4g}")
        else:
            terms.append(f"+ {intercept_raw:.4g}")

    return "y_hat = " + " ".join(terms)
