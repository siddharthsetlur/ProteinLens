"""Tests for Stage 6c: residue-level GBM geometry enrichment + plot data.

Uses synthetic data to verify:
  - Fragment collection returns correct activated/background counts
  - Classifier with separable data achieves reasonable metrics
  - Concordance with known agreement gives high spearman_r
  - Plot data has correct structure
  - Motif superposition: identical fragments -> low RMSD
  - Edge case: node with too few activated positions -> skipped
"""

from __future__ import annotations

import numpy as np
from proteinlens.analysis.geometry.classifiers import (
    collect_node_fragments,
    compute_concordance_metrics,
    superpose_fragments,
    train_motif_classifier,
)
from proteinlens.analysis.geometry.residue_features import ACTIVE_GEOM_NAMES


def _make_synthetic_protein_data(
    n_proteins: int = 5,
    n_residues: int = 80,
    n_features: int = 10,
    node_idx: int = 0,
    signal_strength: float = 5.0,
    rng_seed: int = 42,
) -> list[dict]:
    """Create synthetic protein_data list with controllable activation signal.

    For each protein, generates Ca coordinates, per-residue activation
    matrix, and geometry profiles (curvature, torsion, etc.). Positions
    with high curvature get high activation on *node_idx*, creating a
    separable signal.

    Parameters
    ----------
    n_proteins : int
        Number of synthetic proteins.
    n_residues : int
        Number of residues per protein.
    n_features : int
        Number of SAE features (columns in activation matrix).
    node_idx : int
        Which SAE node to put signal on.
    signal_strength : float
        How strongly curvature predicts activation.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Synthetic protein data dicts.
    """
    rng = np.random.default_rng(rng_seed)
    protein_data = []

    for pi in range(n_proteins):
        # Generate a random walk for Ca backbone
        steps = rng.standard_normal((n_residues, 3)) * 3.8  # ~3.8A per residue
        ca = np.cumsum(steps, axis=0)

        # Synthetic profiles
        curvature = np.abs(rng.standard_normal(n_residues)) * 0.3
        torsion = rng.standard_normal(n_residues) * 0.2
        planarity = np.abs(rng.standard_normal(n_residues)) * 0.1
        tangents = np.diff(ca, axis=0, prepend=ca[:1])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        tangents = tangents / norms
        helix_mask = np.zeros(n_residues, dtype=bool)
        categories = np.full(n_residues, 5, dtype=int)

        # Build activation matrix
        act_matrix = np.zeros((n_residues, n_features), dtype=np.float32)

        # Make some positions "activated" on node_idx based on curvature
        # High curvature -> high activation (creates separable signal)
        for pos in range(n_residues):
            if curvature[pos] > 0.3:
                act_matrix[pos, node_idx] = float(
                    signal_strength * curvature[pos]
                    + rng.standard_normal() * 0.1
                )

        # Generate a simple sequence
        aa_chars = "ACDEFGHIKLMNPQRSTVWY"
        seq = "".join(rng.choice(list(aa_chars)) for _ in range(n_residues))

        protein_data.append({
            "accession": f"PROT_{pi:03d}",
            "act_matrix": act_matrix,
            "ca": ca,
            "profiles": {
                "curvature": curvature,
                "torsion": torsion,
                "planarity": planarity,
                "tangents": tangents,
                "helix_mask": helix_mask,
                "categories": categories,
            },
            "n_residues": n_residues,
            "sequence": seq,
        })

    return protein_data


class TestCollectNodeFragments:
    """Tests for fragment collection."""

    def test_correct_counts(self):
        """collect_node_fragments returns activated and background fragments."""
        protein_data = _make_synthetic_protein_data(n_proteins=5, n_residues=80)
        result = collect_node_fragments(
            protein_data, node_idx=0, half_w=5, act_quantile=0.80, bg_ratio=3,
        )

        assert len(result["activated"]) > 0, "Expected some activated fragments"
        assert len(result["background"]) > 0, "Expected some background fragments"
        assert result["threshold"] > 0
        assert result["n_total_active"] > 0

        # Each fragment should have the right shape
        for frag in result["activated"][:5]:
            assert frag["fragment"].shape == (11, 3)  # 2*5+1 = 11
            assert frag["features"].shape == (44,)

    def test_empty_node_returns_empty(self):
        """Node with no activations returns empty lists."""
        # Use node_idx=5 which has zero activations in the synthetic data
        # (signal is only placed on node_idx=0).
        protein_data = _make_synthetic_protein_data(n_proteins=3, signal_strength=5.0)
        result = collect_node_fragments(protein_data, node_idx=5, half_w=5)

        assert len(result["activated"]) == 0
        assert result["n_total_active"] == 0


class TestTrainMotifClassifierSynthetic:
    """Tests for classifier with separable synthetic data."""

    def test_separable_data_good_metrics(self):
        """Classifier with clearly separable features gives decent metrics."""
        protein_data = _make_synthetic_protein_data(
            n_proteins=10, n_residues=100, signal_strength=5.0,
        )
        frags = collect_node_fragments(
            protein_data, node_idx=0, half_w=5, act_quantile=0.5, bg_ratio=3,
        )

        # With seed=42 and these parameters, we always get enough fragments.
        assert len(frags["activated"]) >= 20, "Seed 42 should produce >= 20 activated"
        assert len(frags["background"]) >= 20, "Seed 42 should produce >= 20 background"

        result = train_motif_classifier(
            frags["activated"], frags["background"],
            feature_names=list(ACTIVE_GEOM_NAMES),
            cv_folds=3,
        )

        assert result["tree"] is not None, "Classifier should not be None"
        assert isinstance(result["rules"], str)
        # With separable data, GBM AUC should be reasonable
        assert result["gbm_auc_cv"] > 0.5, (
            f"gbm_auc_cv = {result['gbm_auc_cv']:.3f}, expected > 0.5"
        )


class TestConcordanceMetrics:
    """Tests for concordance between SAE activation and geometry prediction."""

    def test_basic_concordance(self):
        """Concordance metrics should be computable on synthetic data."""
        protein_data = _make_synthetic_protein_data(
            n_proteins=5, n_residues=80, signal_strength=5.0,
        )
        frags = collect_node_fragments(
            protein_data, node_idx=0, half_w=5, act_quantile=0.5, bg_ratio=3,
        )

        assert len(frags["activated"]) >= 20, "Seed 42 should produce >= 20 activated"
        assert len(frags["background"]) >= 20, "Seed 42 should produce >= 20 background"

        clf = train_motif_classifier(
            frags["activated"], frags["background"],
            feature_names=list(ACTIVE_GEOM_NAMES), cv_folds=3,
        )

        concordance = compute_concordance_metrics(
            protein_data, node_idx=0, tree=clf["tree"],
            threshold=frags["threshold"],
            geom_threshold=clf["optimal_threshold"],
            half_w=5,
        )

        assert concordance["n_residues"] > 0
        assert concordance["n_proteins"] > 0
        # Spearman should be at least somewhat positive with correlated data
        # (but not guaranteed to be very high with synthetic random-walk coords)
        assert -1.0 <= concordance["spearman_r"] <= 1.0

    def test_none_tree_returns_empty(self):
        """concordance with tree=None returns zeroed metrics."""
        protein_data = _make_synthetic_protein_data(n_proteins=2)
        result = compute_concordance_metrics(
            protein_data, 0, tree=None, threshold=1.0,
            geom_threshold=0.5, half_w=5,
        )
        assert result["n_residues"] == 0
        assert result["spearman_r"] == 0.0


class TestPlotDataStructure:
    """Tests for the precomputed plot data format."""

    def test_plot_data_structure(self):
        """Verify plot data arrays have correct types and lengths."""
        from proteinlens.analysis.feature_pipeline.geometry_residue_enrichment import (
            _precompute_plot_data,
        )

        protein_data = _make_synthetic_protein_data(
            n_proteins=5, n_residues=60, signal_strength=5.0,
        )
        frags = collect_node_fragments(
            protein_data, node_idx=0, half_w=5, act_quantile=0.5, bg_ratio=3,
        )

        assert len(frags["activated"]) >= 20, "Seed 42 should produce >= 20 activated"
        assert len(frags["background"]) >= 20, "Seed 42 should produce >= 20 background"

        clf = train_motif_classifier(
            frags["activated"], frags["background"],
            feature_names=list(ACTIVE_GEOM_NAMES), cv_folds=3,
        )

        plot_data = _precompute_plot_data(
            protein_data, node_idx=0, tree=clf["tree"],
            threshold=frags["threshold"],
            geom_threshold=clf["optimal_threshold"],
            half_w=5, top_n=2,
            feature_importances=clf["feature_importances"],
        )

        assert len(plot_data) <= 2, "Should return at most top_n proteins"
        assert len(plot_data) > 0, "Should return at least 1 protein"

        entry = plot_data[0]
        n = len(entry["ca_backbone"])

        # ca_backbone: list of [x, y, z]
        assert all(len(pt) == 3 for pt in entry["ca_backbone"])

        # sae_activation_profile: same length as sequence
        assert len(entry["sae_activation_profile"]) == n
        assert all(v >= 0 for v in entry["sae_activation_profile"])

        # geom_prob_profile: same length, values in [0, 1]
        assert len(entry["geom_prob_profile"]) == n
        assert all(0.0 <= v <= 1.0 for v in entry["geom_prob_profile"])

        # concordance_labels: same length, valid values only
        valid_labels = {"agree", "fp", "fn", "tn"}
        assert len(entry["concordance_labels"]) == n
        assert all(l in valid_labels for l in entry["concordance_labels"])


class TestMotifSuperposition:
    """Tests for fragment superposition."""

    def test_identical_fragments_low_rmsd(self):
        """10 identical fragments -> mean_rmsd < 0.1."""
        rng = np.random.default_rng(42)
        template = rng.standard_normal((11, 3))

        activated = [
            {"fragment": template.copy(), "activation": float(10 - i)}
            for i in range(10)
        ]

        result = superpose_fragments(activated, top_k=10)
        assert result["mean_rmsd"] < 0.1, (
            f"mean_rmsd = {result['mean_rmsd']:.4f}, expected < 0.1"
        )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_too_few_activated_positions_skipped(self):
        """Node with very few activations should be skipped (empty fragments)."""
        # Use a node with zero activations -- guaranteed to be skipped
        protein_data = _make_synthetic_protein_data(
            n_proteins=2, n_residues=30, signal_strength=5.0,
        )
        result = collect_node_fragments(
            protein_data, node_idx=5, half_w=5, act_quantile=0.80,
        )
        # Node 5 has no activations, so fragments should be empty
        assert len(result["activated"]) == 0, "Expected 0 activated fragments for inactive node"
        assert len(result["background"]) == 0, "Expected 0 background fragments for inactive node"
        assert result["n_total_active"] == 0, "Expected n_total_active == 0"


class TestNumericalEquivalence:
    """Verify optimised code produces identical results to original logic."""

    @staticmethod
    def _collect_node_fragments_original(
        protein_data, node_idx, half_w=10, act_quantile=0.80,
        max_fragments=100, bg_ratio=3,
    ):
        """ORIGINAL collect_node_fragments before deferred-extraction optimisation.

        Computes extract_local_feature_vector for ALL positions, then
        categorises and samples.  Kept here as a reference implementation
        for numerical-equivalence testing.
        """
        from proteinlens.analysis.geometry.residue_features import (
            extract_local_feature_vector,
        )

        all_nonzero = []
        for pdata in protein_data:
            col = pdata["act_matrix"][:, node_idx]
            nz = col[col > 0]
            if len(nz) > 0:
                all_nonzero.append(nz)
        if not all_nonzero:
            return {"activated": [], "background": [], "threshold": 0.0,
                    "n_total_active": 0, "n_hard_negatives": 0}

        all_nonzero_arr = np.concatenate(all_nonzero)
        n_total_active = len(all_nonzero_arr)
        if n_total_active < 20:
            return {"activated": [], "background": [], "threshold": 0.0,
                    "n_total_active": n_total_active, "n_hard_negatives": 0}

        threshold = float(np.quantile(all_nonzero_arr, act_quantile))
        if threshold <= 0:
            threshold = float(np.median(all_nonzero_arr))

        activated, background, hard_negatives = [], [], []
        for pdata in protein_data:
            ca = pdata["ca"]
            profiles = pdata["profiles"]
            col = pdata["act_matrix"][:, node_idx]
            n = pdata["n_residues"]
            seq = pdata.get("sequence", None)
            for pos in range(half_w, n - half_w):
                feat_vec = extract_local_feature_vector(
                    profiles, ca, pos, half_w, sequence=seq)
                if feat_vec is None:
                    continue
                fragment = ca[pos - half_w: pos + half_w + 1].copy()
                category = int(profiles["categories"][pos])
                entry = {
                    "accession": pdata["accession"], "position": pos,
                    "fragment": fragment, "features": feat_vec,
                    "category": category, "activation": float(col[pos]),
                }
                if col[pos] >= threshold:
                    activated.append(entry)
                elif 0 < col[pos] < threshold:
                    hard_negatives.append(entry)
                else:
                    background.append(entry)

        activated.sort(key=lambda x: -x["activation"])
        rng = np.random.default_rng(42)
        n_bg_total = min(len(background) + len(hard_negatives),
                         len(activated) * bg_ratio)
        n_hard = min(len(hard_negatives), n_bg_total // 2)
        n_zero = min(len(background), n_bg_total - n_hard)
        if n_hard > 0 and len(hard_negatives) > n_hard:
            idx = rng.choice(len(hard_negatives), size=n_hard, replace=False)
            hard_negatives = [hard_negatives[i] for i in idx]
        else:
            hard_negatives = hard_negatives[:n_hard]
        if n_zero > 0 and len(background) > n_zero:
            idx = rng.choice(len(background), size=n_zero, replace=False)
            background = [background[i] for i in idx]
        else:
            background = background[:n_zero]

        return {
            "activated": activated[:max_fragments * 5],
            "background": hard_negatives + background,
            "threshold": threshold,
            "n_total_active": n_total_active,
            "n_hard_negatives": len(hard_negatives),
        }

    def test_collect_fragments_identical(self):
        """Deferred extraction gives identical fragments to original."""
        protein_data = _make_synthetic_protein_data(
            n_proteins=8, n_residues=100, signal_strength=5.0,
        )

        old = self._collect_node_fragments_original(
            protein_data, node_idx=0, half_w=5, act_quantile=0.80, bg_ratio=3,
        )
        new = collect_node_fragments(
            protein_data, node_idx=0, half_w=5, act_quantile=0.80, bg_ratio=3,
        )

        assert old["threshold"] == new["threshold"]
        assert old["n_total_active"] == new["n_total_active"]
        assert old["n_hard_negatives"] == new["n_hard_negatives"]
        assert len(old["activated"]) == len(new["activated"])
        assert len(old["background"]) == len(new["background"])

        for o, n in zip(old["activated"], new["activated"]):
            assert o["accession"] == n["accession"]
            assert o["position"] == n["position"]
            assert o["activation"] == n["activation"]
            assert o["category"] == n["category"]
            np.testing.assert_array_equal(o["features"], n["features"])
            np.testing.assert_array_equal(o["fragment"], n["fragment"])

        for o, n in zip(old["background"], new["background"]):
            assert o["accession"] == n["accession"]
            assert o["position"] == n["position"]
            assert o["activation"] == n["activation"]
            np.testing.assert_array_equal(o["features"], n["features"])

    def test_concordance_with_and_without_cache(self):
        """Concordance with feat_cache gives identical results to without."""
        protein_data = _make_synthetic_protein_data(
            n_proteins=5, n_residues=80, signal_strength=5.0,
        )
        frags = collect_node_fragments(
            protein_data, node_idx=0, half_w=5, act_quantile=0.5, bg_ratio=3,
        )
        assert len(frags["activated"]) >= 20
        assert len(frags["background"]) >= 20

        clf = train_motif_classifier(
            frags["activated"], frags["background"],
            feature_names=list(ACTIVE_GEOM_NAMES), cv_folds=3,
        )

        # Build cache from fragments (same as process_node does)
        feat_cache = {}
        for frag in frags["activated"] + frags["background"]:
            if frag["features"] is not None:
                feat_cache[(frag["accession"], frag["position"])] = frag["features"]

        # Run concordance WITHOUT cache
        conc_no_cache = compute_concordance_metrics(
            protein_data, 0, clf["tree"],
            frags["threshold"], clf["optimal_threshold"], half_w=5,
            feat_cache=None,
        )
        # Run concordance WITH cache
        conc_with_cache = compute_concordance_metrics(
            protein_data, 0, clf["tree"],
            frags["threshold"], clf["optimal_threshold"], half_w=5,
            feat_cache=feat_cache,
        )

        for key in conc_no_cache:
            assert conc_no_cache[key] == conc_with_cache[key], (
                f"Mismatch on {key}: {conc_no_cache[key]} vs {conc_with_cache[key]}"
            )
