from __future__ import annotations

"""
Build a geometry ↔ SAE activation dataset using **multivariate linear
regression** instead of single-feature correlations.

For every SAE node the script fits a linear model:

    activation_i  ≈  w_1·g_1  +  w_2·g_2  +  …  +  w_F·g_F  +  b

where g_j are the F geometric features of protein i.  This lets us find
nodes whose activation is best explained by a *combination* of geometric
measures (a "geometric monomial") rather than any single one.

Pipeline overview:
  1.  Retrieve accessions (same sources as build_activation_dataset.py).
  2.  Load SAE + ESM; compute per-protein mean activations & geometry.
  3.  For each SAE node, fit a Ridge regression on active proteins.
  4.  Rank nodes by R² (or adjusted R²).  Save the weight vectors.
  5.  Plot: actual activation vs. linear-model prediction for the top nodes.
  6.  Save per-feature weight bar charts and summary YAML.

Usage:
    python build_activation_multiset.py --resume --max-proteins 1000
    python build_activation_multiset.py --mixed-organisms --max-proteins 5000

All heavy data (activation_matrix.npy, geometry_matrix.npy, pdb_cache/) is
**shared** with build_activation_dataset.py in the same --output-dir, so
--resume works across both scripts.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Re-use everything from the single-feature pipeline
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_activation_dataset import (
    # constants
    SAE_DIR, ESM_MODEL_NAME, ESM_LAYER, TOP_K_PER_NODE, MAX_SEQ_LEN,
    ALPHAFOLD_API_URL, UNIPROT_FASTA_URL, UNIPROT_SEARCH_URL,
    DIVERSE_ORGANISMS, GEOM_FEATURE_NAMES,
    # accession helpers
    fetch_mixed_organism_accessions,
    fetch_swissprot_accessions,
    load_accessions_from_file,
    load_accessions_from_fasta,
    # data helpers
    fetch_sequence,
    fetch_alphafold_pdb,
    get_protein_sequence_from_pdb,
    compute_activation,
    compute_geometry,
    find_top_k_per_node,
)
from kabsch_top_alignment import plot_kabsch_alignments
from proteinlens.sae.inference import load_sae
from proteinlens.embedders.esm import ESM
from proteinlens.utils import get_device
from domain_annotation import annotate_domains_from_pdb_cache


# =================== LINEAR REGRESSION ANALYSIS ============================

MIN_ACTIVE = 300  # need at least this many active proteins to fit a model


def fit_linear_regressors(
    geom_matrix: np.ndarray,
    act_matrix: np.ndarray,
    geom_names: list[str],
    alpha: float = 1.0,
    cv_folds: int = 5,
    top_n: int = 50,
) -> list[dict]:
    """
    For every SAE node, fit a LassoCV regression:
        activation ~ w · geometry + b
    on the subset of proteins that actually fire on that node.

    LassoCV automatically selects the best regularisation via cross-validation
    and drives irrelevant weights to exactly zero, yielding sparse monomials
    that generalise instead of memorising.

    Returns a list of dicts sorted by R²_cv descending, containing:
        sae_node, r2, r2_adj, r2_cv, pearson_r, weights, intercept,
        n_samples, n_nonzero, top_features.
    """
    n_prot, n_nodes = act_matrix.shape
    n_geom = geom_matrix.shape[1]

    # Pre-compute once (not per node)
    geom_valid = np.all(np.isfinite(geom_matrix), axis=1)

    results: list[dict] = []
    n_skipped = 0
    n_negative_cv = 0
    import time as _time
    _t0 = _time.time()

    for ni in range(n_nodes):
        if ni % 100 == 0:
            elapsed = _time.time() - _t0
            print(f"    node {ni:>5d}/{n_nodes}  "
                  f"kept={len(results)}  skipped={n_skipped}  "
                  f"neg_cv={n_negative_cv}  [{elapsed:.1f}s]")

        # Only fit on proteins that actually activate at this node
        active = act_matrix[:, ni] > 0
        mask = active & geom_valid

        if mask.sum() < MIN_ACTIVE:
            n_skipped += 1
            continue

        X = geom_matrix[mask]
        y = act_matrix[mask, ni]

        if y.std() < 1e-12:
            n_skipped += 1
            continue

        # Standardise features so weights are comparable
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        n = int(mask.sum())
        n_cv = min(cv_folds, n)

        # Fit LassoCV — picks best alpha AND gives CV scores internally
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

        # Count nonzero weights (the sparse "monomial" terms)
        n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-8))
        p = n_nonzero  # adjusted R² uses only the active features
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)

        # R²_cv from LassoCV's internal cross-validation (no extra call needed)
        best_alpha_idx = np.argmin(model.mse_path_.mean(axis=1))
        mse_cv = model.mse_path_[best_alpha_idx].mean()
        r2_cv = 1.0 - mse_cv / y.var() if y.var() > 0 else 0.0

        # Skip nodes that don't generalise at all
        if r2_cv < 0.0:
            n_negative_cv += 1
            continue

        # Pearson between actual and predicted
        pr, _ = stats.pearsonr(y, y_pred)

        # Standardised weights → tells which features matter most
        w = model.coef_
        order = np.argsort(np.abs(w))[::-1]
        # Only include nonzero features in top_features
        top_feats = [
            {"feature": geom_names[int(i)],
             "weight": float(w[i]),
             "abs_weight": float(abs(w[i]))}
            for i in order
            if abs(w[i]) > 1e-8
        ][:10]

        # Store un-standardised weights for actual prediction
        # y_pred = X_raw @ w_raw + b_raw
        w_raw = w / scaler.scale_
        b_raw = float(model.intercept_ - (w * scaler.mean_ / scaler.scale_).sum())

        results.append({
            "sae_node": int(ni),
            "r2": float(r2),
            "r2_adj": float(r2_adj),
            "r2_cv": float(r2_cv),
            "pearson_r": float(pr),
            "n_samples": n,
            "n_nonzero": n_nonzero,
            "alpha_chosen": float(model.alpha_),
            "intercept": float(model.intercept_),
            "weights_standardised": [float(x) for x in w],
            "weights_raw": [float(x) for x in w_raw],
            "intercept_raw": b_raw,
            "top_features": top_feats,
        })

    # Sort by cross-validated R² — the trustworthy metric
    results.sort(key=lambda d: d["r2_cv"], reverse=True)
    return results[:top_n]


# ======================== PLOTTING =========================================

def format_monomial(
    weights_raw: list[float],
    intercept_raw: float,
    geom_names: list[str],
    threshold: float = 1e-6,
    max_terms: int = 10,
) -> str:
    """
    Format the linear model as a human-readable monomial string:
        ŷ = 0.342·hairpin_score − 0.187·avg_curvature + 0.003

    Only includes terms with |weight| > threshold (important for Lasso
    where most weights are exactly zero).  Sorted by |weight| descending.
    """
    pairs = [
        (geom_names[i], w)
        for i, w in enumerate(weights_raw)
        if abs(w) > threshold
    ]
    # Sort by absolute weight descending
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    pairs = pairs[:max_terms]

    if not pairs:
        return f"ŷ = {intercept_raw:.4g}"

    terms = []
    for idx, (name, w) in enumerate(pairs):
        if idx == 0:
            terms.append(f"{w:.4g}·{name}")
        elif w < 0:
            terms.append(f"− {abs(w):.4g}·{name}")
        else:
            terms.append(f"+ {w:.4g}·{name}")

    # Intercept
    if abs(intercept_raw) > threshold:
        if intercept_raw < 0:
            terms.append(f"− {abs(intercept_raw):.4g}")
        else:
            terms.append(f"+ {intercept_raw:.4g}")

    return "ŷ = " + " ".join(terms)


COLOURS = ["#2980b9", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad",
           "#1abc9c", "#d35400", "#c0392b", "#2c3e50", "#16a085"]


def plot_pred_vs_actual(
    geom_matrix: np.ndarray,
    act_matrix: np.ndarray,
    results: list[dict],
    save_dir: Path,
    plots_per_figure: int = 6,
):
    """
    For each top node, scatter-plot actual activation vs. the linear-model
    predicted value  y_hat = X_raw @ w_raw + b_raw.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    k = len(results)
    if k == 0:
        return
    n_figures = (k + plots_per_figure - 1) // plots_per_figure

    for fig_idx in range(n_figures):
        start = fig_idx * plots_per_figure
        end = min(start + plots_per_figure, k)
        n_plots = end - start
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = np.atleast_2d(axes)

        for idx in range(n_plots):
            ax = axes.flat[idx]
            entry = results[start + idx]
            ni = entry["sae_node"]
            w_raw = np.array(entry["weights_raw"])
            b_raw = entry["intercept_raw"]

            active = act_matrix[:, ni] > 0
            geom_valid = np.all(np.isfinite(geom_matrix), axis=1)
            mask = active & geom_valid

            y = act_matrix[mask, ni]
            y_hat = geom_matrix[mask] @ w_raw + b_raw

            ax.scatter(y_hat, y, s=14, alpha=0.55, edgecolors="none",
                       color=COLOURS[idx % len(COLOURS)])

            # Perfect-prediction line
            lo = min(y_hat.min(), y.min())
            hi = max(y_hat.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)

            # Top-3 feature labels
            top3 = ", ".join(f["feature"] for f in entry["top_features"][:3])
            ax.set_xlabel("Linear prediction")
            ax.set_ylabel(f"SAE node {ni} act.")
            ax.set_title(
                f"Node {ni}  R²={entry['r2']:.3f}  "
                f"R²_cv={entry['r2_cv']:.3f}\n"
                f"top: {top3}",
                fontsize=9,
            )

        for idx in range(n_plots, rows * cols):
            axes.flat[idx].set_visible(False)

        plt.suptitle(
            f"Activation vs Linear Geometric Prediction  "
            f"(nodes #{start + 1}–#{end})",
            y=1.02, fontsize=12,
        )
        plt.tight_layout()
        path = save_dir / f"multivariate_scatter_{fig_idx + 1}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Scatter figure {fig_idx + 1} saved → {path}")
        plt.close()


def plot_weight_bars(
    results: list[dict],
    geom_names: list[str],
    save_dir: Path,
    top_n_nodes: int = 20,
):
    """
    For each top node, plot a horizontal bar chart of the standardised
    regression weights (which features contribute most to the linear combo).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for entry in results[:top_n_nodes]:
        ni = entry["sae_node"]
        w = np.array(entry["weights_standardised"])
        order = np.argsort(np.abs(w))[::-1][:20]

        fig, ax = plt.subplots(figsize=(10, 5))
        colours = ["#e74c3c" if w[i] < 0 else "#2980b9" for i in order]
        ax.barh(range(len(order)), w[order], color=colours)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([geom_names[i] for i in order], fontsize=9)
        ax.set_xlabel("Standardised weight")
        ax.set_title(
            f"Node {ni} — Linear regression weights  "
            f"(R²={entry['r2']:.3f}, R²_cv={entry['r2_cv']:.3f})",
            fontsize=10,
        )
        ax.invert_yaxis()
        plt.tight_layout()
        path = save_dir / f"weights_node{ni}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"  Weight bar charts saved → {save_dir}/")


def plot_r2_ranking(results: list[dict], save_path: Path):
    """Bar chart showing R² (train) and R²_cv for the top nodes."""
    nodes = [r["sae_node"] for r in results]
    r2s = [r["r2"] for r in results]
    r2_cvs = [r["r2_cv"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(nodes))
    w = 0.35
    ax.bar(x - w / 2, r2s, w, label="R² (train)", color="#2980b9", alpha=0.85)
    ax.bar(x + w / 2, r2_cvs, w, label="R² (CV)", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{n}" for n in nodes], rotation=70, fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("SAE Nodes Ranked by Multivariate Geometric R²")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  R² ranking bar chart saved → {save_path}")


# ========================= MAIN PIPELINE ===================================

def main():
    parser = argparse.ArgumentParser(
        description="Multivariate linear regression: geometry → SAE node activation."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--organism", type=int, default=None,
        help="UniProt taxonomy ID (9606=human, 10090=mouse, …)."
    )
    src.add_argument(
        "--mixed-organisms", action="store_true",
        help="Pull from ~20 diverse organisms and shuffle."
    )
    src.add_argument(
        "--fasta", type=Path, default=None,
        help="Path to a FASTA file."
    )
    src.add_argument(
        "--accession-list", type=Path, default=None,
        help="Plain text file with one accession per line."
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--per-organism-cap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_NODE)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "alphafold_analysis",
    )
    parser.add_argument("--sae-dir", type=Path, default=SAE_DIR)
    parser.add_argument("--esm-model", type=str, default=ESM_MODEL_NAME)
    parser.add_argument("--esm-layer", type=int, default=ESM_LAYER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Lasso regularisation strength (LassoCV auto-tunes, this is fallback)."
    )
    parser.add_argument(
        "--top-nodes", type=int, default=50,
        help="Number of best-fitting nodes to keep and plot."
    )
    parser.add_argument(
        "--skip-domains", action="store_true",
        help="Skip Pfam domain annotation (geometric features only)."
    )
    parser.add_argument(
        "--pfam-dir", type=Path, default=None,
        help="Directory for Pfam HMM database (default: <output-dir>/pfam)."
    )
    parser.add_argument(
        "--min-domain-freq", type=int, default=5,
        help="Min proteins a Pfam domain must appear in to be a feature."
    )
    parser.add_argument(
        "--domain-cpus", type=int, default=4,
        help="CPU threads for pyhmmer domain scanning."
    )
    args = parser.parse_args()

    if (args.organism is None and args.fasta is None
            and args.accession_list is None and not args.mixed_organisms):
        args.organism = 9606

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdb_cache = out / "pdb_cache"
    pdb_cache.mkdir(exist_ok=True)

    device = get_device()
    print("=" * 72)
    print("Multivariate Linear Regression: Geometry → SAE Activation")
    print("=" * 72)
    print(f"[✓] Device: {device}")
    print(f"[i] Output dir: {out}")
    print()

    # ── Step 1: Get accession list ────────────────────────────────────────
    fasta_seqs: dict[str, str] | None = None
    if args.mixed_organisms:
        accessions = fetch_mixed_organism_accessions(
            max_proteins=args.max_proteins,
            per_organism_cap=getattr(args, "per_organism_cap", None),
            seed=args.seed,
        )
    elif args.fasta:
        accessions, fasta_seqs = load_accessions_from_fasta(args.fasta)
        print(f"[1/7] Loaded {len(accessions)} proteins from FASTA: {args.fasta}")
    elif args.accession_list:
        accessions = load_accessions_from_file(args.accession_list)
        print(f"[1/7] Loaded {len(accessions)} accessions from: {args.accession_list}")
    else:
        accessions = fetch_swissprot_accessions(
            organism_taxid=args.organism,
            max_proteins=args.max_proteins,
        )

    if args.max_proteins:
        accessions = accessions[: args.max_proteins]

    (out / "accessions.txt").write_text("\n".join(accessions))
    print(f"  {len(accessions)} proteins to process.\n")

    # ── Step 2: Load SAE + ESM ────────────────────────────────────────────
    print("[2/7] Loading SAE …")
    sae = load_sae(args.sae_dir, device=device)
    sae.eval()
    print(f"  SAE: {sae.__class__.__name__}  dict_size={sae.dict_size}  "
          f"activation_dim={sae.activation_dim}")

    print("[2/7] Loading ESM embedder …")
    embedder = ESM(model_name=args.esm_model, device=device)
    print(f"  ESM: {args.esm_model}  layer {args.esm_layer}\n")

    # ── Step 3: Compute activations + geometry ────────────────────────────
    act_cache = out / "activation_matrix.npy"
    geom_cache = out / "geometry_matrix.npy"
    acc_cache = out / "processed_accessions.txt"

    if args.resume and act_cache.exists() and geom_cache.exists():
        print("[3/7] Loading cached activation matrix …")
        act_matrix = np.load(act_cache)
        accessions_ok = acc_cache.read_text().strip().split("\n")
        print(f"  {len(accessions_ok)} proteins loaded from cache.")

        # Recompute geometry from cached PDB files
        print("  Recomputing geometry from cached PDB files …\n")
        geom_rows: list[list[float]] = []
        act_rows_keep: list[int] = []
        n_geom_skip = 0
        for idx, acc in enumerate(accessions_ok):
            if idx % 500 == 0 or idx == len(accessions_ok) - 1:
                print(
                    f"    [{idx + 1:>6d}/{len(accessions_ok)}]  "
                    f"ok={len(geom_rows)}  skip={n_geom_skip}  current={acc}"
                )
            cached_files = list(pdb_cache.glob(f"AF-{acc}-F1-model_v*.pdb"))
            if not cached_files:
                n_geom_skip += 1
                continue
            pdb_text = cached_files[0].read_text()
            geom = compute_geometry(pdb_text)
            if geom is None:
                n_geom_skip += 1
                continue
            geom_rows.append([geom[k] for k in GEOM_FEATURE_NAMES])
            act_rows_keep.append(idx)

        if not geom_rows:
            print("\n✘ No geometry recomputed. Check PDB cache.")
            sys.exit(1)

        geom_matrix = np.array(geom_rows, dtype=float)
        act_matrix = act_matrix[act_rows_keep]
        accessions_ok = [accessions_ok[i] for i in act_rows_keep]

        np.save(geom_cache, geom_matrix)
        np.save(act_cache, act_matrix)
        acc_cache.write_text("\n".join(accessions_ok))
        print(f"\n  [✓] Geometry recomputed for {len(accessions_ok)} proteins, "
              f"{n_geom_skip} skipped.\n")
    else:
        print("[3/7] Computing activations & geometry for each protein …")
        print("  This fetches AlphaFold structures from the EBI API "
              "(cached on disk).\n")

        session = requests.Session()
        accessions_ok: list[str] = []
        act_rows: list[np.ndarray] = []
        geom_rows: list[list[float]] = []
        n_skip = 0

        for idx, acc in enumerate(accessions):
            if idx % 10 == 0 or idx == len(accessions) - 1:
                print(
                    f"  [{idx + 1:>6d}/{len(accessions)}]  "
                    f"ok={len(accessions_ok)}  skip={n_skip}  current={acc}"
                )

            if fasta_seqs and acc in fasta_seqs:
                seq = fasta_seqs[acc]
            else:
                seq = fetch_sequence(acc, session)
            if not seq or len(seq) < 4 or len(seq) > MAX_SEQ_LEN:
                n_skip += 1
                continue

            pdb_text = fetch_alphafold_pdb(acc, pdb_cache, session)
            if pdb_text is None:
                n_skip += 1
                continue

            act = compute_activation(sae, embedder, seq, device, args.esm_layer)
            if act is None:
                n_skip += 1
                continue

            geom = compute_geometry(pdb_text)
            if geom is None:
                n_skip += 1
                continue

            accessions_ok.append(acc)
            act_rows.append(act)
            geom_rows.append([geom[k] for k in GEOM_FEATURE_NAMES])

            if len(accessions_ok) % 500 == 0 and len(accessions_ok) > 0:
                print(f"    … checkpoint at {len(accessions_ok)} proteins")
                np.save(act_cache, np.vstack(act_rows))
                np.save(geom_cache, np.array(geom_rows, dtype=float))
                acc_cache.write_text("\n".join(accessions_ok))

        if not accessions_ok:
            print("\n✘ No proteins processed. Check network / accessions.")
            sys.exit(1)

        act_matrix = np.vstack(act_rows)
        geom_matrix = np.array(geom_rows, dtype=float)

        np.save(act_cache, act_matrix)
        np.save(geom_cache, geom_matrix)
        acc_cache.write_text("\n".join(accessions_ok))
        print(f"\n  [✓] {len(accessions_ok)} proteins processed, "
              f"{n_skip} skipped.\n")

    n_prot, n_nodes = act_matrix.shape
    n_geom = geom_matrix.shape[1]
    print(f"  Dataset: {n_prot} proteins × {n_geom} geom features × "
          f"{n_nodes} SAE nodes\n")

    # ── Step 3b: Pfam domain annotation ───────────────────────────────────
    if not args.skip_domains:
        print("[3b] Annotating Pfam domains with pyhmmer + biotite …")
        pfam_dir = args.pfam_dir or out / "pfam"
        dom_cache = out / "domain_matrix.npy"
        dom_names_cache = out / "domain_names.json"

        if args.resume and dom_cache.exists() and dom_names_cache.exists():
            print("  Loading cached domain matrix …")
            domain_matrix = np.load(dom_cache)
            with open(dom_names_cache) as _f:
                domain_names = json.load(_f)
            print(f"  {domain_matrix.shape[1]} domain features loaded "
                  f"from cache.")
        else:
            domain_matrix, domain_names, domain_counts = (
                annotate_domains_from_pdb_cache(
                    accessions_ok, pdb_cache, pfam_dir,
                    cpus=args.domain_cpus,
                    min_freq=args.min_domain_freq,
                )
            )
            np.save(dom_cache, domain_matrix)
            with open(dom_names_cache, "w") as _f:
                json.dump(domain_names, _f)
            # Save readable domain counts for inspection
            sparse_counts = {
                acc: counts
                for acc, counts in domain_counts.items()
                if counts
            }
            dom_yaml = out / "domain_counts.yaml"
            with open(dom_yaml, "w") as _f:
                yaml.dump(sparse_counts, _f, default_flow_style=False)
            print(f"  Domain counts saved → {dom_yaml}")

        n_dom = domain_matrix.shape[1]
        if n_dom > 0:
            combined_matrix = np.hstack([geom_matrix, domain_matrix])
            combined_names = list(GEOM_FEATURE_NAMES) + [
                f"pfam_{d}" for d in domain_names
            ]
            print(f"  Combined features: {combined_matrix.shape[1]} "
                  f"({n_geom} geometric + {n_dom} Pfam domains)\n")
        else:
            combined_matrix = geom_matrix.copy()
            combined_names = list(GEOM_FEATURE_NAMES)
            print("  No domain features passed filtering — "
                  "using geometric features only.\n")
    else:
        combined_matrix = geom_matrix.copy()
        combined_names = list(GEOM_FEATURE_NAMES)
        print("[3b] Skipping domain annotation (--skip-domains).\n")

    # ── Step 4: Top-K per node ────────────────────────────────────────────
    print(f"[4/7] Finding top-{args.top_k} activating proteins per SAE node …")
    top_k_map = find_top_k_per_node(accessions_ok, act_matrix, k=args.top_k)
    top_k_path = out / "top_activating_per_node.yaml"
    with open(top_k_path, "w") as f:
        yaml.dump(top_k_map, f, default_flow_style=False)
    n_active = sum(1 for v in top_k_map.values() if v)
    print(f"  {n_active}/{n_nodes} nodes have at least one activating protein.")
    print(f"  Saved → {top_k_path}\n")

    # ── Step 5: Multivariate linear regression ────────────────────────────
    print(f"[5/7] Fitting LassoCV regression for each node …")
    results = fit_linear_regressors(
        combined_matrix, act_matrix, combined_names,
        alpha=args.alpha, top_n=args.top_nodes,
    )
    print(f"  [✓] {len(results)} nodes fitted with R²_cv > 0 "
          f"(of {n_nodes} total).\n")

    # Print top
    TOP_PRINT = 40
    print("=" * 110)
    print(f"Top {min(TOP_PRINT, len(results))} nodes by R²_cv "
          f"(multivariate geometry → activation)")
    print("=" * 110)
    print(
        f"{'Node':>6s} {'R²':>8s} {'R²_adj':>8s} {'R²_cv':>8s} "
        f"{'r':>8s} {'N':>6s} {'NZ':>4s}   Monomial"
    )
    print("-" * 110)
    for entry in results[:TOP_PRINT]:
        monomial = format_monomial(
            entry["weights_raw"], entry["intercept_raw"],
            combined_names, max_terms=5,
        )
        print(
            f"{entry['sae_node']:>6d} {entry['r2']:>8.4f} "
            f"{entry['r2_adj']:>8.4f} {entry['r2_cv']:>8.4f} "
            f"{entry['pearson_r']:>8.4f} {entry['n_samples']:>6d} "
            f"{entry['n_nonzero']:>4d}   {monomial}"
        )

    # Save summary YAML (full weight vectors are large, save top features only)
    summary_for_yaml = []
    for entry in results:
        monomial = format_monomial(
            entry["weights_raw"], entry["intercept_raw"],
            combined_names, max_terms=10,
        )
        summary_for_yaml.append({
            "sae_node": entry["sae_node"],
            "r2": round(entry["r2"], 5),
            "r2_adj": round(entry["r2_adj"], 5),
            "r2_cv": round(entry["r2_cv"], 5),
            "pearson_r": round(entry["pearson_r"], 5),
            "n_samples": entry["n_samples"],
            "n_nonzero": entry["n_nonzero"],
            "alpha_chosen": round(entry["alpha_chosen"], 8),
            "monomial": monomial,
            "top_features": entry["top_features"],
        })
    corr_path = out / "multivariate_regression.yaml"
    with open(corr_path, "w") as f:
        yaml.dump(summary_for_yaml, f, default_flow_style=False)
    print(f"\n  Full summary → {corr_path}\n")

    # Also save raw weight matrices for downstream use
    weights_path = out / "regression_weights.npz"
    node_ids = np.array([r["sae_node"] for r in results])
    w_std_mat = np.array([r["weights_standardised"] for r in results])
    w_raw_mat = np.array([r["weights_raw"] for r in results])
    b_raw_vec = np.array([r["intercept_raw"] for r in results])
    np.savez(weights_path, node_ids=node_ids, weights_std=w_std_mat,
             weights_raw=w_raw_mat, intercepts_raw=b_raw_vec,
             feature_names=np.array(combined_names))
    print(f"  Raw weight matrices → {weights_path}\n")

    # ── Step 6: Plots ─────────────────────────────────────────────────────
    print("[6/7] Generating plots …")

    # 6a. Actual vs predicted scatter
    plot_pred_vs_actual(
        combined_matrix, act_matrix, results,
        save_dir=out, plots_per_figure=6,
    )

    # 6b. Weight bar charts for top nodes
    weight_dir = out / "regression_weights"
    plot_weight_bars(results, combined_names, save_dir=weight_dir)

    # 6c. R² ranking overview
    plot_r2_ranking(results, save_path=out / "r2_ranking.png")

    # ── Step 7: Kabsch-aligned backbone overlays ─────────────────────────
    # Build a pseudo-summary compatible with plot_kabsch_alignments
    # (one entry per node, using top features as labels)
    print("\n[7/7] Plotting Kabsch-aligned backbone overlays …")
    kabsch_dir = out / "kabsch_overlays"
    pseudo_summary = []
    for entry in results:
        top_feat = entry["top_features"][0]["feature"] if entry["top_features"] else "combo"
        pseudo_summary.append({
            "geom_feature": f"node{entry['sae_node']}_{top_feat}",
            "sae_node": entry["sae_node"],
            "pearson_r": entry["pearson_r"],
            "spearman_r": entry.get("pearson_r", 0.0),  # reuse r as placeholder
        })
    plot_kabsch_alignments(
        summary=pseudo_summary,
        top_k_map=top_k_map,
        pdb_cache=pdb_cache,
        save_dir=kabsch_dir,
        n_proteins=5,
    )

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("✅  Multivariate pipeline complete!")
    print("=" * 72)
    print(f"All outputs in: {out}/")
    print(f"  • activation_matrix.npy           – ({n_prot}, {n_nodes})")
    n_total_feats = combined_matrix.shape[1]
    n_dom_feats = n_total_feats - n_geom
    print(f"  • geometry_matrix.npy             – ({n_prot}, {n_geom})")
    if n_dom_feats > 0:
        print(f"  • domain_matrix.npy               – ({n_prot}, {n_dom_feats})")
        print(f"  • domain_counts.yaml              – per-protein Pfam counts")
        print(f"  • domain_names.json               – {n_dom_feats} domain names")
    print(f"  • multivariate_regression.yaml    – top {len(results)} nodes")
    print(f"  • regression_weights.npz          – raw weight matrices")
    print(f"  • regression_weights/             – per-node weight bar charts")
    print(f"  • multivariate_scatter_*.png      – actual vs predicted plots")
    print(f"  • r2_ranking.png                  – R² ranking overview")
    print(f"  • kabsch_overlays/                – backbone overlays")
    print("=" * 72)


if __name__ == "__main__":
    main()
