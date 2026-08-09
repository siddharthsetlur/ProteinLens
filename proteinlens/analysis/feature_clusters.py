"""
proteinlens/analysis/feature_clusters.py
=========================================

Spectral clustering of SAE decoder vectors.

Typical workflow
----------------
1. Build clusters from a trained SAE (slow – runs spectral clustering):

    from proteinlens.sae.inference import load_sae
    from proteinlens.analysis.feature_clusters import FeatureClusters

    sae = load_sae("trained_models/fiery-sweep")
    fc = FeatureClusters.from_sae(sae, n_clusters=500)
    fc.save("clusters.yaml")

2. Load pre-computed clusters and query them:

    fc = FeatureClusters.from_file("clusters.yaml")

    # Which features belong to cluster 7?
    feats = fc.get_features(7)

    # Which cluster does feature 42 belong to?
    c = fc.get_cluster(42)

    # Top proteins across all features in cluster 7:
    import yaml
    max_ex = yaml.safe_load(open("protein_results/Per_feature_max_examples.yaml"))
    proteins = fc.get_top_proteins(7, max_ex, n_per_feature=5)

3. Build FeatureIntervention objects for use with intervene_and_fold.py:

    from scripts.intervene_and_fold import FeatureIntervention   # or import directly
    ivs = fc.make_interventions(7, action="zero")

YAML output format
------------------
  0:
    - 42
    - 1337
  1:
    - 7
    - 99
  ...
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.cluster import SpectralClustering


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_decoder_weights(sae) -> torch.Tensor:
    """Return W_dec as a float32 CPU tensor of shape (dict_size, activation_dim).

    Handles:
    * ReLUSAE         – decoder weights live in sae.decoder.weight
    * MatryoshkaBatchTopKSAE – weights live in sae.W_dec
    """
    if hasattr(sae, "W_dec"):
        return sae.W_dec.detach().cpu().float()
    if hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
        # nn.Linear stores weight as (out_features, in_features) = (activation_dim, dict_size).
        # Transpose so rows correspond to features: (dict_size, activation_dim).
        return sae.decoder.weight.T.detach().cpu().float()
    raise AttributeError(
        f"Cannot locate decoder weights on SAE of type {type(sae).__name__}. "
        "Expected attribute 'W_dec' or 'decoder.weight'."
    )


def _spectral_cluster(
    W_dec: torch.Tensor,
    n_clusters: int,
    chunk_size: int = 1024,
    verbose: bool = True,
) -> dict[int, list[int]]:
    """Run spectral clustering on decoder weight vectors.

    Uses angular (cosine) similarity: sim = 1 - arccos(cos) / π,
    so identical directions → 1, orthogonal → 0.5, opposite → 0.

    Args:
        W_dec:      (dict_size, activation_dim) float32 tensor.
        n_clusters: Number of clusters for SpectralClustering.
        chunk_size: Row-chunk size for building the similarity matrix
                    (controls peak memory).
        verbose:    Print progress.

    Returns:
        dict mapping cluster_idx → sorted list of feature indices.
    """
    n = W_dec.shape[0]
    W_norm = F.normalize(W_dec, dim=1)  # (n, dim), unit rows

    # NOTE: The full (n, n) float32 matrix is always allocated here.
    # For n=5120 this is ~100 MB. For n>>10k this will require significant RAM.
    # The chunk_size parameter only reduces intermediate GPU/CPU tensors, not this allocation.
    all_sims = np.zeros((n, n), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = W_norm[start:end] @ W_norm.T          # (chunk, n)
        sims = torch.clamp(sims, -1.0, 1.0)
        sims = 1.0 - torch.arccos(sims) / torch.pi  # angular similarity
        all_sims[start:end] = sims.cpu().numpy()
        if verbose:
            print(f"  Similarity matrix: rows {start}–{end} / {n}", flush=True)

    np.fill_diagonal(all_sims, 0.0)

    if verbose:
        print(f"  Running SpectralClustering (n_clusters={n_clusters}) …", flush=True)

    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=0)
    labels = sc.fit_predict(all_sims)

    clusters: dict[int, list[int]] = {i: [] for i in range(n_clusters)}
    for feat_idx, label in enumerate(labels.tolist()):
        clusters[int(label)].append(feat_idx)

    if verbose:
        sizes = [len(v) for v in clusters.values()]
        print(
            f"  Done. {n_clusters} clusters, "
            f"sizes: min={min(sizes)} mean={np.mean(sizes):.1f} max={max(sizes)}",
            flush=True,
        )

    return clusters


# ─────────────────────────────────────────────────────────────────────────────
#  FeatureClusters
# ─────────────────────────────────────────────────────────────────────────────

class FeatureClusters:
    """Container for SAE feature clusters with query and intervention helpers.

    Internal state
    --------------
    _clusters : dict[int, list[int]]
        cluster_idx → sorted list of feature indices
    _feature_to_cluster : dict[int, int]
        feature_idx → cluster_idx  (reverse map for O(1) look-ups)
    """

    def __init__(self, clusters: dict[int, list[int]]):
        self._clusters: dict[int, list[int]] = {
            int(k): sorted(v) for k, v in clusters.items()
        }
        self._feature_to_cluster: dict[int, int] = {
            feat: cid for cid, feats in self._clusters.items() for feat in feats
        }

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_sae(
        cls,
        sae,
        n_clusters: int,
        chunk_size: int = 1024,
        verbose: bool = True,
    ) -> "FeatureClusters":
        """Build clusters by running spectral clustering on W_dec.

        Args:
            sae:        Loaded SAE model (ReLUSAE or Matryoshka).
            n_clusters: Number of clusters.
            chunk_size: Chunk size for similarity matrix construction.
            verbose:    Print progress.
        """
        W_dec = _get_decoder_weights(sae)
        if verbose:
            print(f"Clustering {W_dec.shape[0]} features into {n_clusters} clusters …")
        clusters = _spectral_cluster(W_dec, n_clusters, chunk_size, verbose)
        return cls(clusters)

    @classmethod
    def from_file(cls, path) -> "FeatureClusters":
        """Load from a YAML file produced by :meth:`save`.

        YAML format: ``{cluster_idx: [feature_idx, ...], ...}``
        """
        path = Path(path)
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls({int(k): list(v) for k, v in data.items()})

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path):
        """Save clusters to a YAML file.

        Format: ``{cluster_idx: [feature_idx, ...], ...}``
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: self._clusters[k] for k in sorted(self._clusters)}
        with open(path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=True)
        print(f"Saved {len(self)} clusters to {path}")

    # ── Queries ───────────────────────────────────────────────────────────

    def get_features(self, cluster_idx: int) -> list[int]:
        """Return the (sorted) list of feature indices in *cluster_idx*."""
        try:
            return list(self._clusters[int(cluster_idx)])
        except KeyError:
            raise KeyError(
                f"cluster_idx={cluster_idx} not found. "
                f"Valid range: 0–{len(self) - 1}."
            )

    def get_cluster(self, feature_idx: int) -> int:
        """Return the cluster index that *feature_idx* belongs to."""
        try:
            return self._feature_to_cluster[int(feature_idx)]
        except KeyError:
            raise KeyError(
                f"feature_idx={feature_idx} not found in any cluster."
            )

    def get_top_proteins(
        self,
        cluster_idx: int,
        max_examples: dict,
        n_per_feature: int = 5,
    ) -> list[str]:
        """Return the top proteins for a cluster.

        For each feature in the cluster, up to *n_per_feature* proteins are
        taken from *max_examples* (the loaded Per_feature_max_examples.yaml).
        Proteins are ranked by how many cluster features they appear in
        (descending), with ties broken by their earliest appearance order.

        Args:
            cluster_idx:   Which cluster to query.
            max_examples:  Dict loaded from Per_feature_max_examples.yaml.
                           Keys may be int or str feature IDs.
            n_per_feature: How many top proteins to take per feature.

        Returns:
            Deduplicated list of protein IDs, sorted by cluster-wide
            support (most features → first).
        """
        features = self.get_features(cluster_idx)
        vote_counter: Counter = Counter()

        for feat in features:
            # YAML keys may be stored as int or str; use explicit None check so that
            # an existing empty list [] does not fall through to the string-key lookup.
            val = max_examples.get(feat)
            if val is None:
                val = max_examples.get(str(feat))
            proteins = val or []
            for prot in proteins[:n_per_feature]:
                vote_counter[prot] += 1

        # Sort by descending vote count; within ties preserve insertion order
        return [prot for prot, _ in vote_counter.most_common()]

    # ── Intervention helpers ──────────────────────────────────────────────

    def make_interventions(
        self,
        cluster_idx: int,
        action: str,
        value: float = 1.0,
        positions: Optional[List[int]] = None,
    ) -> list:
        """Build a list of FeatureIntervention objects for every feature in the cluster.

        This avoids importing scripts.intervene_and_fold at module load time;
        the import happens lazily inside this method.

        Args:
            cluster_idx: Cluster to intervene on.
            action:      "scale" | "set" | "zero" | "add".
            value:       Scalar operand (ignored for "zero").
            positions:   0-indexed residue positions, or None for all.

        Returns:
            List of FeatureIntervention objects, one per feature in the cluster.
        """
        # Lazy import so the module works without the full script dependencies.
        try:
            import sys
            from pathlib import Path as _Path
            _scripts = str(_Path(__file__).resolve().parent.parent.parent / "scripts")
            if _scripts not in sys.path:
                sys.path.insert(0, _scripts)
            from intervene_and_fold import FeatureIntervention
        except ImportError as exc:
            raise ImportError(
                "Could not import FeatureIntervention from scripts/intervene_and_fold.py. "
                "Ensure the scripts/ directory is on your PYTHONPATH."
            ) from exc

        features = self.get_features(cluster_idx)
        return [
            FeatureIntervention(
                feature_idx=f,
                action=action,
                value=value,
                positions=list(positions) if positions is not None else None,
            )
            for f in features
        ]

    # ── Dunder helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of clusters."""
        return len(self._clusters)

    def n_features(self) -> int:
        """Total number of features across all clusters."""
        return len(self._feature_to_cluster)

    def __repr__(self) -> str:
        sizes = [len(v) for v in self._clusters.values()]
        return (
            f"FeatureClusters("
            f"n_clusters={len(self)}, "
            f"n_features={self.n_features()}, "
            f"cluster_sizes=[min={min(sizes)}, "
            f"mean={np.mean(sizes):.1f}, "
            f"max={max(sizes)}])"
        )
