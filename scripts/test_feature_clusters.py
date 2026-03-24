#!/usr/bin/env python
"""
Tests for FeatureClusters and cluster-based interventions.

Covers the tests recommended in code-review.md (2026-03-10):
  - YAML round-trip save/load
  - get_features / get_cluster inverse property
  - get_top_proteins empty-list edge case (the falsy-lookup bug)
  - Smoke-test of _spectral_cluster on a tiny dummy SAE
  - _get_decoder_weights returns (dict_size, activation_dim) for ReLUSAE

Plus an end-to-end equivalence test:
  - Intervening on a cluster_idx with a given action must produce the same
    ESM2 logits as intervening on each component feature_idx individually
    with that same action.

Usage:
  conda activate interplm
  python scripts/test_feature_clusters.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proteinlens.sae.dictionary import ReLUSAE
from proteinlens.analysis.feature_clusters import (
    FeatureClusters,
    _get_decoder_weights,
    _spectral_cluster,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: FeatureClusters
# ─────────────────────────────────────────────────────────────────────────────

def test_yaml_roundtrip():
    """Save clusters to YAML, reload, and check equality."""
    clusters = {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7, 8, 9]}
    fc = FeatureClusters(clusters)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        path = tmp.name

    fc.save(path)
    fc2 = FeatureClusters.from_file(path)

    check(
        "yaml_roundtrip:same_clusters",
        fc._clusters == fc2._clusters,
        f"expected {fc._clusters}, got {fc2._clusters}",
    )
    check(
        "yaml_roundtrip:same_reverse_map",
        fc._feature_to_cluster == fc2._feature_to_cluster,
    )
    Path(path).unlink()


def test_get_features_get_cluster_inverse():
    """get_cluster(f) == c for every f in get_features(c), and vice versa."""
    clusters = {0: [10, 20, 30], 1: [40, 50], 2: [60]}
    fc = FeatureClusters(clusters)

    all_ok = True
    for cid in clusters:
        for feat in fc.get_features(cid):
            if fc.get_cluster(feat) != cid:
                all_ok = False

    check("inverse_property:forward_then_reverse", all_ok)

    # Also check that every feature in the reverse map is in the right cluster
    all_ok2 = True
    for feat, cid in fc._feature_to_cluster.items():
        if feat not in fc.get_features(cid):
            all_ok2 = False

    check("inverse_property:reverse_then_forward", all_ok2)


def test_get_features_unknown_cluster():
    """get_features raises KeyError for unknown cluster."""
    fc = FeatureClusters({0: [1, 2]})
    raised = False
    try:
        fc.get_features(99)
    except KeyError:
        raised = True
    check("get_features:unknown_cluster_raises", raised)


def test_get_cluster_unknown_feature():
    """get_cluster raises KeyError for unknown feature."""
    fc = FeatureClusters({0: [1, 2]})
    raised = False
    try:
        fc.get_cluster(99)
    except KeyError:
        raised = True
    check("get_cluster:unknown_feature_raises", raised)


def test_len_and_n_features():
    """__len__ returns cluster count; n_features returns total features."""
    clusters = {0: [0, 1], 1: [2, 3, 4], 2: [5]}
    fc = FeatureClusters(clusters)
    check("len", len(fc) == 3)
    check("n_features", fc.n_features() == 6)


def test_get_top_proteins_basic():
    """get_top_proteins aggregates across features and ranks by vote count."""
    clusters = {0: [0, 1, 2]}
    fc = FeatureClusters(clusters)

    max_examples = {
        0: ["P1", "P2", "P3"],
        1: ["P2", "P3", "P4"],
        2: ["P3", "P5"],
    }

    top = fc.get_top_proteins(0, max_examples, n_per_feature=3)

    # P3 appears in all 3 features → top; P2 appears in 2 → second
    check("top_proteins:P3_first", top[0] == "P3", f"got {top[0]}")
    check("top_proteins:P2_second", top[1] == "P2", f"got {top[1]}")
    check(
        "top_proteins:all_present",
        set(top) == {"P1", "P2", "P3", "P4", "P5"},
        f"got {set(top)}",
    )


def test_get_top_proteins_empty_list_edge_case():
    """An empty list [] for an int key must NOT fall through to a str key.

    This was the falsy-lookup bug flagged in the code review. The fix uses
    an explicit None check, so max_examples[0]=[] should yield nothing for
    feature 0, even if max_examples['0'] exists with different proteins.
    """
    clusters = {0: [0]}
    fc = FeatureClusters(clusters)

    max_examples = {
        0: [],            # int key → empty list (feature 0 has no top proteins)
        "0": ["WRONG"],   # str key → should NOT be reached
    }

    top = fc.get_top_proteins(0, max_examples, n_per_feature=5)
    check(
        "empty_list_edge_case:no_fallthrough",
        "WRONG" not in top,
        f"got {top} — the falsy-lookup bug is present",
    )
    check("empty_list_edge_case:empty_result", len(top) == 0, f"got {top}")


def test_get_top_proteins_string_keys():
    """When max_examples only has string keys, the str fallback works."""
    clusters = {0: [0, 1]}
    fc = FeatureClusters(clusters)

    max_examples = {
        "0": ["PA", "PB"],
        "1": ["PB", "PC"],
    }

    top = fc.get_top_proteins(0, max_examples, n_per_feature=5)
    check("string_keys:PB_first", top[0] == "PB", f"got {top}")
    check(
        "string_keys:all_present",
        set(top) == {"PA", "PB", "PC"},
        f"got {set(top)}",
    )


def test_get_top_proteins_n_per_feature_limit():
    """n_per_feature truncates the per-feature protein list."""
    clusters = {0: [0]}
    fc = FeatureClusters(clusters)

    max_examples = {0: ["P1", "P2", "P3", "P4", "P5"]}
    top = fc.get_top_proteins(0, max_examples, n_per_feature=2)
    check("n_per_feature:truncated", len(top) == 2, f"got {len(top)}")
    check("n_per_feature:correct_order", top == ["P1", "P2"], f"got {top}")


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: _get_decoder_weights
# ─────────────────────────────────────────────────────────────────────────────

def test_get_decoder_weights_relu_sae():
    """_get_decoder_weights returns (dict_size, activation_dim) for ReLUSAE.

    ReLUSAE.decoder is nn.Linear(dict_size, activation_dim), so
    decoder.weight.shape == (activation_dim, dict_size). The function must
    transpose it to (dict_size, activation_dim).
    """
    activation_dim = 32
    dict_size = 128
    sae = ReLUSAE(activation_dim, dict_size)

    W = _get_decoder_weights(sae)
    check(
        "decoder_weights_relu:shape",
        W.shape == (dict_size, activation_dim),
        f"expected ({dict_size}, {activation_dim}), got {tuple(W.shape)}",
    )
    check("decoder_weights_relu:dtype", W.dtype == torch.float32)
    check("decoder_weights_relu:no_grad", not W.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke Test: _spectral_cluster
# ─────────────────────────────────────────────────────────────────────────────

def test_spectral_cluster_smoke():
    """Run spectral clustering on a tiny set of synthetic feature vectors.

    Create 3 tight groups of 4 vectors each (12 total), cluster into 3,
    and check that the groups are recovered.
    """
    torch.manual_seed(42)
    n_groups = 3
    n_per = 4
    dim = 16

    # Create 3 well-separated directions
    centers = torch.randn(n_groups, dim)
    centers = torch.nn.functional.normalize(centers, dim=1)

    # Add small noise around each center
    features = []
    true_labels = []
    for g in range(n_groups):
        for _ in range(n_per):
            v = centers[g] + 0.05 * torch.randn(dim)
            features.append(v)
            true_labels.append(g)

    W_dec = torch.stack(features)
    clusters = _spectral_cluster(W_dec, n_clusters=3, chunk_size=4, verbose=False)

    # Check: all features assigned
    all_feats = sorted(f for feats in clusters.values() for f in feats)
    check("spectral_smoke:all_assigned", all_feats == list(range(12)))

    # Check: features from the same true group end up in the same cluster
    # (cluster labels may be permuted vs true labels)
    groups_recovered = True
    for g in range(n_groups):
        group_feats = [i for i in range(12) if true_labels[i] == g]
        # All should share the same cluster
        assigned_clusters = set()
        for feat in group_feats:
            for cid, flist in clusters.items():
                if feat in flist:
                    assigned_clusters.add(cid)
        if len(assigned_clusters) != 1:
            groups_recovered = False

    check("spectral_smoke:groups_recovered", groups_recovered)


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: make_interventions
# ─────────────────────────────────────────────────────────────────────────────

def test_make_interventions():
    """make_interventions creates one FeatureIntervention per feature."""
    clusters = {0: [5, 10, 15], 1: [20]}
    fc = FeatureClusters(clusters)

    ivs = fc.make_interventions(0, action="scale", value=2.0, positions=[1, 2])

    check("make_interventions:count", len(ivs) == 3)

    feat_idxs = [iv.feature_idx for iv in ivs]
    check("make_interventions:features", sorted(feat_idxs) == [5, 10, 15])

    all_scale = all(iv.action == "scale" for iv in ivs)
    check("make_interventions:action", all_scale)

    all_val = all(iv.value == 2.0 for iv in ivs)
    check("make_interventions:value", all_val)

    all_pos = all(iv.positions == [1, 2] for iv in ivs)
    check("make_interventions:positions", all_pos)


def test_make_interventions_zero():
    """make_interventions with action='zero' has value=1.0 (doesn't matter)."""
    fc = FeatureClusters({0: [3]})
    ivs = fc.make_interventions(0, action="zero")
    check("make_interventions_zero:action", ivs[0].action == "zero")


# ─────────────────────────────────────────────────────────────────────────────
#  End-to-End: Cluster intervention == component feature interventions
# ─────────────────────────────────────────────────────────────────────────────

def _run_esm_from_layer(esm_model, modified_hidden, token_ids, attn_mask, from_layer):
    """Run ESM2 layers [from_layer, …, end] → LayerNorm → lm_head → logits.

    This replicates what NNsight's trace-and-patch does, but without the
    FakeTensor machinery (which is broken on PyTorch 2.9+).  We feed the
    modified hidden states directly into the remaining transformer layers.
    """
    with torch.no_grad():
        x = modified_hidden
        ext_mask = esm_model.esm.get_extended_attention_mask(
            attn_mask, token_ids.shape,
        )
        for layer_module in esm_model.esm.encoder.layer[from_layer:]:
            layer_out = layer_module(x, ext_mask)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        x = esm_model.esm.encoder.emb_layer_norm_after(x)
        x = esm_model.lm_head(x)
    return x


def test_cluster_vs_individual_interventions():
    """Cluster-level intervention must produce identical ESM2 outputs to
    applying the same action to each component feature individually.

    For each action type (scale, zero, set, add), we:
      1. Encode a sequence through ESM2 layer 3 → SAE features
      2. Apply the intervention cluster-wide (via make_interventions)
      3. Apply the same intervention feature-by-feature (via FeatureIntervention)
      4. Assert the modified feature tensors are identical
      5. Decode both through the SAE → modified hidden states; assert identical
      6. Run both through ESM2 layers 3–5 + lm_head; assert logits are identical
    """
    from proteinlens.sae.inference import load_sae
    from proteinlens.utils import get_device
    from scripts.intervene_and_fold import (
        FeatureIntervention,
        encode_with_sae,
        decode_and_build_hidden,
    )
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = get_device()
    esm_model_name = "facebook/esm2_t6_8M_UR50D"
    sae_dir = Path(__file__).resolve().parent.parent / "trained_models" / "fiery-sweep"
    layer = 3

    if not sae_dir.exists():
        print("  SKIP  cluster_vs_individual (SAE not found at", sae_dir, ")")
        return

    print("  Loading models …")
    tokenizer = AutoTokenizer.from_pretrained(
        esm_model_name, clean_up_tokenization_spaces=True,
    )
    esm_model = EsmForMaskedLM.from_pretrained(esm_model_name).to(device)
    esm_model.eval()
    sae = load_sae(sae_dir, device=device)
    sae.eval()

    # Short test sequence (lysozyme fragment)
    sequence = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"
    seq_len = len(sequence)

    # Run ESM forward to get hidden states at layer 3
    inputs = tokenizer(sequence, return_tensors="pt", padding=False)
    token_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = esm_model(
            token_ids, attention_mask=attn_mask, output_hidden_states=True,
        )
    orig_hidden = outputs.hidden_states[layer]  # (1, seq_len+2, d_model)

    # Encode with SAE
    features, original_norms = encode_with_sae(sae, orig_hidden, seq_len)

    # Build a test cluster from features that are actually active
    active_mask = features.sum(dim=0) > 0
    active_idxs = torch.where(active_mask)[0].tolist()
    cluster_feats = active_idxs[:5]
    if len(cluster_feats) < 2:
        print("  SKIP  cluster_vs_individual (not enough active features)")
        return
    print(f"  Test cluster: features {cluster_feats}")

    test_cluster = {0: cluster_feats}
    fc = FeatureClusters(test_cluster)

    actions = [
        ("scale", 2.5),
        ("zero", 0.0),
        ("set", 3.0),
        ("add", 1.5),
    ]

    for action, value in actions:
        # --- Cluster-level: apply all features via make_interventions ---
        features_cluster = features.clone()
        for iv in fc.make_interventions(0, action=action, value=value):
            iv.apply(features_cluster, seq_len)

        # --- Individual-level: apply each feature one by one ---
        features_individual = features.clone()
        for f in cluster_feats:
            FeatureIntervention(
                feature_idx=f, action=action, value=value,
            ).apply(features_individual, seq_len)

        # Check 1: Feature tensors must be identical
        feat_match = torch.allclose(features_cluster, features_individual, atol=1e-7)
        check(
            f"e2e_{action}:features_match",
            feat_match,
            f"max diff={(features_cluster - features_individual).abs().max().item():.2e}",
        )

        # Check 2: SAE-decoded hidden states must be identical
        hidden_cluster = decode_and_build_hidden(
            sae, features_cluster, orig_hidden, seq_len, original_norms,
        )
        hidden_individual = decode_and_build_hidden(
            sae, features_individual, orig_hidden, seq_len, original_norms,
        )
        hidden_match = torch.allclose(hidden_cluster, hidden_individual, atol=1e-7)
        check(
            f"e2e_{action}:hidden_match",
            hidden_match,
            f"max diff={(hidden_cluster - hidden_individual).abs().max().item():.2e}",
        )

        # Check 3: ESM2 logits from remaining layers must be identical
        logits_cluster = _run_esm_from_layer(
            esm_model, hidden_cluster, token_ids, attn_mask, layer,
        )
        logits_individual = _run_esm_from_layer(
            esm_model, hidden_individual, token_ids, attn_mask, layer,
        )
        logits_match = torch.allclose(logits_cluster, logits_individual, atol=1e-5)
        check(
            f"e2e_{action}:logits_match",
            logits_match,
            f"max diff={(logits_cluster - logits_individual).abs().max().item():.2e}",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global passed, failed

    print("\n" + "=" * 70)
    print("  FeatureClusters Test Suite")
    print("=" * 70)

    print("\n── Unit: YAML round-trip ──")
    test_yaml_roundtrip()

    print("\n── Unit: get_features / get_cluster inverse ──")
    test_get_features_get_cluster_inverse()

    print("\n── Unit: get_features / get_cluster error cases ──")
    test_get_features_unknown_cluster()
    test_get_cluster_unknown_feature()

    print("\n── Unit: len / n_features ──")
    test_len_and_n_features()

    print("\n── Unit: get_top_proteins ──")
    test_get_top_proteins_basic()
    test_get_top_proteins_empty_list_edge_case()
    test_get_top_proteins_string_keys()
    test_get_top_proteins_n_per_feature_limit()

    print("\n── Unit: _get_decoder_weights ──")
    test_get_decoder_weights_relu_sae()

    print("\n── Smoke: _spectral_cluster ──")
    test_spectral_cluster_smoke()

    print("\n── Unit: make_interventions ──")
    test_make_interventions()
    test_make_interventions_zero()

    print("\n── End-to-end: cluster vs individual interventions ──")
    test_cluster_vs_individual_interventions()

    print("\n" + "=" * 70)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 70 + "\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
