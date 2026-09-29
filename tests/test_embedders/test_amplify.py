"""Tests for the AMPLIFY embedder and AMPLIFY loss-recovered fidelity.

All tests load the real chandar-lab/AMPLIFY_120M checkpoint (pinned revision)
and run on CPU, where the remote code uses torch SDPA instead of xformers.
They pin down the three places AMPLIFY differs from ESM2 in HF:
hidden_states indexing, the additive attention mask, and the tokenizer.
"""

import numpy as np
import pytest
import torch

from proteinlens.embedders import Amplify, ESM, load_embedder
from proteinlens.train.fidelity import calculate_loss_recovered
from proteinlens.train.fidelity_amplify import (
    AmplifyFidelityConfig,
    AmplifyFidelityFunction,
)

MODEL = "chandar-lab/AMPLIFY_120M"
LAYER = 12

SEQS = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "ACDEFGHIKLMNPQRSTVWY",
    "MSXUKLLVZAGGO",  # non-AMPLIFY residues -> <unk>, still one token each
    "MKKLLPTAAAGLLLLAAQPAMA" * 5,
]


@pytest.fixture(scope="module")
def emb():
    return Amplify(model_name=MODEL, device="cpu")


def raw_hidden(emb, seq):
    ids = emb.tokenizer(seq, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return emb.model(ids, output_hidden_states=True).hidden_states


def test_load_embedder_dispatch(emb):
    assert isinstance(load_embedder(MODEL, device="cpu"), Amplify)
    assert ESM.MODEL_ALIASES  # ESM path untouched; dispatch covered by name check
    assert emb.get_embedding_dim(LAYER) == 640
    assert emb.available_layers == list(range(1, 25))


def test_tokenization_one_token_per_residue(emb):
    tok = emb.tokenizer
    for seq in SEQS:
        ids = tok(seq)["input_ids"]
        assert len(ids) == len(seq) + 2
        assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id
    ids = tok("MSXUKLLVZAGGO")["input_ids"][1:-1]
    assert ids.count(tok.unk_token_id) == 4  # X, U, Z, O


def test_hidden_states_have_no_embedding_entry(emb):
    hs = raw_hidden(emb, SEQS[0])
    assert len(hs) == emb.model.config.num_hidden_layers == 24


def test_layer_indexing_matches_block_output(emb):
    """layer=12 must be the output of transformer block 12, i.e. hidden_states[11]."""
    seq = SEQS[0]
    got = emb.embed_single_sequence(seq, LAYER)
    expected = raw_hidden(emb, seq)[LAYER - 1][0, 1 : len(seq) + 1].numpy()
    np.testing.assert_array_equal(got, expected)

    captured = {}
    def hook(module, inputs, output):
        captured["h"] = output[0]

    handle = emb.model.transformer_encoder[LAYER - 1].register_forward_hook(hook)
    try:
        raw_hidden(emb, seq)
    finally:
        handle.remove()
    # Separate forward pass: equal up to CPU kernel float noise (~2e-5).
    np.testing.assert_allclose(
        captured["h"][0, 1 : len(seq) + 1].numpy(), expected, atol=1e-4, rtol=0
    )


@pytest.mark.parametrize("layer", [0, 25])
def test_out_of_range_layers_rejected(emb, layer):
    with pytest.raises(ValueError):
        emb.embed_single_sequence(SEQS[0], layer)


def test_embed_single_sequence_shape(emb):
    long_seq = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV" * 20)[:1022]
    for seq in [SEQS[2], long_seq]:
        out = emb.embed_single_sequence(seq, LAYER)
        assert out.shape == (len(seq), 640)
        assert out.dtype == np.float32


def test_batched_matches_single(emb):
    """Padded batch (additive mask) == one-at-a-time (no mask): survey vs extraction paths."""
    batched = emb.extract_embeddings(SEQS, LAYER, batch_size=len(SEQS))
    single = torch.cat(
        [torch.from_numpy(emb.embed_single_sequence(s, LAYER)) for s in SEQS]
    )
    assert batched.shape == (sum(len(s) for s in SEQS), 640)
    torch.testing.assert_close(batched, single, atol=1e-4, rtol=1e-4)


def test_boundaries(emb):
    out = emb.extract_embeddings_with_boundaries(SEQS[:3], LAYER, batch_size=2)
    lens = [len(s) for s in SEQS[:3]]
    assert out["boundaries"] == [(0, lens[0]), (lens[0], lens[0] + lens[1]),
                                 (lens[0] + lens[1], sum(lens))]
    assert out["embeddings"].shape[0] == sum(lens)


def test_fasta_shard_layout(emb, tmp_path):
    import yaml
    fasta = tmp_path / "shard_0.fasta"
    fasta.write_text("".join(f">p{i}\n{s}\n" for i, s in enumerate(SEQS)))
    emb.embed_fasta_file_multiple_layers(fasta, [LAYER], output_dir=tmp_path / "out")
    shard = tmp_path / "out" / f"layer_{LAYER}" / "shard_0"
    acts = torch.load(shard / "activations.pt")
    meta = yaml.safe_load((shard / "metadata.yaml").read_text())
    assert acts.shape == (sum(len(s) for s in SEQS), 640)
    assert meta["d_model"] == 640 and meta["layer"] == LAYER
    assert meta["total_tokens"] == acts.shape[0]


@pytest.fixture(scope="module")
def fidelity(tmp_path_factory):
    path = tmp_path_factory.mktemp("fid") / "eval_sequences.txt"
    # Canonical residues only: AMPLIFY never trained on <unk>, so a sequence
    # with X/U/Z/O has CE ~7 and would swamp the baseline check.
    path.write_text("\n".join(s for s in SEQS if s != "MSXUKLLVZAGGO") + "\n")
    cfg = AmplifyFidelityConfig(
        eval_seq_path=path, model_name=MODEL, layer_idx=LAYER, eval_batch_size=3
    )
    return cfg.build()


def test_fidelity_baselines(fidelity):
    assert isinstance(fidelity, AmplifyFidelityFunction)
    assert fidelity.orig_loss < 1.0  # unmasked token identity is easy
    assert fidelity.zero_loss > fidelity.orig_loss + 1.0


def test_fidelity_identity_patch_recovers_everything(fidelity):
    """Patching the layer's own output back in must reproduce the original CE."""
    identity = lambda x, unnormalize=True: x
    ce = fidelity._CE_from_sae_recon(fidelity.tokenized_batches, identity)
    assert ce == pytest.approx(fidelity.orig_loss, abs=1e-5)
    assert calculate_loss_recovered(ce, fidelity.orig_loss, fidelity.zero_loss) == pytest.approx(100.0)


def test_fidelity_hook_hidden_matches_hidden_states(fidelity):
    tokens, mask = fidelity.tokenized_batches[0]
    _, hidden = fidelity._output_with_intervention(tokens, mask)
    from proteinlens.train.fidelity_amplify import additive_attention_mask
    with torch.no_grad():
        hs = fidelity.model(tokens, attention_mask=additive_attention_mask(mask),
                            output_hidden_states=True).hidden_states
    torch.testing.assert_close(hidden, hs[LAYER - 1])
