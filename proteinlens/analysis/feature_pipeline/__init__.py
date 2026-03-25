"""Feature data pipeline for SAE visualizer.

Two-pass pipeline that computes per-feature data (top activating sequences,
activation range samples, per-residue activation maps, coverage stats) from
a trained SAE and a protein dataset (e.g. human SwissProt).
"""
