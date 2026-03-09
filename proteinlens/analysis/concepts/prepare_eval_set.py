import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import trange

from interplm.analysis.concepts.concept_constants import (
    subconcepts_to_exclude_from_evals,
)


def make_eval_subset(
    uniprot_dir: Path,
    valid_shard_range: Tuple[int, int],
    test_shard_range: Optional[Tuple[int, int]] = None,
    min_aa_per_concept: int = 1500,
    min_domains_per_concept: int = 25,
    valid_name: str = "valid",
    test_name: str = "test",
):
    """
    Create a subset of the UniprotKB amino acid concept data for evaluation.

    Each concept must have at least `min_aa_per_concept` amino acids OR `min_domains_per_concept`
    domains to be included in the evaluation set.

    Args:
        valid_shard_range: Tuple of (start, end) shard indices for validation set (inclusive)
        test_shard_range: Tuple of (start, end) shard indices for test set (inclusive)
        uniprot_dir: Path to the UniprotKB directory
        min_aa_per_concept: Minimum number of amino acids per concept to include
        min_domains_per_concept: Minimum number of domains per concept to include
    """

    # Calculate the number of positive amino acids per concept and the number of positive domains per concept
    n_amino_acids = 0

    # Convert shard ranges to lists of indices
    valid_shards = list(range(valid_shard_range[0], valid_shard_range[1] + 1))
    test_shards = list(range(test_shard_range[0], test_shard_range[1] + 1)) if test_shard_range else []

    # Create list of evaluation sets to process
    eval_sets = [(valid_name, valid_shards)]
    if test_shards:
        eval_sets.append((test_name, test_shards))

    # Get the concept names
    with open(uniprot_dir / "uniprotkb_aa_concepts_columns.txt") as f:
        all_concept_names = f.read().splitlines()

    # Some of the catch-all sub-concepts are not meaningful for evaluation, so we exclude them
    # (e.g. "any Region" is not interesting whereas "any Zinc Finger" is)
    indices_of_concepts_to_ignore = [
        all_concept_names.index(c) for c in subconcepts_to_exclude_from_evals
    ]

    aa_counts_per_concept = {
        valid_name: np.zeros(len(all_concept_names)),
        test_name: np.zeros(len(all_concept_names)),
    }
    domain_counts_per_concept = {
        valid_name: np.zeros(len(all_concept_names)),
        test_name: np.zeros(len(all_concept_names)),
    }

    for eval_name, eval_shards in eval_sets:
        for i in eval_shards:
            res = sparse.load_npz(uniprot_dir / f"shard_{i}/aa_concepts.npz").toarray()
            # Count the number of non-zero values in each column
            aa_counts_per_concept[eval_name] += np.count_nonzero(res, axis=0).tolist()
            # Count the max number value in each column (i.e. the number of domains as each domain can span
            # multiple amino acids so we increment the value for each new instance of a domain)
            domain_counts_per_concept[eval_name] += res.max(axis=0).tolist()
            n_amino_acids += len(res)

    # Combine the counts from the valid and test sets
    n_positive_aa_per_concept_total = (
        aa_counts_per_concept[valid_name] + aa_counts_per_concept[test_name]
    )

    n_positive_domains_per_concept_total = (
        domain_counts_per_concept[valid_name] + domain_counts_per_concept[test_name]
    )

    # Determine which concepts have more than the minimum number of domains or amino acids
    indices_of_many_domains = np.where(
        n_positive_domains_per_concept_total > min_domains_per_concept
    )[0]
    indices_of_many_aa = np.where(n_positive_aa_per_concept_total > min_aa_per_concept)[
        0
    ]

    # Get the concept names
    with open(uniprot_dir / "uniprotkb_aa_concepts_columns.txt") as f:
        all_concept_names = f.read().splitlines()

    # Some of the catch-all sub-concepts are not meaningful for evaluation, so we exclude them
    # (e.g. "any Region" is not interesting whereas "any Zinc Finger" is)
    indices_of_concepts_to_ignore = [
        all_concept_names.index(c) for c in subconcepts_to_exclude_from_evals
    ]

    # Combine the indices of concepts with many domains and amino acids, then remove the indices of concepts to ignore
    concept_idx_to_keep = (
        set(indices_of_many_domains) | set(indices_of_many_aa)
    ) - set(indices_of_concepts_to_ignore)
    concept_idx_to_keep = sorted([int(i) for i in concept_idx_to_keep])

    print(
        f"Filtered from {len(all_concept_names)} concepts to {len(concept_idx_to_keep)} concepts "
        f"with at least {min_aa_per_concept:,} amino acids or {min_domains_per_concept:,} domains"
    )

    # Make the evaluation directory
    for eval_name, eval_shards in eval_sets:
        test_dir = uniprot_dir / eval_name
        test_dir.mkdir(parents=True, exist_ok=True)

        # Get the full paths to the shards so they can be accessed easily later
        full_paths_per_shard = {
            i: str((uniprot_dir / f"shard_{i}" / "aa_concepts.npz").resolve())
            for i in eval_shards
        }

        # Save the concept names to a file
        with open(test_dir / "aa_concepts_columns.txt", "w") as f:
            f.write("\n".join([all_concept_names[i] for i in concept_idx_to_keep]))

        # Save the metadata for the evaluation set
        with open(test_dir / "metadata.json", "w") as f:
            metadata = {
                "n_concepts": len(concept_idx_to_keep),
                "n_amino_acids": n_amino_acids,
                "shard_source": eval_shards,
                "n_positive_aa_per_concept": aa_counts_per_concept[eval_name][
                    concept_idx_to_keep
                ].tolist(),
                "n_positive_domains_per_concept": domain_counts_per_concept[eval_name][
                    concept_idx_to_keep
                ].tolist(),
                "indices_of_concepts_to_keep": concept_idx_to_keep,
                "path_to_shards": full_paths_per_shard,
            }
            json.dump(metadata, f)

        print(f"Concept evaluation subset created in {test_dir}")


if __name__ == "__main__":
    from tap import tapify

    tapify(make_eval_subset)
