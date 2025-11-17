#!/usr/bin/env python
"""
This script filters protein sequences by length and randomly selects a subset.
"""
import gzip
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm


def count_proteins(file_path: Union[str, Path]) -> int:
    """Count number of proteins in a FASTA file."""
    count = 0
    with gzip.open(file_path, "rt") if str(file_path).endswith('.gz') else open(file_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def filter_and_select_proteins(
    input_file: str,
    output_file: str,
    num_proteins: int,
    max_length: Optional[int] = 1022
) -> None:
    """
    Filter protein sequences by length and randomly select a subset.

    Args:
        input_file: Path to input FASTA file (can be .fasta or .fasta.gz)
        output_file: Path to output FASTA file (will be compressed if ends with .gz)
        num_proteins: Number of proteins to randomly select
        max_length: Maximum allowed length for protein sequences (default: 1022)
    """
    # First, create a temporary file with length-filtered proteins
    temp_file = output_file + ".temp"

    print(f"Filtering proteins by length (max_length: {max_length})...")

    # Counter for proteins that meet length criteria
    filtered_count = 0
    # Store headers and sequences
    filtered_proteins: List[Tuple[str, str]] = []

    # Open input file (handling both compressed and uncompressed)
    open_input = gzip.open if str(input_file).endswith('.gz') else open

    # First pass: filter by length and store valid proteins
    with open_input(input_file, "rt") as infile:
        current_header = ""
        current_sequence = ""

        for line in tqdm(infile, desc="Filtering proteins"):
            if line.startswith(">"):
                # Process previous sequence if it exists
                if current_header and len(current_sequence.replace("\n", "")) <= max_length:
                    filtered_proteins.append(
                        (current_header, current_sequence))
                    filtered_count += 1
                # Start new sequence
                current_header = line
                current_sequence = ""
            else:
                current_sequence += line

        # Handle the last sequence
        if current_header and len(current_sequence.replace("\n", "")) <= max_length:
            filtered_proteins.append((current_header, current_sequence))
            filtered_count += 1

    print(f"Found {filtered_count} proteins meeting length criteria")

    # If num_proteins is greater than filtered_count, select all filtered proteins
    num_to_select = min(num_proteins, filtered_count)

    # Randomly select from filtered proteins
    selected_indices = set(random.sample(range(filtered_count), num_to_select))

    # Open output file (handling both compressed and uncompressed)
    open_output = gzip.open if str(output_file).endswith('.gz') else open

    print(f"Writing {num_to_select} randomly selected proteins...")

    # Write selected proteins to output file
    with open_output(output_file, "wt") as outfile:
        for idx in tqdm(selected_indices, desc="Writing selected proteins"):
            header, sequence = filtered_proteins[idx]
            outfile.write(header)
            outfile.write(sequence)

    print(f"Successfully wrote {num_to_select} proteins to {output_file}")


if __name__ == "__main__":
    from tap import tapify
    tapify(filter_and_select_proteins)
