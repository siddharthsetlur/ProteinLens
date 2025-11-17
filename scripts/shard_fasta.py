#!/usr/bin/env python
"""
Split large FASTA files into smaller shards while respecting system file limits.
This script provides utilities to count proteins in FASTA files and split them
into manageable shards with a specified number of proteins per shard.
"""

import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, TextIO, Union

from tqdm import tqdm


def get_file_limit() -> int:
    """
    Determine the system's maximum number of open files limit.

    Uses the 'ulimit -n' command to get the system limit and falls back to a
    conservative default if the command fails.

    Returns:
        int: Maximum number of files that can be opened simultaneously
    """
    try:
        result = subprocess.run(
            ['ulimit', '-n'],
            capture_output=True,
            text=True,
            shell=True
        )
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return 1024  # Default to a conservative value


def count_proteins(file_path: Union[str, Path]) -> int:
    """
    Count the total number of proteins in a FASTA file.

    Args:
        file_path: Path to the FASTA file to count proteins in

    Returns:
        int: Number of proteins (sequences) in the file
    """
    print("Counting total proteins...")
    count = 0
    with open(file_path, "r") as f:
        for line in tqdm(f):
            if line.startswith(">"):
                count += 1
    return count


def shard_fasta(
    input_file: Path,
    output_dir: Path,
    proteins_per_shard: int = 1000,
    max_open_files: Optional[int] = None
) -> int:
    """
    Split a large FASTA file into smaller shards with a specified number of proteins per shard.

    This function processes large FASTA files in batches to respect system file handle
    limits. It creates numbered shard files in the specified output directory.

    Args:
        input_file: Path to the input FASTA file
        output_dir: Directory where shard files will be created
        proteins_per_shard: Number of proteins to include in each shard
        max_open_files: Maximum number of files to keep open simultaneously.
            If None, calculated based on system limits

    Returns:
        int: Total number of shards created
    """
    print(f"Proteins per shard: {proteins_per_shard}")

    # Configure file handling limits
    system_limit = get_file_limit()
    if max_open_files is None:
        max_open_files = max(100, min(system_limit - 100, 1000))

    print(f"System file limit: {system_limit}")
    print(f"Using max_open_files: {max_open_files}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Count proteins and calculate shards
    total_proteins = count_proteins(input_file)
    num_shards = (total_proteins + proteins_per_shard -
                  1) // proteins_per_shard
    print(f"Total proteins: {total_proteins}")
    print(f"Number of shards: {num_shards}")

    # Process shards in batches
    print("Sharding proteins...")
    with tqdm(total=total_proteins, unit=" proteins") as pbar:
        for start_shard in range(0, num_shards, max_open_files):
            end_shard = min(start_shard + max_open_files, num_shards)

            # Open current batch of shard files
            current_shard_files: Dict[int, TextIO] = {
                i: open(output_dir / f"shard_{i}.fasta", "w")
                for i in range(start_shard, end_shard)
            }

            try:
                with open(input_file, "r") as infile:
                    current_protein = 0
                    current_content: list[str] = []

                    # Process each line in the input file
                    for line in infile:
                        if line.startswith(">"):
                            # Write previous protein if it belongs to current batch
                            if current_content:
                                shard = current_protein // proteins_per_shard
                                if start_shard <= shard < end_shard:
                                    current_shard_files[shard].write(
                                        "".join(current_content))
                                    pbar.update(1)
                                current_content = []
                            current_protein += 1

                        # Collect lines for current protein if it belongs to current batch
                        shard = (current_protein - 1) // proteins_per_shard
                        if start_shard <= shard < end_shard:
                            current_content.append(line)

                    # Handle the last protein in the file
                    if current_content:
                        shard = (current_protein - 1) // proteins_per_shard
                        if start_shard <= shard < end_shard:
                            current_shard_files[shard].write(
                                "".join(current_content))
                            pbar.update(1)

            finally:
                # Ensure all shard files are properly closed
                for file in current_shard_files.values():
                    file.close()

    return num_shards


if __name__ == "__main__":
    from tap import tapify
    tapify(shard_fasta)
