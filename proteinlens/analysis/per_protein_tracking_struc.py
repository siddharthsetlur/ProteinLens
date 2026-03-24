"""
Organizes proteins based on their maximum activation value for each feature. Both finds
proteins that have the higest activation value for each feature and finds proteins where
the maximum activation value *within that protein* is in a pre-specified quantile range. Also compute and save the 3D backbone of that protein
"""

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interplm.sae.dictionary import Dictionary
from interplm.sae.inference import get_sae_feats_in_batches, split_up_feature_list
from interplm.utils import get_device
from interplm.data_processing.embedding_loader import load_shard_embeddings
from transformers import AutoTokenizer, EsmForProteinFolding
from interplm.analysis.per_protein_tracking import PerProteinActivationTracker
def load_esmfold_model(device: str):
    """Load and configure ESMFold model for structure prediction."""
    print("Loading ESMFold model for structure prediction...")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1",
        clean_up_tokenization_spaces=True
    )
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.float()
    # Always use CPU for ESMFold to avoid memory issues on limited GPU memory
    # model = model.to("cpu")
    model.eval()
    print("✓ ESMFold model loaded on CPU")
    return tokenizer, model


def get_protein_sequence(metadata_dir: Path, shard: int, protein_id: str) -> Optional[str]:
    """Retrieve protein sequence from metadata."""
    import pandas as pd
    
    # Try to load protein data from the shard
    protein_data_path = metadata_dir / f"shard_{shard}" / "protein_data.tsv"
    if protein_data_path.exists():
        df = pd.read_csv(protein_data_path, sep="\t")
        # Convert protein IDs to uppercase to match
        df['Entry'] = df['Entry'].str.upper()
        protein_id = protein_id.upper()
        
        if protein_id in df['Entry'].values:
            row = df[df['Entry'] == protein_id].iloc[0]
            return row['Sequence'] if 'Sequence' in df.columns else None
    
    return None


def generate_pdb_structure(sequence: str, tokenizer, model, device: str) -> str:
    """Generate PDB structure from protein sequence using ESMFold."""
    with torch.no_grad():
        pdb_string = model.infer_pdb(sequence)
    return pdb_string


def find_max_examples_with_structures(
    sae,
    aa_embeds_dir: Path,
    aa_metadata_dir: Path,
    shards_to_search: List[int],
    feature_chunk_size: int = 200,
    n_top_proteins_to_track: int = 8,
    lower_quantile_thresholds: List = None,
    activation_threshold: float = 0.05,
    save_protein_structure: bool = False,
    esmfold_model: Optional[Any] = None,
    esmfold_tokenizer: Optional[Any] = None,
) -> Dict:
    """
    Extended version of find_max_examples_per_feat that optionally generates structures.
    """
    if lower_quantile_thresholds is None:
        lower_quantile_thresholds = [(0, 0.4), (0.4, 0.8), (0.8, 1.0)]
    
    total_features = sae.dict_size
    device = get_device()

    # Initialize enhanced tracker
    tracker = PerProteinActivationTrackerWithStructures(
        total_features,
        n_top=n_top_proteins_to_track,
        lower_quantile_thresholds=lower_quantile_thresholds,
        activation_threshold=activation_threshold,
        save_structures=save_protein_structure,
    )

    # Process each shard
    for shard_idx, shard in enumerate(shards_to_search):
        print(f"Processing shard {shard} ({shard_idx+1}/{len(shards_to_search)})...")
        
        # Load embeddings
        shard_data = load_shard_embeddings(aa_embeds_dir, shard, device, return_tensor_only=False)
        
        # Extract embeddings and protein IDs
        if isinstance(shard_data, dict) and 'protein_ids' in shard_data and 'boundaries' in shard_data:
            aa_embeddings = shard_data['embeddings']
            boundaries = shard_data['boundaries']
            protein_ids = shard_data['protein_ids']
            
            # Convert boundaries to per-AA protein ID mapping
            uniprot_id_per_aa = []
            for protein_id, (start, end) in zip(protein_ids, boundaries):
                uniprot_id_per_aa.extend([protein_id] * (end - start))
            uniprot_id_per_aa = pd.Series(uniprot_id_per_aa, name="protein_id")
        else:
            # Legacy format - load from metadata
            aa_embeddings = shard_data if isinstance(shard_data, torch.Tensor) else shard_data['embeddings']
            uniprot_id_per_aa = _get_protein_ids(aa_metadata_dir, shard)
        
        # Map amino acid indices to protein IDs
        from collections import defaultdict
        prot_id_to_idx = defaultdict(list)
        for i, prot_id in enumerate(uniprot_id_per_aa):
            prot_id_to_idx[prot_id].append(i)
        
        # Store shard info for structure generation
        tracker.set_current_shard(shard)
        
        # Process features in chunks
        from tqdm import tqdm
        from interplm.sae.inference import split_up_feature_list, get_sae_feats_in_batches
        
        for feature_list in tqdm(
            split_up_feature_list(
                total_features=total_features,
                max_feature_chunk_size=feature_chunk_size,
            ),
            desc=f"Processing feature chunks for shard {shard}",
        ):
            # Get SAE features
            sae_feats = get_sae_feats_in_batches(
                sae=sae,
                device=device,
                aa_embds=aa_embeddings,
                chunk_size=25_000,
                feat_list=feature_list,
                normalize_features=True,
            )
            
            # Process each protein
            for prot_id, prot_idx in prot_id_to_idx.items():
                protein_features = sae_feats[prot_idx].cpu().numpy()
                
                # Get sequence if we need structures
                sequence = None
                if save_protein_structure:
                    sequence = get_protein_sequence(aa_metadata_dir, shard, prot_id)
                
                # Update tracker
                tracker.update(
                    protein_features,
                    protein_id=prot_id,
                    feature_ids=feature_list,
                    sequence=sequence
                )
            
            del sae_feats
    
    # Generate structures for top proteins if requested
    if save_protein_structure and esmfold_model is not None:
        print("\nGenerating 3D structures for top proteins...")
        tracker.generate_structures_for_top_proteins(
            esmfold_tokenizer, esmfold_model, device, aa_metadata_dir, shards_to_search
        )
    
    return tracker.get_results()


class PerProteinActivationTrackerWithStructures(PerProteinActivationTracker):
    """Extended tracker that can store protein structures."""
    
    def __init__(self, *args, save_structures: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_structures = save_structures
        self.current_shard = None
        # Store sequences for top proteins
        self.max_heap_sequences = [[] for _ in range(self.num_features)]
        # Will store PDB structures after generation
        self.protein_structures = {}
    
    def set_current_shard(self, shard: int):
        """Set the current shard being processed."""
        self.current_shard = shard
    
    def update(self, feature_activations, protein_id: str, feature_ids: List, sequence: Optional[str] = None):
        """Extended update that tracks sequences for structure generation."""
        # Call parent update
        super().update(feature_activations, protein_id, feature_ids)
        
        if self.save_structures and sequence is not None:
            # Store sequences for proteins in max heap
            for i, feature_id in enumerate(feature_ids):
                # Check if this protein is in the max heap for this feature
                if any(pid == protein_id for _, pid in self.max_heap[feature_id]):
                    # Store protein info for later structure generation
                    if protein_id not in self.protein_structures:
                        self.protein_structures[protein_id] = {
                            'sequence': sequence,
                            'shard': self.current_shard,
                            'pdb': None  # Will be filled later
                        }
    
    def generate_structures_for_top_proteins(self, tokenizer, model, device, metadata_dir, shards):
        """Generate PDB structures for all top proteins."""
        from tqdm import tqdm
        
        # Collect unique proteins that need structures
        proteins_needing_structures = set()
        for feature_heap in self.max_heap:
            for _, protein_id in feature_heap:
                if protein_id in self.protein_structures and self.protein_structures[protein_id]['pdb'] is None:
                    proteins_needing_structures.add(protein_id)
        
        print(f"Generating structures for {len(proteins_needing_structures)} unique proteins...")
        
        for protein_id in tqdm(proteins_needing_structures, desc="Generating structures"):
            if protein_id in self.protein_structures:
                protein_info = self.protein_structures[protein_id]
                
                # If we don't have sequence, try to get it from metadata
                if protein_info['sequence'] is None:
                    for shard in shards:
                        seq = get_protein_sequence(metadata_dir, shard, protein_id)
                        if seq:
                            protein_info['sequence'] = seq
                            break
                
                # Generate structure if we have sequence
                if protein_info['sequence']:
                    try:
                        pdb_string = generate_pdb_structure(
                            protein_info['sequence'], tokenizer, model, "cpu"
                        )
                        self.protein_structures[protein_id]['pdb'] = pdb_string
                    except Exception as e:
                        print(f"Warning: Failed to generate structure for {protein_id}: {e}")
    
    def get_results(self) -> Dict:
        """Get results including protein structures."""
        results = super().get_results()
        
        if self.save_structures:
            # Add structures to the max examples
            max_with_structures = {}
            for feature_id, protein_ids in results['max'].items():
                max_with_structures[feature_id] = []
                for protein_id in protein_ids:
                    protein_entry = {'protein_id': protein_id}
                    if protein_id in self.protein_structures:
                        protein_entry['pdb'] = self.protein_structures[protein_id]['pdb']
                    max_with_structures[feature_id].append(protein_entry)
            
            results['max_with_structures'] = max_with_structures
        
        return results


def _get_protein_ids(aa_metadata_dir: Path, shard_num: int):
    """Helper function to get protein IDs from metadata."""
    import pandas as pd
    import numpy as np
    
    data = pd.read_csv(
        aa_metadata_dir / f"shard_{shard_num}" / "protein_data.tsv", sep="\t"
    )
    return pd.Series(
        np.repeat(data["Entry"], data["Length"]), name="protein_id"
    ).reset_index(drop=True)