categorical_meta_cols = [
    "Modified residue",
    "Region",
    "Motif",
    "Zinc finger",
    "Compositional bias",
    "Domain [FT]",
]
binary_meta_cols = [
    "Turn",
    "Helix",
    "Beta strand",
    "Coiled coil",
    "Lipidation",
]
paired_binary_cols = ["Disulfide bond"]

categorical_concepts = [
    ("Active site", "ACT_SITE", "note"),
    ("Binding site", "BINDING", "ligand"),
    ("Cofactor", "COFACTOR", "Name"),
    ("Glycosylation", "CARBOHYD", "note"),
    ("Modified residue", "MOD_RES", "note"),
    ("Transit peptide", "TRANSIT", "note"),
    ("Compositional bias", "COMPBIAS", "note"),
    ("Domain [FT]", "DOMAIN", "note"),
    ("Region", "REGION", "note"),
    ("Zinc finger", "ZN_FING", "note"),
    ("Motif", "MOTIF", "note"),
    ("Signal peptide", "SIGNAL", "note"),
]

ptm_groups = {
    "phospho": "Phosphorylation",
    "acetyl": "Acetylation",
    "methyl": "Methylation",
    "hydroxy": "Hydroxylation",
    "deamidat": "Deamidation",
    "ribosyl": "ADP-ribosylation",
    "carboxy": "Carboxylation",
    "citrulline": "Citrullination",
    "ester": "Esterification",
    "glutamyl": "Glutamylation",
    "formyl": "Formylation",
    "succinyl": "Acylation",
    "lactoyl": "Acylation",
    "glutaryl": "Acylation",
    "crotonyl": "Acylation",
    "malonyl": "Acylation",
    "butyryl": "Acylation",
    "nitro": "Oxidation/Reduction",
    "sulf": "Oxidation/Reduction",
    "pantetheine": "Cofactor attachment",
    "fmn": "Cofactor attachment",
    "pyridoxal": "Cofactor attachment",
    "coenzyme": "Cofactor attachment",
    "amp": "Nucleotide attachment",
    "ump": "Nucleotide attachment",
}

subconcepts_to_exclude_from_evals = [
    "Region_any", "Modified residue_any", "Domain_any", "Compositional bias_any", "Motif_any"]
per_aa_concepts = ["Active site", "Cofactor", "Glycosylation",
                   "Modified residue", "amino_acid", "Disulfide bond"]


def is_aa_level_concept(concept_name: str) -> bool:
    return any(aa_concept in concept_name for aa_concept in per_aa_concepts)


from interplm.constants import AMINO_ACIDS, AMINO_ACID_LIST, AMINO_ACID_TO_IDX

# For backward compatibility
amino_acids = AMINO_ACID_LIST
aa_map = AMINO_ACID_TO_IDX


def get_ptm_group(ptm_groups, ptm_name):
    ptm_name = ptm_name.lower()

    for key, group in ptm_groups.items():
        if key in ptm_name:
            return group

    return "Other/Unknown"
