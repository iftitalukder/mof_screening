"""
featurize.py
------------
Builds four feature matrices from the precursor (metal + linker) string
ONLY -- nothing from PXRD or crystal structure, consistent with the
"available immediately at synthesis" premise of the paper this work is
inspired by.

Feature groups (co-author's ablation request, docx point #2):
    precursor_only        : Morgan fingerprint of the linker SMILES
                             + one-hot of the primary metal element
                             + metal atom counts (no chemistry knowledge
                             baked in, just structural string encoding)
    descriptor_only        : RDKit physicochemical descriptors of the
                             linker + metal physicochemical properties
                             (electronegativity, covalent radius, etc.)
                             -- explicit chemistry knowledge
    precursor_descriptor   : concatenation of the two above
    shap_selected           : built later, in shap_analysis.py, by
                             subsetting precursor_descriptor to its
                             top-K SHAP features (needs a trained model
                             first, so it's not built here)

Circularity guard: none of the six raw property columns (CO2 uptake,
CH4 uptake, logKH, pore diameter, density) are ever used as inputs
here -- only the precursor string is parsed.
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

# MOF precursor fragments are often ionic/unusual valence SMILES (e.g.
# carboxylate radicals written without explicit charge balancing) that
# RDKit parses fine but complains about loudly. These warnings are
# expected and harmless here, so silence them.
RDLogger.DisableLog("rdApp.*")

import config
import metal_properties as mp

RDKIT_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "MolLogP": Descriptors.MolLogP,
    "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumCarboxylGroups": lambda m: len(
        m.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]"))
    ),
    "NumPyridylN": lambda m: len(
        m.GetSubstructMatches(Chem.MolFromSmarts("n"))
    ),
}


def _safe_mol(smiles: str):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def linker_descriptor_row(smiles: str) -> dict:
    mol = _safe_mol(smiles)
    if mol is None:
        return {name: 0.0 for name in RDKIT_DESCRIPTOR_FUNCS}
    row = {}
    for name, func in RDKIT_DESCRIPTOR_FUNCS.items():
        try:
            row[name] = float(func(mol))
        except Exception:
            row[name] = 0.0
    return row


def linker_fingerprint(smiles: str) -> np.ndarray:
    mol = _safe_mol(smiles)
    if mol is None:
        return np.zeros(config.MORGAN_NBITS, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=config.MORGAN_RADIUS, nBits=config.MORGAN_NBITS
    )
    arr = np.zeros((config.MORGAN_NBITS,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(master: pd.DataFrame):
    """
    Returns (feature_df_precursor_only, feature_df_descriptor_only,
    feature_df_combined) -- all indexed identically to `master`.
    """
    n = len(master)

    # ---- precursor_only: fingerprint bits + metal one-hot + counts ----
    fp_matrix = np.stack([linker_fingerprint(s) for s in master["linker_smiles"]])
    fp_cols = [f"fp_{i}" for i in range(config.MORGAN_NBITS)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=master.index)

    metal_feat_rows = [mp.metal_fragment_features(m) for m in master["metal_frag"]]
    metal_feat_df = pd.DataFrame(metal_feat_rows, index=master.index)

    # keep only the top-N most frequent metals as explicit one-hot columns,
    # bucket the long tail into "metal_other" to avoid an exploding matrix
    top_metals = metal_feat_df["primary_metal"].value_counts().head(15).index.tolist()
    metal_feat_df["primary_metal_bucketed"] = metal_feat_df["primary_metal"].where(
        metal_feat_df["primary_metal"].isin(top_metals), "metal_other"
    )
    metal_onehot = pd.get_dummies(
        metal_feat_df["primary_metal_bucketed"], prefix="metal"
    ).astype(int)

    precursor_only = pd.concat(
        [
            fp_df,
            metal_onehot,
            metal_feat_df[["num_metal_atoms", "num_distinct_metals"]],
        ],
        axis=1,
    )

    # ---- descriptor_only: RDKit linker descriptors + metal physchem ----
    desc_rows = [linker_descriptor_row(s) for s in master["linker_smiles"]]
    desc_df = pd.DataFrame(desc_rows, index=master.index)

    descriptor_only = pd.concat(
        [
            desc_df,
            metal_feat_df[
                [
                    "metal_electronegativity_avg",
                    "metal_covalent_radius_avg",
                    "metal_atomic_weight_avg",
                    "metal_atomic_number_avg",
                ]
            ],
        ],
        axis=1,
    )

    # ---- combined ----
    precursor_descriptor = pd.concat([precursor_only, descriptor_only], axis=1)

    return precursor_only, descriptor_only, precursor_descriptor


def main():
    master_path = os.path.join(config.DATA_PROCESSED, "master_table.csv")
    master = pd.read_csv(master_path)

    precursor_only, descriptor_only, precursor_descriptor = build_features(master)

    # sanity: no leakage columns anywhere
    import data_prep

    for group_name, feat_df in [
        ("precursor_only", precursor_only),
        ("descriptor_only", descriptor_only),
        ("precursor_descriptor", precursor_descriptor),
    ]:
        for app_name in config.APPLICATIONS:
            data_prep.assert_no_leakage(feat_df.columns, app_name)

    precursor_only.to_parquet(
        os.path.join(config.DATA_PROCESSED, "features_precursor_only.parquet")
    )
    descriptor_only.to_parquet(
        os.path.join(config.DATA_PROCESSED, "features_descriptor_only.parquet")
    )
    precursor_descriptor.to_parquet(
        os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet")
    )

    print(f"precursor_only:        {precursor_only.shape}")
    print(f"descriptor_only:       {descriptor_only.shape}")
    print(f"precursor_descriptor:  {precursor_descriptor.shape}")
    print("No leakage detected. Saved feature parquet files to data/processed/")


if __name__ == "__main__":
    main()
