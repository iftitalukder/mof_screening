"""
featurize.py (v2)
------------------
Same feature-building logic as v1 (not flagged as incorrect by the
review) -- precursor_only (Morgan fingerprint + metal one-hot),
descriptor_only (RDKit descriptors + metal physicochemical properties),
and their concatenation. The `shap_selected` group is NOT built here
anymore -- see shap_analysis.py, which now builds it correctly from
TRAINING-fold SHAP values only (fix for #11/#12), and models.py now
actually trains and evaluates it as a real 4th ablation arm.
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

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
    "NumPyridylN": lambda m: len(m.GetSubstructMatches(Chem.MolFromSmarts("n"))),
}


def _safe_mol(smiles: str):
    if not smiles or pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
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
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=config.MORGAN_RADIUS, nBits=config.MORGAN_NBITS)
    arr = np.zeros((config.MORGAN_NBITS,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(master: pd.DataFrame):
    fp_matrix = np.stack([linker_fingerprint(s) for s in master["linker_smiles"]])
    fp_cols = [f"fp_{i}" for i in range(config.MORGAN_NBITS)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=master.index)

    metal_feat_rows = [mp.metal_fragment_features(m) for m in master["metal_frag"]]
    metal_feat_df = pd.DataFrame(metal_feat_rows, index=master.index)

    top_metals = metal_feat_df["primary_metal"].value_counts().head(15).index.tolist()
    metal_feat_df["primary_metal_bucketed"] = metal_feat_df["primary_metal"].where(
        metal_feat_df["primary_metal"].isin(top_metals), "metal_other"
    )
    metal_onehot = pd.get_dummies(metal_feat_df["primary_metal_bucketed"], prefix="metal").astype(int)

    precursor_only = pd.concat(
        [fp_df, metal_onehot, metal_feat_df[["num_metal_atoms", "num_distinct_metals"]]], axis=1
    )

    desc_rows = [linker_descriptor_row(s) for s in master["linker_smiles"]]
    desc_df = pd.DataFrame(desc_rows, index=master.index)
    descriptor_only = pd.concat(
        [desc_df, metal_feat_df[[
            "metal_electronegativity_avg", "metal_covalent_radius_avg",
            "metal_atomic_weight_avg", "metal_atomic_number_avg",
        ]]], axis=1
    )

    precursor_descriptor = pd.concat([precursor_only, descriptor_only], axis=1)
    return precursor_only, descriptor_only, precursor_descriptor


def main():
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    precursor_only, descriptor_only, precursor_descriptor = build_features(master)

    import data_prep
    for group_name, feat_df in [
        ("precursor_only", precursor_only),
        ("descriptor_only", descriptor_only),
        ("precursor_descriptor", precursor_descriptor),
    ]:
        for app_name in config.APPLICATIONS:
            data_prep.assert_no_leakage(feat_df.columns, app_name)

    precursor_only.to_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_only.parquet"))
    descriptor_only.to_parquet(os.path.join(config.DATA_PROCESSED, "features_descriptor_only.parquet"))
    precursor_descriptor.to_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet"))

    print(f"precursor_only:        {precursor_only.shape}")
    print(f"descriptor_only:       {descriptor_only.shape}")
    print(f"precursor_descriptor:  {precursor_descriptor.shape}")
    print("No leakage detected. Saved feature parquet files to data/processed/")


if __name__ == "__main__":
    main()
