"""
shap_analysis.py (v2)
----------------------
FIX for #11 (test-set leakage in SHAP feature selection): the
`shap_selected` feature group is now built from SHAP values computed
on the TRAINING fold of the primary (scaffold) split only. The test
fold is never touched during feature selection.

FIX for #12 (README claimed 4 groups, only 3 were ever trained):
`select_shap_features()` is called explicitly in run_pipeline.py BEFORE
models.main(), and the resulting group is added to
config.FEATURE_GROUPS and actually trained/evaluated in the ablation
table, for both applications.

A SEPARATE, later SHAP computation on the test fold of the final
chosen model is still performed for the human-readable trend table
(materials-chemistry interpretation) -- this is legitimate because it
never feeds back into model or feature-set choices used to score that
same test fold; it is purely post-hoc explanation of a fixed, already-
evaluated model.

FIX for #13 (causal-sounding language, undecoded fingerprint bits):
notes now use association language ("associated with", not "->"), and
fingerprint bits are decoded to their actual atom environment via
RDKit's bitInfo wherever possible.
"""
import os
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

import config
import data_prep
import splits as splits_mod

FEATURE_NOTES = {
    "MolWt": "associated with linker/framework size",
    "TPSA": "polar surface area -- associated with framework polarity",
    "NumHDonors": "H-bond donor count -- associated with polar adsorption sites",
    "NumHAcceptors": "H-bond acceptor count -- associated with polar adsorption sites",
    "NumRotatableBonds": "linker flexibility indicator",
    "NumAromaticRings": "aromaticity -- associated with rigidity/pi-stacking",
    "RingCount": "overall ring content of the linker",
    "FractionCSP3": "sp3 fraction -- associated with flexibility vs. rigidity",
    "MolLogP": "hydrophobicity indicator",
    "NumHeteroatoms": "heteroatom content -- associated with polar/coordination sites",
    "HeavyAtomCount": "linker size proxy",
    "NumCarboxylGroups": "carboxylate count -- common MOF coordination group",
    "NumPyridylN": "pyridyl-type N count -- alternative coordination chemistry",
    "metal_electronegativity_avg": "metal electronegativity (avg over node)",
    "metal_covalent_radius_avg": "metal covalent radius (avg over node) -- associated with node geometry",
    "metal_atomic_weight_avg": "correlated with framework density",
    "metal_atomic_number_avg": "correlated with metal identity/periodic row",
    "num_metal_atoms": "cluster nuclearity (single- vs. multi-metal SBU)",
    "num_distinct_metals": "mixed-metal node indicator",
}


def decode_fingerprint_bit(bit_idx: int, sample_smiles_list, radius=None, nbits=None):
    """
    Attempts to decode a Morgan fingerprint bit to the actual atom
    environment it represents, using bitInfo from a handful of sample
    molecules. Returns a human-readable SMARTS-like fragment string, or
    a generic label if the bit can't be traced to any sample molecule.
    """
    radius = radius or config.MORGAN_RADIUS
    nbits = nbits or config.MORGAN_NBITS
    for smiles in sample_smiles_list:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            continue
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits, bitInfo=info)
        if bit_idx in info:
            atom_idx, rad = info[bit_idx][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            try:
                frag_smiles = Chem.MolToSmiles(submol)
                if frag_smiles:
                    return f"substructure: {frag_smiles}"
            except Exception:
                pass
    return "fingerprint bit (no decodable example found in sample)"


def select_shap_features(app_name: str, master: pd.DataFrame,
                          feat_df: pd.DataFrame, split_idx: dict, top_k: int = None):
    """
    Fix for #11: SHAP computed on TRAIN fold only, used purely for
    feature selection (never touches the test fold at this stage).
    """
    top_k = top_k or config.SHAP_TOP_K
    labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)

    data_prep.assert_no_leakage(feat_df.columns, app_name)

    X_train = feat_df.loc[split_idx["train"]]
    y_train = labels.loc[split_idx["train"]]

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=config.SEED, eval_metric="logloss", n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values_train = explainer(X_train)

    mean_abs = np.abs(shap_values_train.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    top_features = X_train.columns[order].tolist()

    return top_features


def build_trend_table_on_test(app_name: str, model, X_test: pd.DataFrame, sample_linkers):
    """
    Separate, legitimate post-hoc SHAP explanation computed on the TEST
    fold of the already-fixed, already-scored final model -- does not
    feed back into training or feature selection for this same test
    fold, so this is not leakage, only interpretation.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:config.SHAP_TOP_K]

    rows = []
    for i in order:
        feat_name = X_test.columns[i]
        vals = X_test.iloc[:, i].values
        svals = shap_values.values[:, i]
        if np.std(vals) == 0 or np.std(svals) == 0:
            direction = "flat"
        else:
            corr = np.corrcoef(vals, svals)[0, 1]
            direction = "nonlinear" if abs(corr) < 0.15 else ("positive" if corr > 0 else "negative")

        if feat_name.startswith("fp_"):
            bit_idx = int(feat_name.split("_")[1])
            note = decode_fingerprint_bit(bit_idx, sample_linkers)
        elif feat_name.startswith("metal_") and feat_name not in FEATURE_NOTES:
            note = "metal identity indicator"
        else:
            note = FEATURE_NOTES.get(feat_name, "n/a")

        rows.append({
            "feature": feat_name,
            "mean_abs_shap": mean_abs[i],
            "shap_trend": direction,
            "materials_meaning": note,
        })
    return pd.DataFrame(rows)
