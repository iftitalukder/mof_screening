"""
data_prep.py
------------
Loads the six bundled core_*.csv files (precursor_string, value),
merges them into a single master table keyed on precursor string,
and builds the binary application-suitability labels.

Circularity guard (co-author point #1):
    The property used to define a label (e.g. CO2 uptake for the
    "co2_capture" label) is tracked in LABEL_SOURCE_PROPERTIES and is
    asserted to be absent from the feature matrix later in
    featurize.py / models.py. See `assert_no_leakage()` below, which
    every training script calls before fitting anything.
"""
import os
import pandas as pd
import numpy as np

import config


def _load_single(prop_key: str) -> pd.DataFrame:
    fname = config.RAW_FILES[prop_key]
    path = os.path.join(config.DATA_RAW, fname)
    df = pd.read_csv(path, header=None, names=["precursor", prop_key])
    df = df.drop_duplicates(subset="precursor")
    return df


def load_master_table() -> pd.DataFrame:
    """Merge all six property files on the precursor string (inner join)."""
    dfs = [_load_single(k) for k in config.RAW_FILES]
    master = dfs[0]
    for d in dfs[1:]:
        master = master.merge(d, on="precursor", how="inner")
    master = master.reset_index(drop=True)
    return master


def split_precursor(precursor: str):
    """
    Precursor strings are '.'-joined SMILES fragments where SOME
    fragments are the metal node (+ bridging halides, e.g. "[Co]",
    "Cl[Mn][Mn]Cl") and OTHERS are the organic linker(s) / counterions.
    Fragment ORDER is not reliable (metal can appear first, last, or in
    the middle - e.g. "...C(=O)[O-].[Pr]"), so instead of splitting on
    position we classify each '.'-separated fragment by whether it
    contains carbon: fragments with no carbon atom are treated as the
    metal/halide part, fragments with carbon are the organic linker
    part. All organic fragments are rejoined with '.' (a MOF can have
    more than one distinct linker), same for metal fragments
    (multi-metal clusters).
    """
    from rdkit import Chem

    if "." not in precursor:
        return precursor, ""

    metal_parts, organic_parts = [], []
    for frag in precursor.split("."):
        mol = Chem.MolFromSmiles(frag, sanitize=False)
        has_carbon = mol is not None and any(
            atom.GetSymbol() == "C" for atom in mol.GetAtoms()
        )
        if has_carbon:
            organic_parts.append(frag)
        else:
            metal_parts.append(frag)

    metal_frag = ".".join(metal_parts)
    linker = ".".join(organic_parts)
    return metal_frag, linker


def build_application_labels(master: pd.DataFrame) -> pd.DataFrame:
    """Threshold each application's source property into a binary label."""
    df = master.copy()
    for app_name, spec in config.APPLICATIONS.items():
        prop = spec["source_property"]
        cutoff_value = np.percentile(df[prop], spec["percentile_cutoff"])
        if spec["higher_is_better"]:
            df[f"label_{app_name}"] = (df[prop] >= cutoff_value).astype(int)
        else:
            df[f"label_{app_name}"] = (df[prop] <= cutoff_value).astype(int)
    return df


def label_source_properties():
    """Returns {application_name: source_property_column} for leakage checks."""
    return {name: spec["source_property"] for name, spec in config.APPLICATIONS.items()}


def assert_no_leakage(feature_columns, application_name):
    """
    Raises if the property that DEFINES the label for `application_name`
    (or any of the six raw property columns, or the label columns
    themselves) is present in the feature column list. Call this right
    before training any model.
    """
    forbidden = set(config.RAW_FILES.keys()) | {
        f"label_{a}" for a in config.APPLICATIONS
    } | {"precursor", "metal_frag", "linker_smiles"}
    leaked = forbidden.intersection(set(feature_columns))
    if leaked:
        raise ValueError(
            f"Data leakage detected for application '{application_name}': "
            f"feature columns {leaked} overlap with label-source / identifier "
            f"columns. Remove them from the feature matrix."
        )


def main():
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    master = load_master_table()
    master[["metal_frag", "linker_smiles"]] = master["precursor"].apply(
        lambda p: pd.Series(split_precursor(p))
    )
    master = build_application_labels(master)

    out_path = os.path.join(config.DATA_PROCESSED, "master_table.csv")
    master.to_csv(out_path, index=False)

    print(f"Master table: {master.shape[0]} MOFs, {master.shape[1]} columns")
    for app_name in config.APPLICATIONS:
        n_pos = master[f"label_{app_name}"].sum()
        print(
            f"  {app_name}: {n_pos} promising / {len(master)} total "
            f"({100*n_pos/len(master):.1f}%)"
        )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
