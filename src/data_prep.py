"""
data_prep.py (v2)
------------------
FIX for point #1 (the most serious issue):
    The old pipeline merged the six property CSVs by TEXT-MATCHING the
    precursor string and then dropped duplicates, discarding 2,494 of
    8,571 real MOF entries. Verified: the six CSV files are row-aligned
    (row i in every file refers to the SAME original MOF, confirmed by
    an exact row-for-row precursor-column match across all six files).
    So the correct merge is by ROW POSITION, not by string content.
    This preserves all 8,571 MOFs.

    A consequence, now made EXPLICIT rather than hidden: many of these
    8,571 MOFs share an identical precursor string but have different
    property values (912 such groups; 853/912 differ in CO2 uptake,
    860/912 differ in CH4 uptake by more than noise). This is a real,
    irreducible limitation of precursor-only prediction -- two MOFs can
    share the same metal + linker "recipe" but differ in topology,
    interpenetration, or pore geometry. We quantify this explicitly via
    `precursor_degeneracy_report()` instead of pretending it doesn't
    exist.

FIX for point #7:
    linker SMILES that fail to parse (RDKit returns None, or the field
    is empty/NaN -- 405 rows total) are flagged with an explicit
    `linker_parse_failed` column. By default (config.EXCLUDE_UNPARSEABLE
    _LINKERS = True) these rows are excluded from modeling, and the
    exclusion count is reported, not silently zero-filled.

FIX for point #3:
    Percentile-based application labels are NO LONGER computed here.
    Raw property columns are kept as-is; the percentile threshold is
    computed per-split, from the TRAINING fold only, inside splits.py.
    This file only prepares the raw merged table.
"""
import os
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import config


def _load_single(prop_key: str) -> pd.DataFrame:
    fname = config.RAW_FILES[prop_key]
    path = os.path.join(config.DATA_RAW, fname)
    df = pd.read_csv(path, header=None, names=["precursor", prop_key])
    return df  # NOTE: no drop_duplicates here -- see module docstring


def load_master_table() -> pd.DataFrame:
    """
    Row-position merge across all six property files (fix for #1).
    Verified upstream that all six files share identical row order /
    identical precursor column content row-for-row, so a plain
    column-wise concat (not a string-key merge) is correct here and
    preserves every one of the 8,571 original MOF entries.
    """
    dfs = [_load_single(k) for k in config.RAW_FILES]

    # sanity check every run: if this ever fails, the row-alignment
    # assumption is violated and a string-key merge would be required
    # instead -- fail loudly rather than silently mis-merging.
    base_precursor = dfs[0]["precursor"].values
    for i, (key, df) in enumerate(zip(config.RAW_FILES.keys(), dfs)):
        if not (df["precursor"].values == base_precursor).all():
            raise ValueError(
                f"Row-alignment assumption violated at file for '{key}' -- "
                f"precursor column does not match row-for-row against the "
                f"first file. A row-position merge is unsafe; investigate "
                f"before proceeding."
            )

    master = pd.concat(
        [dfs[0][["precursor"]]] + [d[[k]] for k, d in zip(config.RAW_FILES.keys(), dfs)],
        axis=1,
    )
    master["mof_id"] = master.index  # explicit unique ID per original MOF row
    return master


def precursor_degeneracy_report(master: pd.DataFrame) -> pd.DataFrame:
    """
    Quantifies point #1's consequence directly: for every precursor
    string shared by more than one MOF row, how often do the two
    application-relevant raw properties actually disagree? This bounds
    how much accuracy is achievable from precursor alone, since
    identical input features cannot yield different predictions.
    """
    rows = []
    grouped = master.groupby("precursor")
    n_degenerate_groups = 0
    for precursor, g in grouped:
        if len(g) < 2:
            continue
        n_degenerate_groups += 1
        rows.append(
            {
                "precursor": precursor,
                "n_mofs_sharing_precursor": len(g),
                "co2_uptake_range": g["co2_uptake_lp"].max() - g["co2_uptake_lp"].min(),
                "ch4_uptake_range": g["ch4_uptake_hp"].max() - g["ch4_uptake_hp"].min(),
            }
        )
    report = pd.DataFrame(rows)
    print(
        f"Precursor degeneracy: {n_degenerate_groups} precursor strings are "
        f"shared by >1 MOF row ({master.shape[0] - master['precursor'].nunique()} "
        f"'extra' rows beyond one-per-precursor)."
    )
    if len(report):
        co2_conflict = (report["co2_uptake_range"] > 0.01).mean()
        ch4_conflict = (report["ch4_uptake_range"] > 0.01).mean()
        print(
            f"  Of those, {co2_conflict*100:.1f}% have a meaningfully different "
            f"CO2 uptake and {ch4_conflict*100:.1f}% a meaningfully different "
            f"CH4 uptake across the group -- i.e. identical precursor-only "
            f"features but different ground truth. This is an irreducible "
            f"ceiling on precursor-only prediction accuracy, not a bug."
        )
    return report


def split_precursor(precursor: str):
    """Same atom-level classification as v1 (this part was not flagged
    as incorrect by the review) -- fragments containing carbon are
    treated as organic linker, fragments without carbon as metal node."""
    if "." not in str(precursor):
        return precursor, ""
    metal_parts, organic_parts = [], []
    for frag in str(precursor).split("."):
        mol = Chem.MolFromSmiles(frag, sanitize=False)
        has_carbon = mol is not None and any(a.GetSymbol() == "C" for a in mol.GetAtoms())
        (organic_parts if has_carbon else metal_parts).append(frag)
    return ".".join(metal_parts), ".".join(organic_parts)


def flag_unparseable_linkers(master: pd.DataFrame) -> pd.DataFrame:
    """Fix for #7: explicit flag instead of silent zero-fill."""
    def _parses(smiles):
        if pd.isna(smiles) or smiles == "":
            return False
        return Chem.MolFromSmiles(str(smiles)) is not None

    master = master.copy()
    master["linker_parse_failed"] = ~master["linker_smiles"].apply(_parses)
    n_failed = master["linker_parse_failed"].sum()
    print(f"Linker SMILES parse failures (flagged): {n_failed} / {len(master)}")
    return master


def assert_no_leakage(feature_columns, application_name):
    """Circularity guard, unchanged in spirit from v1, extended with the
    new raw-property / identifier columns."""
    forbidden = set(config.RAW_FILES.keys()) | {
        f"label_{a}" for a in config.APPLICATIONS
    } | {"precursor", "metal_frag", "linker_smiles", "mof_id", "linker_parse_failed"}
    leaked = forbidden.intersection(set(feature_columns))
    if leaked:
        raise ValueError(
            f"Data leakage detected for application '{application_name}': "
            f"feature columns {leaked} overlap with label-source / identifier "
            f"columns."
        )


def main():
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    master = load_master_table()
    print(f"Loaded master table (row-position merge): {len(master)} MOFs "
          f"(previously 6,077 after incorrect string-based dedup -- fix #1)")

    degeneracy_report = precursor_degeneracy_report(master)
    degeneracy_report.to_csv(
        os.path.join(config.DATA_PROCESSED, "precursor_degeneracy_report.csv"), index=False
    )

    master[["metal_frag", "linker_smiles"]] = master["precursor"].apply(
        lambda p: pd.Series(split_precursor(p))
    )
    master = flag_unparseable_linkers(master)

    if config.EXCLUDE_UNPARSEABLE_LINKERS:
        n_before = len(master)
        master = master[~master["linker_parse_failed"]].reset_index(drop=True)
        print(f"Excluded {n_before - len(master)} rows with unparseable linkers "
              f"(config.EXCLUDE_UNPARSEABLE_LINKERS=True). Remaining: {len(master)}")

    out_path = os.path.join(config.DATA_PROCESSED, "master_table.csv")
    master.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(master)} MOFs, {master['precursor'].nunique()} "
          f"distinct precursor strings)")


if __name__ == "__main__":
    main()
