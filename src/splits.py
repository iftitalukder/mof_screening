"""
splits.py (v2)
---------------
FIX for #3 (percentile leakage): `make_labels_from_train_threshold()`
computes the percentile cutoff from the TRAINING fold's raw property
values only, then applies that fixed cutoff to valid/test. This means
each split scheme (random / scaffold / metal-holdout) gets its own
threshold, computed without ever looking at its own test fold.

FIX for #4 (55% train/test chemical overlap in the old random split):
Two changes:
  (a) `random_split()` is now GROUP-AWARE by precursor string, so exact
      duplicate MOFs (912 groups, see data_prep.py) can never be split
      across train and test.
  (b) a new `scaffold_split()` groups MOFs by Bemis-Murcko scaffold of
      the linker and assigns whole scaffold groups to train/valid/test,
      so structurally similar linkers can't leak across the split
      either. This is reported as the PRIMARY, more rigorous
      evaluation; the (still group-deduplicated) random split is kept
      for comparison but explicitly labeled as an optimistic upper
      bound.

FIX for #5 (metal holdout wasn't actually element-level): the metal
fragment string (e.g. "[Zn][Zn]" vs "[Zn]") is no longer the holdout
unit. `metal_element_holdout_split()` holds out individual metal
ELEMENTS: every training row is guaranteed to contain ZERO atoms of
any held-out element (verified programmatically, not just by
construction), and the test set is split into a "partial-unseen"
subset (contains at least one held-out element) and a stricter
"fully-unseen" subset (every metal atom in the node is a held-out
element) for honest reporting of both.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

import config
import metal_properties as mp


def make_labels_from_train_threshold(master: pd.DataFrame, train_idx, application_name: str):
    """Fix for #3: percentile computed from TRAIN fold only."""
    spec = config.APPLICATIONS[application_name]
    prop = spec["source_property"]
    cutoff_value = np.percentile(master.loc[train_idx, prop], spec["percentile_cutoff"])
    if spec["higher_is_better"]:
        labels = (master[prop] >= cutoff_value).astype(int)
    else:
        labels = (master[prop] <= cutoff_value).astype(int)
    return labels, cutoff_value


def _murcko_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "INVALID"
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf is not None else "NO_SCAFFOLD"
    except Exception:
        return "INVALID"


def random_split(master: pd.DataFrame, seed: int = None):
    """
    Fix for #4(a): group-aware by exact precursor string (GroupShuffleSplit),
    so the 912 duplicate-precursor groups can never straddle train/test.
    Labels are NOT computed here anymore -- call
    make_labels_from_train_threshold() with the returned train indices.
    """
    seed = seed if seed is not None else config.SEED
    groups = master["precursor"].values
    idx = master.index.values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.TEST_RATIO + config.VALID_RATIO, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))
    train_idx, temp_idx = idx[train_idx], idx[temp_idx]

    temp_groups = master.loc[temp_idx, "precursor"].values
    relative_test = config.TEST_RATIO / (config.TEST_RATIO + config.VALID_RATIO)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    valid_rel, test_rel = next(gss2.split(temp_idx, groups=temp_groups))
    valid_idx, test_idx = temp_idx[valid_rel], temp_idx[test_rel]

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def scaffold_split(master: pd.DataFrame, seed: int = None):
    """
    Fix for #4(b): PRIMARY rigorous split. Groups MOFs by the linker's
    Bemis-Murcko scaffold; whole scaffold groups go entirely to one of
    train/valid/test, so structurally related linkers cannot leak
    across the split.
    """
    seed = seed if seed is not None else config.SEED
    scaffolds = master["linker_smiles"].apply(_murcko_scaffold).values
    idx = master.index.values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.TEST_RATIO + config.VALID_RATIO, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=scaffolds))
    train_idx, temp_idx = idx[train_idx], idx[temp_idx]

    temp_scaffolds = pd.Series(scaffolds, index=master.index).loc[temp_idx].values
    relative_test = config.TEST_RATIO / (config.TEST_RATIO + config.VALID_RATIO)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    valid_rel, test_rel = next(gss2.split(temp_idx, groups=temp_scaffolds))
    valid_idx, test_idx = temp_idx[valid_rel], temp_idx[test_rel]

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def metal_element_holdout_split(master: pd.DataFrame, seed: int = None,
                                 holdout_frac: float = None):
    """
    Fix for #5: element-level holdout with a programmatic no-leakage
    guarantee (checked, not assumed).
    """
    seed = seed if seed is not None else config.SEED
    holdout_frac = holdout_frac if holdout_frac is not None else config.METAL_ELEMENT_HOLDOUT_FRAC
    rng = np.random.RandomState(seed)

    element_sets = master["metal_frag"].apply(lambda f: set(mp.get_metal_atoms(f)))
    all_elements = sorted(set().union(*element_sets)) if len(element_sets) else []
    n_holdout = max(1, int(len(all_elements) * holdout_frac))
    holdout_elements = set(rng.choice(all_elements, size=n_holdout, replace=False))

    has_any_holdout = element_sets.apply(lambda s: len(s & holdout_elements) > 0)
    all_holdout = element_sets.apply(lambda s: len(s) > 0 and s.issubset(holdout_elements))

    train_pool_idx = master.index[~has_any_holdout].values
    test_idx = master.index[has_any_holdout].values
    fully_unseen_test_idx = master.index[all_holdout].values

    # no-leakage verification (fail loudly if ever violated)
    train_elements = set().union(*element_sets.loc[train_pool_idx]) if len(train_pool_idx) else set()
    violation = train_elements & holdout_elements
    if violation:
        raise ValueError(f"Metal-holdout leakage: {violation} present in both train and holdout sets.")

    y_dummy = np.zeros(len(train_pool_idx))  # stratification not needed here; done post-labeling by caller
    n_valid = int(len(train_pool_idx) * 0.2)
    train_pool_idx = train_pool_idx.copy()
    rng.shuffle(train_pool_idx)
    valid_idx = train_pool_idx[:n_valid]
    train_idx = train_pool_idx[n_valid:]

    return {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
        "test_fully_unseen": fully_unseen_test_idx,
        "holdout_elements": sorted(holdout_elements),
    }


def main():
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    print("=== Random split (group-aware by exact precursor string) ===")
    rs = random_split(master)
    print(f"train={len(rs['train'])} valid={len(rs['valid'])} test={len(rs['test'])}")
    train_prec = set(master.loc[rs['train'], 'precursor'])
    test_prec = set(master.loc[rs['test'], 'precursor'])
    print(f"Exact precursor overlap test/train: {len(train_prec & test_prec)} "
          f"(should be 0)")

    print("\n=== Scaffold split (primary, rigorous) ===")
    ss = scaffold_split(master)
    print(f"train={len(ss['train'])} valid={len(ss['valid'])} test={len(ss['test'])}")
    train_linkers = set(master.loc[ss['train'], 'linker_smiles'])
    test_linkers = set(master.loc[ss['test'], 'linker_smiles'])
    overlap_rows = master.loc[ss['test'], 'linker_smiles'].isin(train_linkers).mean()
    print(f"Test-set rows with linker also seen in train: {overlap_rows*100:.1f}% "
          f"(was 55.4% under the old plain random split)")

    print("\n=== Metal element-holdout split ===")
    ms = metal_element_holdout_split(master)
    print(f"train={len(ms['train'])} valid={len(ms['valid'])} test={len(ms['test'])} "
          f"(of which {len(ms['test_fully_unseen'])} are fully-unseen-metal rows)")
    print(f"Held-out elements ({len(ms['holdout_elements'])}): {ms['holdout_elements']}")
    print("No-leakage check passed (would have raised otherwise).")


if __name__ == "__main__":
    main()
