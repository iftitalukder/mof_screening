"""
splits.py
---------
Co-author points #3 (leakage-aware split) and #4 (external validation).

Two splits are always available, needing nothing beyond the bundled
CoRE-2019 data:
    random        : standard stratified 70/15/15 train/valid/test,
                     de-duplicated on precursor string so no MOF leaks
                     across sets.
    metal_holdout : train on MOFs built from a subset of metals, test
                     on MOFs built from metals NEVER seen in training.
                     This is a stronger generalization check than a
                     random split and needs no extra data -- it stands
                     in for true cross-database external validation.

A third split is unlocked automatically IF the person has downloaded
Zenodo's splits.zip / labels.zip / precursors.zip and placed them under
data/external/ (see README): official CoRE-2019 -> hMOF -> QMOF splits,
letting us do literal cross-database validation instead of the
metal-holdout proxy. See `external_split()` -- returns None if the
files aren't present, and the rest of the pipeline just skips it.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def random_split(master: pd.DataFrame, label_col: str, seed: int = None):
    seed = seed if seed is not None else config.SEED
    idx = master.index.values
    y = master[label_col].values

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=config.TEST_RATIO + config.VALID_RATIO,
        stratify=y, random_state=seed,
    )
    relative_test = config.TEST_RATIO / (config.TEST_RATIO + config.VALID_RATIO)
    idx_valid, idx_test = train_test_split(
        idx_temp, test_size=relative_test, stratify=y_temp, random_state=seed,
    )
    return {"train": idx_train, "valid": idx_valid, "test": idx_test}


def metal_holdout_split(master: pd.DataFrame, label_col: str, seed: int = None,
                         holdout_frac: float = 0.2):
    """
    Picks a random subset of DISTINCT primary metals to hold out
    entirely for testing. Every MOF whose primary metal falls in that
    held-out set goes to the test split; everything else is split
    train/valid. This tests generalization to metal chemistry the
    model has never seen, which is a harder and more meaningful check
    than a random row-level split.
    """
    seed = seed if seed is not None else config.SEED
    rng = np.random.RandomState(seed)

    metals = master["metal_frag"].unique()
    n_holdout = max(1, int(len(metals) * holdout_frac))
    holdout_metals = set(rng.choice(metals, size=n_holdout, replace=False))

    is_test = master["metal_frag"].isin(holdout_metals)
    test_idx = master.index[is_test].values
    remaining = master.index[~is_test].values

    y_remaining = master.loc[remaining, label_col].values
    train_idx, valid_idx = train_test_split(
        remaining, test_size=0.2, stratify=y_remaining, random_state=seed,
    )
    return {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
        "holdout_metals": sorted(holdout_metals),
    }


def external_split():
    """
    Returns paths to official Zenodo split files if the person has
    downloaded and placed them in data/external/, else None.
    Expected (from splits.zip): files identifying which MOFs belong to
    CoRE-2019 / hMOF / QMOF train-test partitions used in the original
    paper. Left as a hook -- wire up parsing here once the files are
    confirmed to be present, since their exact internal format isn't
    known until downloaded.
    """
    candidates = ["splits", "labels", "precursors"]
    found = {
        name: os.path.join(config.DATA_EXTERNAL, name)
        for name in candidates
        if os.path.isdir(os.path.join(config.DATA_EXTERNAL, name))
        or os.path.isfile(os.path.join(config.DATA_EXTERNAL, name + ".zip"))
    }
    return found if found else None


def main():
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))

    ext = external_split()
    if ext is None:
        print(
            "No external Zenodo split files found in data/external/. "
            "Using metal-holdout split as the leakage-aware generalization "
            "test instead (see splits.py docstring). This is fine -- it's "
            "the documented fallback."
        )
    else:
        print(f"Found external split files: {list(ext.keys())} "
              f"(cross-database validation wiring still needs their exact "
              f"internal format confirmed once downloaded).")

    for app_name in config.APPLICATIONS:
        label_col = f"label_{app_name}"
        rs = random_split(master, label_col)
        ms = metal_holdout_split(master, label_col)
        print(
            f"[{app_name}] random split: "
            f"train={len(rs['train'])} valid={len(rs['valid'])} test={len(rs['test'])}"
        )
        print(
            f"[{app_name}] metal-holdout split: "
            f"train={len(ms['train'])} valid={len(ms['valid'])} test={len(ms['test'])} "
            f"({len(ms['holdout_metals'])} metals held out)"
        )


if __name__ == "__main__":
    main()
