"""
Central configuration for the MOF application-screening pipeline.
Edit thresholds / paths here — nothing else in the codebase should
hardcode a path or a threshold.
"""
import os

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_EXTERNAL = os.path.join(ROOT, "data", "external")   # optional hMOF/QMOF from Zenodo
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS_TABLES = os.path.join(ROOT, "results", "tables")
RESULTS_FIGURES = os.path.join(ROOT, "results", "figures")

RAW_FILES = {
    "co2_uptake_lp": "core_uptake.csv",           # CO2 uptake @ low pressure  (chemistry-reliant)
    "ch4_uptake_hp": "core_ch4uptake_highP.csv",   # CH4 uptake @ high pressure (geometry-reliant)
    "logKH_CO2": "core_logKH_CO2.csv",             # Henry's coefficient, CO2 (optional/bonus)
    "logKH_CH4": "core_logKH_CH4.csv",             # Henry's coefficient, CH4 (optional/bonus)
    "pore_diameter": "core_di.csv",                # largest cavity diameter, Å (auxiliary target only)
    "density": "core_density.csv",                 # crystal density, g/cm3   (auxiliary target only)
}

# ---------------------------------------------------------------- seed
SEED = 42

# ---------------------------------------------------------- applications
# Labels are built by thresholding a *property that is never used as an
# input feature* (circularity guard is enforced in data_prep.py).
# Thresholds below use a percentile rule (top X% = promising) rather than
# an absolute industrial cutoff, since we don't have the exact SI cutoffs
# from the source paper. This is documented explicitly in the paper as a
# methodological choice.
APPLICATIONS = {
    "co2_capture": {
        "source_property": "co2_uptake_lp",
        "percentile_cutoff": 75,   # top 25% by CO2 uptake -> "promising"
        "higher_is_better": True,
    },
    "ch4_storage": {
        "source_property": "ch4_uptake_hp",
        "percentile_cutoff": 75,   # top 25% by CH4 uptake -> "promising"
        "higher_is_better": True,
    },
}

# ------------------------------------------------------- feature groups
# These are filled in by featurize.py; listed here so every module agrees
# on the four group names used throughout (ablation table, SHAP, etc.)
FEATURE_GROUPS = [
    "precursor_only",       # Morgan fingerprint of linker + one-hot metal
    "descriptor_only",      # RDKit physicochemical descriptors + metal properties
    "precursor_descriptor", # concatenation of the two above
    "shap_selected",        # top-K features from best precursor_descriptor model
]

MORGAN_RADIUS = 2
MORGAN_NBITS = 256          # kept small on purpose -> fast, low-effort, CPU-friendly
SHAP_TOP_K = 15

# --------------------------------------------------------------- models
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]

TEST_RATIO = 0.15
VALID_RATIO = 0.15
