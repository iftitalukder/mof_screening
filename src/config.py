"""
config.py (v2)
---------------
Every setting introduced to address the co-author's 20-point review is
flagged inline with the point number it addresses.
"""
import os

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS_TABLES = os.path.join(ROOT, "results", "tables")
RESULTS_FIGURES = os.path.join(ROOT, "results", "figures")

RAW_FILES = {
    "co2_uptake_lp": "core_uptake.csv",
    "ch4_uptake_hp": "core_ch4uptake_highP.csv",
    "logKH_CO2": "core_logKH_CO2.csv",
    "logKH_CH4": "core_logKH_CH4.csv",
    "pore_diameter": "core_di.csv",
    "density": "core_density.csv",
}

SEED = 42
# Point #17: multiple seeds instead of one, mean +/- std reported everywhere.
# Kept at 3 (not more) to preserve the low-compute premise given the added
# hyperparameter-tuning overhead introduced by fix #9.
SEEDS = [42, 7, 123]

# --------------------------------------------------------- applications
# Point #3: percentile cutoff is now computed from the TRAINING fold only,
# inside splits.py (never from the full dataset). config just stores which
# percentile / which property.
APPLICATIONS = {
    "co2_capture": {"source_property": "co2_uptake_lp", "percentile_cutoff": 75, "higher_is_better": True},
    "ch4_storage": {"source_property": "ch4_uptake_hp", "percentile_cutoff": 75, "higher_is_better": True},
}

FEATURE_GROUPS = ["precursor_only", "descriptor_only", "precursor_descriptor", "shap_selected", "tanimoto_knn_baseline"]

MORGAN_RADIUS = 2
MORGAN_NBITS = 256
SHAP_TOP_K = 15

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]

TEST_RATIO = 0.15
VALID_RATIO = 0.15

# Point #7: unparseable linkers (405 rows) are now explicitly flagged.
# Default is to EXCLUDE them from modeling (documented, not silent).
EXCLUDE_UNPARSEABLE_LINKERS = True

# Point #9: small, cheap hyperparameter grids (kept intentionally small to
# preserve the "low-compute" premise). Selected using the validation fold
# (point #8 -- valid set is now actually used).
HYPERPARAM_GRIDS = {
    "random_forest": [
        {"n_estimators": 150, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 16},
    ],
    "xgboost": [
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    ],
    "logistic_regression": [
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ],
}

# Point #5: element-level metal holdout fraction (fraction of DISTINCT
# metal ELEMENTS held out, not fragment strings).
METAL_ELEMENT_HOLDOUT_FRAC = 0.2

# Point #14: probability calibration method for the confidence table.
CALIBRATION_METHOD = "sigmoid"  # Platt scaling, fit on the validation fold

# Point #16: additional metrics beyond accuracy/F1/AUC.
ENRICHMENT_TOP_FRACTIONS = [0.10, 0.25]
