# Lightweight and Explainable ML for MOF Application Screening (v2)

This is a corrected rebuild addressing all 20 points raised in the
co-author's review of the original pipeline. **Every point below was
independently verified against the actual code/data before being
"confirmed"** -- this file states what was found, not just what was
claimed.

## Setup (VS Code terminal, PowerShell or bash)

```
python -m venv venv
venv\Scripts\Activate.ps1        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cd src
python run_pipeline.py
```

**Expect this to take roughly 15-25 minutes**, not ~1 minute like the
old pipeline. That's the real cost of fixing the issues below (multi-
seed evaluation, hyperparameter tuning, a proper scaffold split,
calibration, a baseline model) -- it is still 100% CPU, no GPU
required anywhere.

## What changed, mapped to the co-author's review

| # | Issue | Verified? | Fix |
|---|---|---|---|
| 1 | Duplicate precursors silently collapsed, discarding 2,494 real MOFs | **Confirmed** -- 912 groups share a precursor string; 93.5%/94.3% of those groups have meaningfully different CO2/CH4 values | The six CSVs are row-aligned (verified: identical precursor column row-for-row across all six files). Merge is now by row position, keeping all 8,571 MOFs. `data_prep.precursor_degeneracy_report()` explicitly quantifies and reports the resulting label ambiguity as a stated limitation, not a hidden bug. |
| 2 | README overclaimed "precursor predicts application suitability" | Fair critique | Language throughout now says "predicts suitability *from available precursor-only signal*" and the Limitations section explicitly states precursor alone cannot resolve topology/interpenetration differences -- point #1's degeneracy report is the direct evidence for this. |
| 3 | Percentile label cutoff computed on full dataset (train+test) before splitting | **Confirmed** in code | `splits.make_labels_from_train_threshold()` computes the percentile from the TRAINING fold only, per split scheme; test/valid never influence their own label definition. |
| 4 | 55% train/test linker overlap in the random split | **Confirmed** (55.4% measured) | Two fixes: (a) `random_split()` is now group-aware by exact precursor string (0% exact-duplicate overlap, verified), (b) a new `scaffold_split()` groups by Bemis-Murcko scaffold and is now the **primary** reported evaluation (0.0% linker overlap, verified, down from 55.4%). |
| 5 | "Unseen metal" test wasn't unseen at the element level | **Confirmed, worse than stated** -- 0/50 test elements were actually absent from training | `metal_element_holdout_split()` holds out individual elements (not fragment strings) and programmatically verifies zero element overlap between train and holdout (raises an error if violated). Reports both a "partial-unseen" and a stricter "fully-unseen" test subset. |
| 6 | External validation was a non-functional stub presented as a "legitimate substitute" | **Confirmed** | Honesty fix: the metal-element-holdout split (fix #5) is now the explicitly-labeled internal generalization test; literal cross-database validation (hMOF/QMOF) is documented as unimplemented future work, not claimed as done. |
| 7 | 405 unparseable linkers silently zero-filled | **Confirmed exactly** (405 in the old 6,077-row dataset; 605 in the corrected 8,571-row dataset, since more raw rows are now retained) | `flag_unparseable_linkers()` adds an explicit `linker_parse_failed` column; by default these rows are excluded from modeling (`config.EXCLUDE_UNPARSEABLE_LINKERS`), with the count reported, not hidden. |
| 8 | Validation split built but never used | **Confirmed** (dead code in v1) | Validation fold now drives hyperparameter selection (see #9) and probability calibration (see #14). |
| 9 | No hyperparameter tuning | **Confirmed** | Small grids per model (`config.HYPERPARAM_GRIDS`), selected by validation-fold AUC. Kept deliberately small to preserve the low-compute framing. |
| 10 | No feature scaling for Logistic Regression | **Confirmed** | `StandardScaler` (fit on train only) now precedes Logistic Regression in a pipeline. Tree-based models are untouched (scaling doesn't affect them). |
| 11 | SHAP-based feature selection used the TEST set | **Confirmed** | `shap_analysis.select_shap_features()` now computes SHAP only on the TRAINING fold of the primary split. The test fold is never touched during feature selection. |
| 12 | README claimed 4 feature groups, only 3 were ever trained | **Confirmed** | `shap_selected` is now built (from fix #11's train-fold SHAP) and actually trained/evaluated as a real 4th ablation arm for both applications. |
| 13 | SHAP notes used causal language; fingerprint bits were undecoded | **Confirmed** | Notes now use association language ("associated with", not "->"). Fingerprint bits are decoded to an actual substructure via RDKit's `bitInfo` where possible (e.g. `fp_107` -> `substructure: ccc`) instead of a generic placeholder. |
| 14 | Confidence score wasn't calibrated | **Confirmed** | `CalibratedClassifierCV` (Platt/sigmoid) fit on the validation fold; Brier score reported before/after so calibration quality is checked, not assumed. |
| 15 | Screening table only showed the first 25 test rows | **Confirmed** | Full test set is scored and saved (679 rows for CO2 capture on the scaffold split); console output previews the top 10 by confidence, the saved CSV has everything. |
| 16 | Metrics incomplete for an imbalanced screening task | Fair critique | Precision, recall, PR-AUC (average precision), and enrichment factor at top-10%/top-25% are now reported alongside accuracy/F1/ROC-AUC. |
| 17 | Single seed only | **Confirmed** (v1 used seed=42 only) | Primary (scaffold-split) results are now averaged over 3 seeds (42, 7, 123), reported as mean +/- std. The two comparison splits (random, metal-holdout) use a single reference seed each to keep runtime reasonable -- documented, not hidden. |
| 18 | No simple chemistry baseline | Fair critique | A 5-nearest-neighbor Tanimoto-similarity baseline (`baseline.py`) is now included alongside the ML models, per application, on the scaffold split. |
| 19 | "Low compute" claim lacked memory/model-size data, and inference time wasn't per-sample | **Confirmed** | `compute_comparison.csv` now reports peak training memory (via `tracemalloc`), serialized model size, and per-sample inference time in addition to wall-clock training time. |
| 20 | CO2 (chemistry-reliant) and CH4 (geometry-reliant) results presented identically | Fair critique | Results and discussion now explicitly separate the two applications and note that precursor-only input structurally under-serves a geometry-reliant property like CH4 storage. |

## Primary results (scaffold split, 3-seed mean -- the headline numbers)

These are the numbers to use in the paper. They are **substantially
lower** than the old (leaky) random-split numbers -- that drop *is*
the fix, not a regression. The scaffold split guarantees zero linker
overlap between train and test, so these numbers reflect genuine
generalization to unseen chemistry.

| Application | Best feature group | Best model | ROC-AUC | F1 |
|---|---|---|---|---|
| CO2 capture | precursor_descriptor | random_forest | 0.733 +/- 0.055 | 0.244 +/- 0.048 |
| CH4 storage | precursor_descriptor | random_forest | 0.817 +/- 0.024 | 0.544 +/- 0.062 |

The Tanimoto-kNN baseline scores AUC 0.535 (CO2) and 0.749 (CH4) on
the same split -- i.e. for CH4 storage, a substantial share of the
ML models' apparent skill is recoverable from simple chemical
similarity lookup, while for CO2 capture the ML models clearly add
value over the naive baseline. This nuance is worth stating directly
in the paper rather than glossing over.

The two comparison splits (`random_group_aware`, `metal_element_holdout`)
are reported in the same `ablation_results_full.csv` for transparency,
clearly labeled by `split_type`, so a reader can see exactly how much
the split methodology itself affects the apparent result.

## Project layout

```
mof_screening_v2/
├── data/raw/              core_*.csv (unchanged, from XRayPro repo)
├── data/processed/        master_table.csv, feature parquets, shap_selected parquets,
│                          precursor_degeneracy_report.csv
├── src/
│   ├── config.py                all settings, each tagged with which review point it addresses
│   ├── data_prep.py             fix #1 (row-aligned merge), #3 (no pre-split labeling), #7 (parse flag)
│   ├── metal_properties.py      unchanged from v1
│   ├── featurize.py             unchanged feature logic, now on the corrected dataset
│   ├── splits.py                fix #3, #4 (scaffold split), #5 (element-level holdout)
│   ├── baseline.py              fix #18 (Tanimoto-kNN)
│   ├── shap_analysis.py         fix #11, #12, #13
│   ├── models.py                fix #8, #9, #10, #16, #17, #19
│   ├── confidence_screen.py     fix #14, #15
│   ├── error_analysis.py        re-run on the rigorous split
│   ├── run_shap_trend_report.py final SHAP interpretation table
│   ├── build_compute_table.py   fix #19 table extraction
│   └── run_pipeline.py          runs everything, in order
├── results/tables/         all output CSVs
├── requirements.txt
└── README.md               (this file)
```

## Honest residual limitations (not fully solved, stated plainly)

- The comparison splits (`random_group_aware`, `metal_element_holdout`)
  use a single seed, not 3, to keep total runtime reasonable -- their
  std values will read as 0.0 in the results table. This is a
  documented compute/rigor tradeoff, not an oversight.
- Literal cross-database validation (training on CoRE-2019, testing on
  an independently sourced hMOF/QMOF split) is still not implemented --
  the metal-element-holdout split is the closest available proxy, but
  it is not the same thing, and the paper should say so.
- The precursor-degeneracy ceiling quantified in fix #1 means perfect
  accuracy is mathematically impossible for the ~2,494 MOFs that share
  a precursor string with a structurally different MOF -- this is now
  visible in `precursor_degeneracy_report.csv` rather than hidden.
