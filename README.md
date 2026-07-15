# Lightweight and Explainable ML for MOF Application Screening

Predicts MOF application suitability (CO2 capture, CH4 storage) using
**only precursor chemistry** (metal + linker SMILES) -- no PXRD, no
crystal structure, no GPU. Inspired by XRayPro
(Khan & Moosavi, *Nature Communications* 2025), but deliberately
lighter-weight: classical ML (Logistic Regression / Random Forest /
XGBoost) on RDKit descriptors + Morgan fingerprints, with SHAP
explainability layered on top.

Runs end-to-end on CPU in under a minute.

## Setup (VS Code terminal)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run everything

```bash
cd src
python run_pipeline.py
```

All output tables land in `results/tables/`:
- `ablation_results.csv` -- feature-group x model x application accuracy/F1/AUC
- `metal_holdout_results.csv` -- generalization to unseen metals
- `compute_comparison.csv` -- training/inference time per model
- `shap_trend_<application>.csv` -- global SHAP feature importance + direction
- `confidence_screen_<application>.csv` -- promising/uncertain/deprioritize table
- `error_analysis_<application>.csv` -- false positive/negative breakdown

## Data

The `data/raw/` folder already contains the six precursor+property CSVs
(8,571 CoRE-2019 MOFs) copied from the XRayPro GitHub repo
(https://github.com/AI4ChemS/XRayPro) -- no download needed to run the
pipeline as-is.

**Optional (for literal cross-database validation):** download these
three files from Zenodo (https://zenodo.org/records/14908210) and drop
them in `data/external/`:
- `labels.zip` (733KB)
- `precursors.zip` (45.2MB)
- `splits.zip` (1.2MB, official CoRE-2019/hMOF/QMOF train-test splits)

Without these, the pipeline uses a metal-holdout split as its
leakage-aware generalization test instead (documented in `splits.py`) --
this is a legitimate substitute, not a placeholder, so the pipeline is
fully functional either way.

## Project layout

```
mof_screening_paper/
├── data/
│   ├── raw/              core_*.csv (bundled, from XRayPro repo)
│   ├── external/         optional Zenodo files go here
│   └── processed/        generated: master_table.csv + feature parquets
├── src/
│   ├── config.py             paths, thresholds, seeds, feature-group names
│   ├── data_prep.py          merge CSVs, build labels, circularity guard
│   ├── metal_properties.py   metal element property lookup table
│   ├── featurize.py          builds the 3 base feature groups
│   ├── splits.py             random split + metal-holdout split
│   ├── models.py             ablation training + compute-time table
│   ├── shap_analysis.py      SHAP trend table + shap_selected feature group
│   ├── confidence_screen.py  promising/uncertain/deprioritize table
│   ├── error_analysis.py     false positive/negative breakdown
│   └── run_pipeline.py       runs everything, in order
├── results/
│   ├── tables/            all output CSVs
│   └── figures/           (empty -- add plots here if you want them)
├── requirements.txt
└── README.md
```

## Methodology notes for the paper

- **Circularity guard**: the property used to define each application
  label (CO2 uptake for `co2_capture`, CH4 uptake for `ch4_storage`) is
  never included as an input feature. Enforced programmatically via
  `data_prep.assert_no_leakage()`, called before every model fit.
- **Labels**: built by thresholding the raw property at its 75th
  percentile (top 25% = "promising"). This is a percentile-based
  methodological choice (documented, not an industrial cutoff from the
  source paper's SI, which wasn't available to us).
- **Feature groups**: `precursor_only` (Morgan fingerprint + metal
  one-hot), `descriptor_only` (RDKit physicochemical descriptors +
  metal electronegativity/radius/weight), `precursor_descriptor`
  (concatenation), `shap_selected` (top-15 SHAP features from the best
  `precursor_descriptor` model).
