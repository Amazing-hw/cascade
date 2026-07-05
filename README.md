# Cascade Commercial Guard

This folder is a standalone cascade solution for PPG/ACC watch wearing-liveness
detection. It keeps the frozen commercial feature/model contract and adds a
small serial veto guard after commercial positive decisions.

## Standalone Boundary

You can copy this `cascade/` folder to another location and run it directly.
It does not import scripts from the parent `new_codex_1` directory.

Runtime inputs still come from outside the folder:

- an H5 dataset directory passed by `--dataset_dir`;
- Python packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `h5py`,
  `joblib`, `matplotlib`, and `scipy`;
- optional Graphviz `dot` for PNG tree rendering.

The pipeline creates its own local `artifacts/` directory by default.

## Commercial Contract

The commercial model lives in `s01_model.py` and is treated as frozen.
`commercial_model_manifest()` records:

- commercial feature names;
- AdaBoost `tree_num`, `tree_node`, and `detect_tree_threshold`;
- Stage1 timing/gate parameters;
- SHA256 hashes for commercial tree index/value arrays;
- `frozen=True`.

The pipeline writes `artifacts/cascade/commercial_model_manifest.json` as
acceptance evidence. Guard code must not edit the commercial feature names or
tree arrays.

## Data Format

The dataset directory should contain `.h5` files. Supported sample layouts:

- normal sample group containing `ppg`, `target`, and optional `acc`;
- grouped-window sample where child groups are named like `*_w20_1` and each
  child contains `ppg` and optional `acc`.

Supported PPG shapes:

- `(40, T)`;
- `(N_windows, 40, T_window)`.

The pipeline scans H5 files, creates `artifacts/splits.json`, then reuses that
split on later runs unless `--force_split` is provided.

## Quick Start

```bash
cd cascade
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --guard_mode shadow --explain
```

Use `--dry_run` to print commands without executing:

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --dry_run
```

Regenerate the split:

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --force_split
```

## Pipeline

`s10_pipeline.py` runs:

1. Auto split: scan H5 files and write `artifacts/splits.json` when needed.
2. `s05_run_commercial.py`: run the frozen commercial model.
3. `s06_extract_errors.py`: extract features only from commercial-positive,
   Stage2-enabled guard candidates.
4. `s07_select_features.py`: rank and select deployment-friendly guard
   features.
5. `s08_train_corrector.py`: train a tiny XGBoost veto guard, or a constant
   fallback when training labels have one class.
6. `s09_evaluate.py`: compare commercial-only output with selected guard mode.
7. `s11_explain.py`: optional explainability reports when `--explain` is set.

## Guard Modes

- `bypass`: final output equals commercial output.
- `shadow`: final output equals commercial output; guard risk is logged.
- `soft_guard`: final output equals commercial output; high risk requests
  extended detection.
- `hard_veto`: can change a commercial positive to negative only when risk is
  persistent.

Default mode is `shadow`.

Persistent veto requires:

```text
risk_count >= min_veto_windows
risk_ratio >= min_veto_ratio
```

Defaults:

```text
min_veto_windows = 2
min_veto_ratio = 0.4
```

The guard never promotes a commercial negative to positive.

## Hard Negative Policy

Hard negatives are non-wear samples that already passed the frozen commercial
positive filter. They represent direct false-wearing risk.

The cascade training label is:

```text
should_veto = 1 when target == 0 among commercial-positive candidates
```

Hard negatives are used for guard training, data audit, and shadow-mode sample
review. They are not used to retrain or replace the commercial model.

## Outputs

Main artifacts:

- `artifacts/splits.json`
- `artifacts/cascade/commercial_model_manifest.json`
- `artifacts/cascade/commercial_results_{train,valid,test}.csv`
- `artifacts/cascade/error_features_{train,valid,test}.csv`
- `artifacts/cascade/selected_features.json`
- `artifacts/cascade/corrector_model.json`
- `artifacts/cascade/corrector_bundle.pkl`
- `artifacts/cascade/evaluation_report.json`
- `artifacts/cascade/evaluation_samples.csv`
- `artifacts/cascade/evaluation_comparison.csv`

Audit and explainability artifacts:

- `artifacts/cascade/hard_negative_audit/*`
- `artifacts/cascade/feature_review/*`
- `artifacts/cascade/commercial_filter_report/*`
- `artifacts/cascade/figures/*.png`
- `artifacts/cascade/tree_export/*`
- `artifacts/cascade/error_trace/*`

Current image policy is high-resolution PNG only. CSV, JSON, DOT, and Markdown
source/audit files are retained.

## Manual Feature Review

Feature selection writes:

```text
artifacts/cascade/feature_review/ranked_features.csv
artifacts/cascade/feature_review/ranked_features.json
artifacts/cascade/feature_review/ranked_features.md
artifacts/cascade/feature_review/manual_feature_selection_template.json
```

Edit the template, save it as `manual_feature_selection.json`, then run:

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --manual_features artifacts/cascade/feature_review/manual_feature_selection.json
```

Label and label-proxy columns are rejected from manual feature files.

## Important Parameters

- `--dc_threshold`: commercial Stage1 DC threshold, default `0.3e6`.
- Stage1 feature extraction defaults in `s02_features.py`:
  - `DEFAULT_STAGE1_DC_THRESHOLD = 0.3e6`
  - `DEFAULT_STAGE1_AC_DC_THRESHOLD = 1.0`

If strict online-commercial threshold parity is required, pass or restore the
online threshold explicitly and keep it recorded in the produced artifacts.

## Recommended Use

Use cascade as the conservative first-line solution:

1. Run `shadow` to collect disagreement and hard-negative audit data.
2. Review `hard_negative_audit/` and `feature_review/`.
3. Try `soft_guard` for extended-detection behavior.
4. Use `hard_veto` only for offline evaluation or tightly controlled gray
   release.
