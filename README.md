# Home occupancy detection — shallow neural network (7-day final test)

This is the complete and rerun project. **Only the final test-window definition and associated result documentation were changed.** The neural network, input features, original humidity source, expanding-window cross-validation, 24-hour gaps, and threshold sweep from 0.25 to 0.75 are preserved.

## Run the project

1. Keep `NEW Cleaned final data for models.xlsx` in this directory.
2. Install requirements: `python -m pip install -r requirements.txt`.
3. Run `python occupancy_shallow_nn.py`, or open `Occupancy_Shallow_Neural_Network.ipynb` in Jupyter and choose Run All.
4. Run tests with `python tests/test_no_leakage.py`.
5. Windows users may run `RUN_MODEL_WINDOWS.bat` instead.

You can rerun CV with another number of folds using `--cv-folds 4` (four is the unchanged default).

## What changed: final 7-calendar-day holdout

- The **last seven recorded calendar dates** are reserved for final testing, irrespective of observation count.
- The final test window is **Monday May 25 through Sunday May 31, 2020**, with **2,004** five-minute sensor observations (about 22.7% of the 8,838-row dataset).
- May 25–30 contain 288 readings each. The source workbook ends at **May 31 22:55**, so May 31 has 276 readings. **This is seven consecutive calendar days, but the final day is incomplete**. No nonexistent measurements have been fabricated.
- All prior observations are used only in training, validation or CV subject to 24-hour gaps. The former **70:15 train-to-validation ratio** is preserved within the reduced pre-test development pool, and 24-hour embargo gaps remain on both boundaries.
- Actual final training: 5,340 observations, May 1 06:30–May 19 19:25; internal validation: 918, May 20 19:30–May 23 23:55. May 24 is excluded by the test embargo.
- **No test feature or label** enters fitting, scaling, early stopping, the expanded CV folds or threshold selection.

## Unchanged network and humidity source

- The only worksheet used is **`FINAL CLEAN DATA TO DOWNLOAD`**. `Relative Humidity (%)` is read directly from the main final-data sheet. The alternative humidity sheet is never read.
- Inputs: CO₂ ppm, light lux, indoor temperature, outdoor temperature, relative humidity.
- Target `Presence (1 = presence)`; Home=1, Away=0.
- NumPy implementation of 5 → 12 tanh → 1 sigmoid network; weighted cross-entropy, L2 regularization, early stopping, train-only missing-value medians/standardization.
- **Caveat:** if the final-sheet humidity was created upstream with future-aware interpolation, there may be look-ahead leakage in its values. This cannot be fixed through a chronological split; verify the source pipeline before real-time deployment.

## Unchanged expanding-window CV and threshold sensitivity

- **Four explicit expanding forward-chaining folds** on development data; each starts on May 1 and grows. Each fold has its own training-only preprocessing, separate internal validation (for early stopping and threshold), and outer scoring period, with 24-hour gaps.
- Four scoring folds of **654 readings each** (sizes change automatically with the 7-day holdout). CV never sees the final holdout.
- The threshold comparison remains **0.25, 0.30, ..., 0.75** and reports accuracy, Home precision, Home recall and Home F1 on both pooled CV out-of-fold predictions and holdout diagnostics. The holdout sweep is for explanation **only**; do not select a production threshold based on test data.
- The deployed threshold is still independently selected on the internal validation split using macro-F1.

## Updated results (rerun on seven-calendar-day holdout)

| Test metric | Result |
|---|---:|
| Number of held-out readings | 2,004 |
| Test date range | May 25–31, 2020 |
| Test accuracy | **76.7%** |
| Majority-class baseline accuracy | 70.3% |
| Balanced accuracy | 69.8% |
| Home precision | 81.2% |
| Home recall | 87.0% |
| Home F1 | 84.0% |
| Macro F1 | 70.7% |
| ROC AUC | 0.843 |
| Threshold chosen on validation data | 0.23 |

Confusion matrix (actual rows Away, Home; predicted columns Away, Home): `[[313, 283], [183, 1225]]`.

CV mean (four expanding-window outer folds): **balanced accuracy 75.9%, macro F1 76.3%, ROC AUC 84.8%**.

The held-out test metric is an unbiased out-of-sample diagnostic for this frozen configuration, **not** an unbiased selection criterion for new tuning once these test results have been examined. For any future revisions, evaluate on newly recorded data.

## Files and visualizations

Main files: `occupancy_shallow_nn.py`, `Occupancy_Shallow_Neural_Network.ipynb`, `NEW Cleaned final data for models.xlsx`, `requirements.txt`, `RUN_MODEL_WINDOWS.bat`, `README.md`, and tests.

`occupancy_nn_outputs/` provides:

- `prediction_quality_dashboard.png` — new seven-day timeline and confusion matrix/performance.
- `prediction_timeline.png` — Home/Away truth versus prediction with marked errors.
- `confusion_matrix.png`, `roc_curve.png`, `learning_curve.png`.
- `expanding_window_folds.png`, `cross_validation_performance.png` — CV visualization.
- `threshold_comparison.png` — threshold-by-threshold 0.25–0.75 metrics.
- `cross_validation_folds.csv`, `cross_validation_predictions.csv`, `cross_validation_summary.json`.
- `cross_validation_threshold_sweep.csv`, `cross_validation_thresholds_by_fold.csv`, `holdout_threshold_sweep_diagnostic.csv`.
- `occupancy_test_predictions.csv`, `test_metrics.json`, `occupancy_shallow_nn_model.npz`, `leakage_audit.json`.

`training_log.txt`, `leakage_test_log.txt` and `notebook_code_execution_log.txt` document run/testing status.

## Validation

Seven structural/reproducibility test groups pass, with an additional explicit assertion that the final test occupies **May 25–31** and contains **2,004 readings**. Notebook source cells compile and are also checked through in-process execution. As with the original project, upstream sensor construction and future deployment behavior are not fully auditable from the workbook.
