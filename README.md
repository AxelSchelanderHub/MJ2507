# Editable hyperparameters — start here

Edit **config.json**, not Python code. The `model` section contains the saved winning settings; `search` describes the original 24-candidate search. **config_small_search.json** is a ready-to-edit nine-candidate grid around the winning learning rate and L2 values. Those small changes have not been evaluated for superiority.

```powershell
python -m pip install -r requirements.txt
# Plot the included original model and test results:
python plot_model_performance.py
# Try small changes using development data only:
python train_model.py --config config_small_search.json --output-dir small_search
```

Read `small_search/search_ranking.csv` and `ensemble_folds.csv` to compare validation balanced accuracy. Each run saves the exact configuration in `resolved_config.json`, split dates in `protocol.json`, and selected settings in `final_parameters.json`. Existing output directories are never overwritten.

To fit one specific setup without a search, edit the `model` section and use:

```powershell
python train_model.py --config config.json --fit-only --output-dir custom_fit
```

To reproduce the saved winning setup AND explicitly evaluate the historical test:

```powershell
python train_model.py --config config.json --fit-only --evaluate-test --output-dir baseline_reproduction
python verify_model.py --results-dir baseline_reproduction
python plot_model_performance.py --results-dir baseline_reproduction --output-dir baseline_charts
```

`--evaluate-test` is deliberately opt-in. Without it the model is saved, but no test metrics/predictions are generated, so test plotting and test verification are not applicable. Do not use repeated test comparisons to choose settings. Only unchanged-default parity was evaluated in this update; new settings cannot be guaranteed to perform similarly.

## Editable controls

- Architecture: hidden width, tanh/ReLU activation, feature family.
- Training: Adam/gradient descent, learning rate, L2, maximum epochs, inverse-frequency class-weight exponent and transition emphasis.
- Early stopping: check interval, patience in checks, minimum improvement, stopping-window hours.
- Adam: moment decay coefficients, gradient coefficients and numerical epsilon. Gradient coefficients are explicitly retained for exact numerical compatibility with the original code; normally pair each with one minus its corresponding beta.
- Initialization scale, ensemble seeds, search seeds and finalist count.
- Features: history duration, delta lags, night start/end hours and normalization floor. Current/past sensors only; no target inputs.
- Transition weighting: emphasis duration and maximum allowed time gap.
- Splits: gaps, calibration duration, development scoring duration/start dates and historical test dates.
- Classification: explicit list of candidate thresholds, selected on calibration balanced accuracy only.
- Search: `recipes` mode combines families and listed recipe overrides; `grid` mode takes every combination of the lists under `grid`. Any model field can be overridden by a recipe or grid entry. Explicit overrides take precedence over `model`; edit the grid/recipe entry when that field is overridden.

The pipeline remains a full-batch, single-hidden-layer five-minute classifier with fixed input sensor definitions and balanced-accuracy selection. Those are algorithm/data-design choices, not silently tuned settings. Numerical legacy defaults remain in model_core.py for loading older saved weights, but every model control is explicitly specified by the JSON for new training runs. Inference reads each model's saved settings, so later JSON edits do not change existing model predictions.

## Compatibility verification

The final three-seed model was retrained using config.json with no search or setting changes. All saved numerical weights, all predictions and every test metric matched the original exactly (maximum probability difference **0.0**). See `PARITY_REPORT.json` and `parity_check/`. Balanced accuracy remains **77.165%**, Home recall **62.216%**, Away recall **92.114%**. Saved-model leakage checks and split tests passed. A short two-candidate integration search verified changed hyperparameters and confirmed test evaluation remained disabled; it is not a performance study. The complete 24-candidate search was not repeated in this refactor.

---

# Optimized 5-minute occupancy model

This is a five-minute **current occupancy** detector. It uses the independently recorded Presence target at each timestamp, with no 15-minute aggregation or any-presence target.

## Quick start

Extract the ZIP and open a terminal in its folder:

```powershell
python -m pip install -r requirements.txt
python plot_model_performance.py
python verify_model.py
```

The included frozen model requires no retraining to reproduce its chart. The chart exports PNG, SVG, PDF and CSV beside the script, using KTH-inspired blues and placing balanced/macro measures first in each group. Windows users may double-click PLOT_WINDOWS.bat.

Predict from new chronological five-minute data using the exact sensor headers and Timestep column (labels are optional):

```powershell
python predict.py readings.csv --output predictions.csv
```

Use the same sensor units and local timestamp convention as training. Calendar features are derived from the timestamp. This model was not validated on the separate external occupancy dataset.

To reproduce all tuning into a new directory:

```powershell
python train_model.py --output-dir rerun_results --evaluate-test
```

The destination must not exist. The plotting script reads the included `results/test_metrics.json` by default; use `--results-dir rerun_results --output-dir rerun_charts` to plot another run. Use train_model.py and predict.py as entry points; model_core.py also contains inherited study utilities that are not entry points for this package.

## Selection and final hyperparameters

We tested 24 predefined recipes: four feature families (sensor levels, levels plus clock, causal dynamics, dynamics plus clock) crossed with six architecture/regularization/weighting recipes. Each used three expanding chronological development folds (72 fits). The top three candidates were re-evaluated as three-seed ensembles on the same folds (27 fits), then the winner was fitted with three seeds (3 fits). These are development-selection scores, not independent performance estimates.

Selection maximized mean validation-fold balanced accuracy. Each fold separately used training, early-stopping, threshold-calibration and scoring windows, with 24-hour gaps. Thresholds 0.05–0.95 in steps of 0.01 were calibrated for balanced accuracy. No smoothing or hysteresis was applied. Full recipes and fold dates are in results/protocol.json; all search results are included.

Winner: 8 inputs → 16 tanh hidden units → sigmoid output; Adam learning rate 0.01; L2 coefficient 0.1 under the included implementation; inverse-frequency training class weights; no transition boost; maximum 900 epochs; seeds 42, 7, 21. The best restored epochs were 880, 40 and 40. Predictions average the three probabilities. Final validation-selected classification threshold: **0.47**. Early stopping checks every ten epochs, with patience of 15 checks and minimum loss improvement 1e-5.

Inputs: CO2 (ppm), Light (lux), Indoor temperature (°C), Outdoor temperature (°C), hour sine, hour cosine, weekend and night. Weekend means Saturday/Sunday; night means 19:00–06:59 at the current timestamp. Humidity is excluded because its upstream interpolation was not verified. No occupancy-derived columns are inputs. Missing-value medians and normalization are fitted on training data only.

## Historical test results

Final fitting uses timestamps before May 18; early stopping May 19–20; calibration May 22–23; test May 25–31 (exclusive June 1), with gaps between each stage. The saved model and selected parameters are written before test evaluation. The test contains 2,004 available five-minute observations, 1,408 Home and 596 Away.

| Metric | Result |
|---|---:|
| Development ensemble mean balanced accuracy | 78.4% |
| Historical test balanced accuracy | 77.2% |
| Home recall | 62.2% |
| Away recall | 92.1% |
| Accuracy | 71.1% |
| Home precision | 94.9% |
| Away precision | 50.8% |
| Macro precision | 72.8% |
| Home F1 | 75.2% |
| Away F1 | 65.5% |
| Macro F1 | 70.3% |
| ROC AUC | 84.5% |
| Always-Home accuracy / balanced accuracy | 70.3% / 50.0% |

Confusion matrix, actual rows and predicted columns ordered Away, Home: `[[549,47],[532,876]]`.

Compared with the earlier five-minute study (74.9% balanced accuracy), this run achieves 77.2% on the same historical period. It trades lower Home recall for higher Away recall. The earlier study optimized a combined event/occupancy utility; this run optimizes balanced accuracy. Change-event recall is only 17/113 = 15.0%, with 54 false events (7.76/day), under the inherited 30-minute, one-to-one event-matching definition. Strong Away recall therefore does not imply reliable transition detection. Do not describe this as uniformly good performance.

These May dates were previously inspected. They are **not a fresh untouched test**, although this run excludes test rows from fitting, stopping, threshold calibration and candidate selection. Existing source cleaning cannot be certified causal merely by these split checks. A new independently collected period is needed for a credible new generalization estimate. The 15-minute model's 82.5% result uses a different target and is not directly comparable.

## Executed checks

verify_model.py passed: exact saved-prediction reproduction; future-sensor and label perturbation invariance; prefix prediction invariance; training-only preprocessing; target/humidity feature exclusions; 24-hour stage gaps; balanced-accuracy arithmetic; reproduction of the final threshold using calibration labels only. The plot was generated successfully. No test-driven retuning followed the reported test evaluation.

The unchanged source Excel and extracted CSV are included in data/. Final parameters, frozen weights, test predictions, validation threshold sweep and complete search logs are in results/.

## Additional evaluation plots

Run `python plot_model_performance.py` to create all four figures: the metric bar chart, `Confusion_Matrix_5min`, `Test_Timeline_5min`, and `Test_Week_Dashboard_5min`. Each is exported as PNG, SVG and PDF. The dashboard shows actual presence, predicted Home probability, the fixed validation threshold (0.47), actual/predicted occupancy strips, the confusion matrix and test metrics. Plotting reads saved test predictions, checks their confusion matrix and threshold decisions, and never retrains or tunes the model.

## Configurable early-stopping window and temporal gaps

The command-line options are:

- `--stop-window-hours` (default **48**): duration of the data window used to monitor early-stopping loss.
- `--gap-hours` (default **24**): gap between training and early stopping, between early stopping and calibration, and between calibration and scoring.

For a **24-hour early-stopping window** while retaining **24-hour gaps**:

```powershell
python train_model.py --stop-window-hours 24 --gap-hours 24 --output-dir run_stop24 --evaluate-test
python verify_model.py --results-dir run_stop24
python plot_model_performance.py --results-dir run_stop24 --output-dir charts_stop24
```

Use a new output folder for every training run. The options apply to all development folds and the final fit. Scoring dates stay fixed; the calibration window stays 48 hours. The code works backward from each scoring start, shifting earlier windows and the training cutoff as needed. Settings and actual dates are recorded in protocol.json; final_parameters.json also records the settings. Nonpositive stopping durations, negative gaps and nonfinite values are rejected. Training/early-stopping/calibration windows must each contain both classes, otherwise training exits with an explanation.

These hours refer to spans of data, not training patience. Patience remains 15 loss checks, with a check every 10 epochs. Smaller gaps reduce temporal separation. Choose these settings based on development results, not repeated comparisons on the historical test week.

The included model was trained with the original defaults (48-hour stopping window, 24-hour gaps); its weights and scores have not changed. New durations require retraining. Tests executed after this update: three split-parameter test groups (default preservation, custom windows/gaps, invalid parameters), plus existing saved-model and leakage checks. Full tuning was not rerun merely to add these controls.
