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
python train_model.py --output-dir rerun_results
```

The destination must not exist. The plotting script reads the included `results/test_metrics.json`; change SOURCE in that script to plot a new run. Use train_model.py and predict.py as entry points; model_core.py also contains inherited study utilities that are not entry points for this package.

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
