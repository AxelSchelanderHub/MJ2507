"""
Occupancy detection with a Deep Neural Network (MJ2507), grid search 16 November
================================================================================

Copy of "Deep Neural Network Project rebuilt model 15 november.py" with a grid
search (L7) over the learning rate and the network architecture added.
The network, training loop, early stopping, scaling and windows are unchanged.

Model:      [LINEAR -> RELU] x (L-1) -> LINEAR -> SIGMOID      (L7, homework)
Init:       W = randn / sqrt(n[l-1]), b = 0 (Xavier)       (DNN Application homework, dnn_app_utils_v3)
Cost:       cross-entropy                                      (L2, homework)
Optimizer:  batch gradient descent                             (homework)
Scaling:    min-max to [0, 1], fitted on the fit-weights window only   (L3)
Evaluation: chronological expanding window, 4 separate windows per fold:
              fit weights -> early stopping -> choose threshold -> score
            with a 24 h gap between windows (same as the group's shallow NN)
Grid search (L7, L9):
            every combination of learning rate and architecture is run on the
            3 validation folds only. The combination with the highest mean
            score over the 3 folds wins. Only the winner is then run on the
            "Final historical check" (the test week), once.
Metrics:    confusion matrix, precision, recall, F1, macro F1, balanced accuracy  (L9, L11)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================
FOLDER = Path(__file__).resolve().parent
DATA_FILE = FOLDER / "NEW Cleaned final data for models 15 november.xlsx"

# Grid search (L7). Rows and columns are the same as the table in the slides.
# The first number of each architecture is the number of features (7).
LEARNING_RATES = [0.1, 0.01, 0.001]
ARCHITECTURES = [[7, 32, 16, 8, 1],
                 [7, 16, 8, 1],
                 [7, 8, 4, 1]]

# Score used to compare the combinations: the mean over the 3 validation folds.
# Balanced accuracy is also what the threshold is chosen with, so the whole
# pipeline is tuned towards the same goal. ("macro_f1" or "f1" also work.)
SELECT_METRIC = "balanced_accuracy"

MAX_ITERATIONS = 10000     # upper limit, early stopping usually ends earlier

# Early stopping (L9: keep the model with the lowest validation loss)
CHECK_EVERY = 10           # compute the early-stopping cost every 10 iterations
PATIENCE = 2000            # stop if no improvement for this many iterations

# Threshold search on the threshold window
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# Pre-processing options
LOG_LIGHT = False          # True: use log(1 + lux) instead of lux
CLIP_TO_0_1 = False        # True: clip scaled values outside [0, 1]

# Evaluation windows, read from the group's shallow-NN figure.
# Each window is [start, end). The 24 h gaps are the days between windows.
# "fit" always starts at the first row of the data (expanding window).
WINDOWS = {
    "Validation fold 1": {"fit_end": "2020-05-07",
                          "stop": ("2020-05-08", "2020-05-10"),
                          "threshold": ("2020-05-11", "2020-05-13"),
                          "score": ("2020-05-14", "2020-05-17")},
    "Validation fold 2": {"fit_end": "2020-05-10",
                          "stop": ("2020-05-11", "2020-05-13"),
                          "threshold": ("2020-05-14", "2020-05-16"),
                          "score": ("2020-05-17", "2020-05-20")},
    "Validation fold 3": {"fit_end": "2020-05-14",
                          "stop": ("2020-05-15", "2020-05-17"),
                          "threshold": ("2020-05-18", "2020-05-20"),
                          "score": ("2020-05-21", "2020-05-24")},
    "Final historical check": {"fit_end": "2020-05-18",
                               "stop": ("2020-05-19", "2020-05-21"),
                               "threshold": ("2020-05-22", "2020-05-24"),
                               "score": ("2020-05-25", "2020-06-02")},
}
FINAL_NAME = "Final historical check"

# Column keywords (short name -> text that appears in the Excel header)
TIME_KEYWORD = "Timestep"
TARGET_KEYWORD = "Presence"
FEATURE_KEYWORDS = {
    "CO2": "CO2",
    "Light": "Light",
    "Indoor temp": "Indoor",
    "Outdoor temp": "Outdoor",
    "Humidity": "Humidity",
    "Weekend": "Weekend",
    "Night": "Night",
}

# Colours used in the slides
NAVY, MID_BLUE, LIGHT_BLUE = "#000060", "#004791", "#6699CC"


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def find_column(df, keyword):
    for col in df.columns:
        if keyword.lower() in str(col).lower():
            return col
    raise ValueError(f"No column contains '{keyword}'")


def load_data(path):
    df = pd.read_excel(path)

    rename = {find_column(df, TIME_KEYWORD): "time",
              find_column(df, TARGET_KEYWORD): "presence"}
    for short_name, keyword in FEATURE_KEYWORDS.items():
        rename[find_column(df, keyword)] = short_name
    df = df.rename(columns=rename)[list(rename.values())]

    # The Excel file ends with empty rows where only Presence = 0 is filled in.
    # They are not real measurements, so every row with a missing value is dropped.
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(df)} empty rows, {len(df)} rows left.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["presence"] = df["presence"].astype(int)

    if LOG_LIGHT:
        df["Light"] = np.log1p(df["Light"])
    return df


def select(df, start, end):
    """Rows with start <= time < end."""
    return df[(df["time"] >= pd.Timestamp(start)) & (df["time"] < pd.Timestamp(end))]


# =============================================================================
# 2. MIN-MAX NORMALIZATION (L3), fitted on the fit-weights window only
# =============================================================================
def fit_min_max(X_fit):
    """Learn min and max of every feature from the fit-weights window."""
    x_min = X_fit.min(axis=1, keepdims=True)
    x_max = X_fit.max(axis=1, keepdims=True)
    return x_min, x_max


def apply_min_max(X, x_min, x_max):
    """x' = (x - min) / (max - min), using the min and max from the fit window."""
    span = np.where(x_max - x_min == 0, 1, x_max - x_min)   # avoid dividing by 0
    X_scaled = (X - x_min) / span
    if CLIP_TO_0_1:
        X_scaled = np.clip(X_scaled, 0, 1)
    return X_scaled


def make_xy(df, feature_names):
    """Homework shapes: X is (n_features, m), Y is (1, m)."""
    X = df[feature_names].values.T.astype(float)
    Y = df["presence"].values.reshape(1, -1).astype(float)
    return X, Y


# =============================================================================
# 3. DNN FUNCTIONS FROM THE HOMEWORK ("Step by Step")
# =============================================================================
def sigmoid(Z):
    # np.clip only stops np.exp from overflowing for very large |Z|;
    # it does not change the result.
    A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def initialize_parameters_deep(layer_dims):
    """
    Initialization from the DNN Application homework's helper file
    (dnn_app_utils_v3): W = randn / sqrt(n[l-1]), b = 0  (Xavier).

    Why not * 0.01 (L5)? Every layer multiplies the signal by its weights.
    With * 0.01 the signal and the gradients shrink layer after layer, and the
    network sat on a flat cost for ~1,500 iterations before it started using the
    sensors. Dividing by sqrt(n[l-1]) keeps the signal about the same size in
    every layer, e.g. std 1/sqrt(7) = 0.38 for layer 1 and 1/sqrt(20) = 0.22
    for layer 2.
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    # Clip so we never take log(0) if a prediction is exactly 0 or 1.
    AL = np.clip(AL, 1e-12, 1 - 1e-12)
    cost = -(1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return np.squeeze(cost)


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    AL = np.clip(AL, 1e-12, 1 - 1e-12)   # same reason as in compute_cost
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(params, grads, learning_rate):
    parameters = {k: v.copy() for k, v in params.items()}
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters


# =============================================================================
# 4. L-LAYER MODEL (Application homework) + EARLY STOPPING (L9)
# =============================================================================
def L_layer_model(X, Y, layers_dims, X_stop, Y_stop,
                  learning_rate, max_iterations=MAX_ITERATIONS):
    """
    Same loop as the homework: forward -> cost -> backward -> update.
    Added: every CHECK_EVERY iterations, the cost on the early-stopping window
    is computed. The parameters with the lowest early-stopping cost are kept,
    and training stops when it has not improved for PATIENCE iterations.
    """
    np.random.seed(1)
    parameters = initialize_parameters_deep(layers_dims)

    best_cost = np.inf
    best_parameters = parameters
    best_iteration = 0
    history = {"iteration": [], "train_cost": [], "stop_cost": []}

    for i in range(max_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % CHECK_EVERY == 0:
            AL_stop, _ = L_model_forward(X_stop, parameters)
            stop_cost = compute_cost(AL_stop, Y_stop)
            history["iteration"].append(i)
            history["train_cost"].append(float(cost))
            history["stop_cost"].append(float(stop_cost))

            if stop_cost < best_cost:
                best_cost = stop_cost
                best_parameters = parameters
                best_iteration = i
            elif i - best_iteration >= PATIENCE:
                break

    return best_parameters, best_iteration, history


def predict_proba(X, parameters):
    AL, _ = L_model_forward(X, parameters)
    return AL


# =============================================================================
# 5. METRICS (L9, L11)
# =============================================================================
def compute_metrics(Y_true, Y_pred):
    """
    Positive class = 1 = presence ("home").
    Every "away" metric is the same formula with "away" as the positive class,
    and every "macro" metric is the average of the home and away versions.
    """
    y, p = Y_true.ravel(), Y_pred.ravel()
    TP = int(np.sum((p == 1) & (y == 1)))
    TN = int(np.sum((p == 0) & (y == 0)))
    FP = int(np.sum((p == 1) & (y == 0)))
    FN = int(np.sum((p == 0) & (y == 1)))

    def f1_score(precision, recall):
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    accuracy = (TP + TN) / len(y)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0        # home precision
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0           # home recall
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0      # away recall
    precision_away = TN / (TN + FN) if TN + FN > 0 else 0.0
    f1 = f1_score(precision, recall)
    f1_away = f1_score(precision_away, specificity)
    baseline = max(np.mean(y), 1 - np.mean(y))   # always guess the majority class

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "accuracy": accuracy, "baseline_accuracy": baseline,
            "precision": precision, "recall": recall, "specificity": specificity,
            "precision_away": precision_away,
            "balanced_accuracy": (recall + specificity) / 2,
            "macro_precision": (precision + precision_away) / 2,
            "f1": f1, "f1_away": f1_away, "macro_f1": (f1 + f1_away) / 2}


def choose_threshold(Y_true, proba):
    """Try every threshold on the threshold window, keep the best balanced accuracy."""
    best_thr, best_score = 0.5, -1
    for thr in THRESHOLDS:
        score = compute_metrics(Y_true, (proba > thr).astype(int))["balanced_accuracy"]
        if score > best_score:
            best_thr, best_score = thr, score
    return float(best_thr), best_score


def print_metrics(m, title):
    print(f"\n--- {title} ---")
    print(f"  Confusion matrix:      predicted 0   predicted 1")
    print(f"    actual 0 (away)      {m['TN']:>11}   {m['FP']:>11}")
    print(f"    actual 1 (present)   {m['FN']:>11}   {m['TP']:>11}")
    print(f"  Accuracy:          {m['accuracy']:.3f}   (baseline {m['baseline_accuracy']:.3f})")
    print(f"  Balanced accuracy: {m['balanced_accuracy']:.3f}")
    print(f"  Precision:         {m['precision']:.3f}")
    print(f"  Recall:            {m['recall']:.3f}")
    print(f"  F1 (home):         {m['f1']:.3f}")
    print(f"  Macro F1:          {m['macro_f1']:.3f}")


# =============================================================================
# 6. ONE FOLD: fit -> early stopping -> threshold -> score
# =============================================================================
def run_fold(df, feature_names, layers_dims, w, learning_rate):
    fit_df = df[df["time"] < pd.Timestamp(w["fit_end"])]
    stop_df = select(df, *w["stop"])
    thr_df = select(df, *w["threshold"])
    score_df = select(df, *w["score"])

    X_fit, Y_fit = make_xy(fit_df, feature_names)
    X_stop, Y_stop = make_xy(stop_df, feature_names)
    X_thr, Y_thr = make_xy(thr_df, feature_names)
    X_score, Y_score = make_xy(score_df, feature_names)

    # Normalization: min and max come from the fit window only (no leakage)
    x_min, x_max = fit_min_max(X_fit)
    X_fit = apply_min_max(X_fit, x_min, x_max)
    X_stop = apply_min_max(X_stop, x_min, x_max)
    X_thr = apply_min_max(X_thr, x_min, x_max)
    X_score = apply_min_max(X_score, x_min, x_max)

    parameters, best_iter, history = L_layer_model(X_fit, Y_fit, layers_dims, X_stop, Y_stop,
                                                   learning_rate=learning_rate)
    threshold, thr_bal_acc = choose_threshold(Y_thr, predict_proba(X_thr, parameters))
    proba_score = predict_proba(X_score, parameters)
    metrics = compute_metrics(Y_score, (proba_score > threshold).astype(int))

    info = {"n_fit": len(fit_df), "n_stop": len(stop_df), "n_thr": len(thr_df),
            "n_score": len(score_df), "best_iteration": best_iter,
            "threshold": threshold, "thr_bal_acc": thr_bal_acc}
    return metrics, info, history, score_df, Y_score, proba_score


# =============================================================================
# 7. GRID SEARCH (L7) on the 3 validation folds only
# =============================================================================
def grid_search(df, feature_names):
    val_folds = {n: w for n, w in WINDOWS.items() if n != FINAL_NAME}
    results = []
    for lr in LEARNING_RATES:
        for dims in ARCHITECTURES:
            dims = [len(feature_names)] + dims[1:]
            scores, iterations = [], []
            for name, w in val_folds.items():
                m, info, *_ = run_fold(df, feature_names, dims, w, lr)
                scores.append(m[SELECT_METRIC])
                iterations.append(info["best_iteration"])
            r = {"lr": lr, "dims": dims, "scores": scores,
                 "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                 "iterations": iterations,
                 # Kept parameters near the iteration limit -> still learning,
                 # the combination may lose only because it ran out of iterations.
                 "hit_limit": max(iterations) >= MAX_ITERATIONS - PATIENCE}
            results.append(r)
            flag = "   <- still improving near the iteration limit" if r["hit_limit"] else ""
            print(f"  lr = {lr:<6}  {str(dims):18s}  {SELECT_METRIC} = "
                  f"{r['mean']:.3f} +/- {r['std']:.3f}   folds {np.round(scores, 3)}   "
                  f"kept iterations {iterations}{flag}")
    best = max(results, key=lambda r: r["mean"])
    return best, results


# =============================================================================
# 8. PLOTS (same style as the slides)
# =============================================================================
def plot_windows(df):
    """Same layout as the shallow-NN figure, so the two can be shown side by side."""
    colors = {"fit": "#2166ac", "stop": "#1b9e77", "threshold": "#d95f02", "score": "#7570b3"}
    names = list(WINDOWS)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    start = df["time"].min()
    for i, name in enumerate(names):
        y = len(names) - 1 - i
        w = WINDOWS[name]
        spans = {"fit": (start, pd.Timestamp(w["fit_end"])),
                 "stop": tuple(map(pd.Timestamp, w["stop"])),
                 "threshold": tuple(map(pd.Timestamp, w["threshold"])),
                 "score": (pd.Timestamp(w["score"][0]),
                           min(pd.Timestamp(w["score"][1]), df["time"].max()))}
        for key, (a, b) in spans.items():
            ax.barh(y, mdates.date2num(b) - mdates.date2num(a), left=mdates.date2num(a),
                    height=0.55, color=colors[key])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.set_xlabel("2020; each white gap between stages is 24 hours")
    ax.set_title("Chronological training, stopping, threshold and scoring (DNN)")
    ax.legend(handles=[Patch(color=colors["fit"], label="Fit weights"),
                       Patch(color=colors["stop"], label="Early stopping"),
                       Patch(color=colors["threshold"], label="Choose threshold"),
                       Patch(color=colors["score"], label="Score")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(FOLDER / "DNN_grid_windows.png", dpi=150, bbox_inches="tight")
    return fig


def plot_grid_table(results, best):
    """The grid from the slides, filled in with mean +/- std over the 3 folds."""
    table = np.array([[next(r["mean"] for r in results
                            if r["lr"] == lr and r["dims"][1:] == dims[1:])
                       for dims in ARCHITECTURES] for lr in LEARNING_RATES])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(table, cmap="Blues", aspect="auto",
              vmin=table.min() - 0.05, vmax=table.max())
    for i, lr in enumerate(LEARNING_RATES):
        for j, dims in enumerate(ARCHITECTURES):
            r = next(r for r in results if r["lr"] == lr and r["dims"][1:] == dims[1:])
            is_best = r is best
            text = (f"{r['mean']:.1%}" + (" *" if r["hit_limit"] else "")
                    + f"\n± {r['std']:.1%}" + ("\n(best)" if is_best else ""))
            ax.text(j, i, text, ha="center", va="center", fontsize=11,
                    fontweight="bold" if is_best else "normal",
                    color="white" if r["mean"] > table.min() + 0.6 * (table.max() - table.min()) else NAVY)
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#DB4437", linewidth=3))
    ax.set_xticks(range(len(ARCHITECTURES)))
    ax.set_xticklabels([str(d) for d in ARCHITECTURES])
    ax.set_yticks(range(len(LEARNING_RATES)))
    ax.set_yticklabels([rf"$\alpha$ = {lr}" for lr in LEARNING_RATES], fontsize=11)
    ax.xaxis.tick_top()
    nice = SELECT_METRIC.replace("_", " ")
    ax.set_title(f"Grid search: mean {nice} ± std over the 3 validation folds",
                 color=NAVY, fontweight="bold", pad=30)
    if any(r["hit_limit"] for r in results):
        fig.text(0.5, 0.01, f"* still improving near the {MAX_ITERATIONS}-iteration limit "
                 "in at least one fold (may be under-trained)",
                 ha="center", fontsize=9, color="grey")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FOLDER / "DNN_grid_search_table.png", dpi=150, bbox_inches="tight")
    return fig


def plot_learning_curve(history, best_iteration, title, lr):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["iteration"], history["train_cost"], label="Fit-weights window (training)")
    ax.plot(history["iteration"], history["stop_cost"], label="Early-stopping window")
    ax.axvline(best_iteration, color="grey", linestyle="--",
               label=f"Kept parameters (iteration {best_iteration})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J (cross-entropy)")
    ax.set_title(f"Learning curve, {title} (lr = {lr})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FOLDER / "DNN_grid_learning_curve.png", dpi=150, bbox_inches="tight")
    return fig


def plot_performance(m, subtitle):
    """Confusion matrix (left) and grouped metric bars (right), as in the slides."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1, 1.6]})

    # Confusion matrix with the share of each actual class
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    shares = cm / cm.sum(axis=1, keepdims=True)
    ax1.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i, label in enumerate(["Away", "Home"]):
        for j in range(2):
            ax1.text(j, i, f"{cm[i, j]}\n{shares[i, j]:.1%}\nof actual {label}",
                     ha="center", va="center", fontsize=10,
                     color="white" if shares[i, j] > 0.5 else NAVY)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Away", "Home"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["Away", "Home"])
    ax1.set_xlabel("Predicted occupancy"); ax1.set_ylabel("Actual occupancy")
    ax1.set_title("Confusion matrix", color=NAVY, fontweight="bold")

    # Metric bars in groups: (label, value, colour, bold)
    groups = [[("Balanced accuracy", m["balanced_accuracy"], NAVY, True),
               ("Home recall", m["recall"], MID_BLUE, False),
               ("Away recall", m["specificity"], LIGHT_BLUE, False)],
              [("Macro precision", m["macro_precision"], NAVY, True),
               ("Home precision", m["precision"], MID_BLUE, False),
               ("Away precision", m["precision_away"], LIGHT_BLUE, False)],
              [("Macro F1", m["macro_f1"], NAVY, True),
               ("Home F1", m["f1"], MID_BLUE, False),
               ("Away F1", m["f1_away"], LIGHT_BLUE, False)],
              [("Accuracy", m["accuracy"], NAVY, True)]]
    y, ticks, labels, bold = 0, [], [], []
    for g, group in enumerate(groups):
        for label, value, colour, is_bold in group:
            ax2.barh(y, value, color=colour, height=0.65)
            ax2.text(value + 0.01, y, f"{value:.1%}", va="center",
                     fontweight="bold", color=NAVY)
            ticks.append(y); labels.append(label); bold.append(is_bold)
            y += 1
        if g < len(groups) - 1:
            ax2.axhline(y - 0.0, color="#DDE3EA", linewidth=1)
            y += 0.6
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels)
    for tick, is_bold in zip(ax2.get_yticklabels(), bold):
        tick.set_fontweight("bold" if is_bold else "normal")
        tick.set_color(NAVY)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.08)
    ax2.set_xticklabels([])
    ax2.grid(axis="x", color="#DDE3EA")
    ax2.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax2.spines[side].set_visible(False)

    fig.suptitle(f"Deep Neural Network: performance on the test week  ({subtitle})",
                 color=NAVY, fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(FOLDER / "DNN_grid_performance.png", dpi=150, bbox_inches="tight")
    return fig


def plot_timeseries(score_df, Y, proba, threshold):
    """Predicted probability vs actual presence, and the occupancy-state strip below."""
    times = score_df["time"].values
    y_true = Y.ravel()
    y_pred = (proba.ravel() > threshold).astype(int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.45})
    ax1.plot(times, proba.ravel(), color=MID_BLUE, linewidth=1.2, label="Predicted P(Home)")
    ax1.plot(times, y_true, color="#555555", linewidth=0.8, drawstyle="steps-post",
             label="Actual presence (1 = Home)")
    ax1.axhline(threshold, color="#C0392B", linestyle="--", linewidth=1.3,
                label=f"Classification threshold = {threshold:.2f}")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Probability / actual state")
    ax1.grid(color="#EEEEEE")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    # Occupancy states: dark blue = Home, pale blue = Away
    from matplotlib.colors import ListedColormap
    states = np.vstack([y_true, y_pred])
    t_num = mdates.date2num(pd.to_datetime(times))
    t_edges = np.append(t_num, t_num[-1] + (t_num[-1] - t_num[-2]))
    ax2.pcolormesh(t_edges, [0, 1, 2], states, cmap=ListedColormap(["#E3ECF5", MID_BLUE]),
                   vmin=0, vmax=1, shading="flat")
    ax2.set_yticks([0.5, 1.5]); ax2.set_yticklabels(["Actual", "Predicted"])
    ax2.invert_yaxis()
    ax2.set_title("Occupancy states: blue = Home, pale blue = Away",
                  loc="left", fontsize=10, color=NAVY)
    ax2.set_xlabel("Date (historical test week)")
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.savefig(FOLDER / "DNN_grid_test_week_timeseries.png", dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    df = load_data(DATA_FILE)
    feature_names = list(FEATURE_KEYWORDS)

    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Early stopping: check every {CHECK_EVERY} it., patience {PATIENCE} it., "
          f"max {MAX_ITERATIONS} it.")
    print(f"Share of 'present' in the whole dataset: {df['presence'].mean():.1%}")

    # 1) Grid search on the 3 validation folds (the test week is not used here)
    n_runs = len(LEARNING_RATES) * len(ARCHITECTURES) * (len(WINDOWS) - 1)
    print(f"\n{'=' * 64}\nGRID SEARCH: {n_runs} training runs, compared on mean "
          f"{SELECT_METRIC}\n{'=' * 64}")
    best, results = grid_search(df, feature_names)
    print(f"\nBest: lr = {best['lr']}, architecture {best['dims']}, "
          f"{SELECT_METRIC} = {best['mean']:.3f} +/- {best['std']:.3f}")

    # 2) The winner is run ONCE on the final historical check (test week)
    m, info, history, score_df, Y_score, proba = run_fold(
        df, feature_names, best["dims"], WINDOWS[FINAL_NAME], best["lr"])
    print(f"\n{'=' * 64}\n{FINAL_NAME} (test week), best combination only\n{'=' * 64}")
    print(f"  Rows: fit {info['n_fit']}, early stop {info['n_stop']}, "
          f"threshold {info['n_thr']}, score {info['n_score']}")
    print(f"  Kept parameters from iteration {info['best_iteration']}")
    print(f"  Threshold {info['threshold']:.2f} "
          f"(balanced accuracy {info['thr_bal_acc']:.3f} on the threshold window)")
    print_metrics(m, f"{FINAL_NAME}: score window")

    # 3) Plots
    subtitle = f"lr = {best['lr']}, layers {best['dims']}"
    plot_grid_table(results, best)
    plot_windows(df)
    plot_learning_curve(history, info["best_iteration"], FINAL_NAME, best["lr"])
    plot_performance(m, subtitle)
    plot_timeseries(score_df, Y_score, proba, info["threshold"])
    print(f"\nFigures saved in: {FOLDER}")
    plt.show()


if __name__ == "__main__":
    main()
