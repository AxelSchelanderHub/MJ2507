"""
Occupancy detection with a Deep Neural Network (MJ2507)
=======================================================

Uses the same DNN functions as the course assignment
"Building your Deep Neural Network: Step by Step":
    initialize_parameters_deep, linear_forward, linear_activation_forward,
    L_model_forward, compute_cost, linear_backward, linear_activation_backward,
    L_model_backward, update_parameters

Architecture: [LINEAR -> RELU] x (L-1) -> LINEAR -> SIGMOID

Evaluation:
    - Expanding-window time series cross-validation (4 folds)
    - Embargo gap between train/validation/test to prevent leakage
    - Threshold tuned on inner validation set
    - Metrics: accuracy, balanced accuracy, macro F1, ROC AUC
    - Final evaluation on held-out test set (last 7 days)
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# SETTINGS
# =============================================================================
DATA_FILE = r"C:\Users\Gabriel\Desktop\AI energy course\AI project\TEST 1 Cleaned final data for models.xlsx"
TEST_DAYS = 7
USE_TIME_FEATURES = False

# DNN hyperparameters
LAYERS_DIMS = [5, 20, 10, 1]
LEARNING_RATE = 0.1
NUM_ITERATIONS = 3000

# Cross-validation settings
N_FOLDS = 4
EMBARGO_HOURS = 6         # gap between sections to prevent leakage
VAL_DAYS = 3              # validation window size per fold
OUTER_TEST_DAYS = 3       # outer scoring window size per fold

# Column keywords
TIME_KEYWORD = "Timestep"
TARGET_KEYWORD = "Presence"
FEATURE_KEYWORDS = {
    "CO2": "CO2",
    "Light": "Light",
    "Humidity": "Humidity",
    "Indoor temp": "Indoor",
    "Outdoor temp": "Outdoor",
}


# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
def find_column(df, keyword):
    for col in df.columns:
        if keyword.lower() in str(col).lower():
            return col
    return None


def load_data(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, sep=";", decimal=",")
    else:
        df = pd.read_excel(path)

    rename = {find_column(df, TIME_KEYWORD): "time",
              find_column(df, TARGET_KEYWORD): "presence"}
    for short_name, keyword in FEATURE_KEYWORDS.items():
        col = find_column(df, keyword)
        if col is not None:
            rename[col] = short_name
    if None in rename:
        raise ValueError("Could not find the time or presence column.")
    df = df.rename(columns=rename)[list(rename.values())]

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    features = [c for c in FEATURE_KEYWORDS if c in df.columns]
    empty = [c for c in features if df[c].isna().all()]
    if empty:
        print(f"Note: these columns are empty and are skipped: {empty}")
        df = df.drop(columns=empty)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        print(f"Note: dropped {before - len(df)} rows with missing values.")

    df["presence"] = df["presence"].astype(int)
    return df


def add_time_features(df):
    hour = df["time"].dt.hour + df["time"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    return df


def split_train_test(df, test_days):
    split_time = df["time"].max() - pd.Timedelta(days=test_days)
    train = df[df["time"] <= split_time]
    test = df[df["time"] > split_time]
    return train, test


# =============================================================================
# 2. DNN HELPER FUNCTIONS (same as HW3_1 Step-by-Step)
# =============================================================================

def sigmoid(Z):
    A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s, _ = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
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
    AL = np.clip(AL, 1e-12, 1 - 1e-12)
    cost = -(1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost


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
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    AL = np.clip(AL, 1e-12, 1 - 1e-12)
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
    parameters = params.copy()
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


# =============================================================================
# 3. THE L-LAYER MODEL
# =============================================================================
def L_layer_model(X, Y, layers_dims, learning_rate=0.5, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0 or i == num_iterations - 1:
            costs.append(cost)
            if print_cost and i % 500 == 0:
                print(f"  Cost after iteration {i}: {cost:.6f}")

    return parameters, costs


# =============================================================================
# 4. PREDICTION AND METRICS
# =============================================================================
def predict_proba(X, parameters):
    AL, _ = L_model_forward(X, parameters)
    return AL


def predict(X, parameters, threshold=0.5):
    proba = predict_proba(X, parameters)
    return (proba > threshold).astype(int)


def compute_metrics(Y_true, Y_pred, Y_proba=None):
    """Compute accuracy, balanced accuracy, precision, recall, macro F1, and ROC AUC."""
    y, p = Y_true.ravel(), Y_pred.ravel()
    TP = np.sum((p == 1) & (y == 1))
    TN = np.sum((p == 0) & (y == 0))
    FP = np.sum((p == 1) & (y == 0))
    FN = np.sum((p == 0) & (y == 1))

    accuracy = (TP + TN) / len(y)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
    balanced_acc = (recall + specificity) / 2

    f1_pos = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    prec_neg = TN / (TN + FN) if TN + FN > 0 else 0.0
    rec_neg = specificity
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if prec_neg + rec_neg > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2

    roc_auc = None
    if Y_proba is not None:
        roc_auc = compute_roc_auc(y, Y_proba.ravel())

    return {
        "accuracy": accuracy, "balanced_accuracy": balanced_acc,
        "precision": precision, "recall": recall,
        "specificity": specificity, "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
    }


def compute_roc_auc(y_true, y_scores):
    """ROC AUC computed from scratch (no sklearn dependency for this)."""
    desc_indices = np.argsort(-y_scores)
    y_sorted = y_true[desc_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def find_best_threshold(Y_true, Y_proba, metric="macro_f1"):
    """Search thresholds on validation data to maximize the chosen metric."""
    best_thr = 0.5
    best_score = -1
    for thr in np.arange(0.05, 0.95, 0.01):
        preds = (Y_proba > thr).astype(int)
        m = compute_metrics(Y_true, preds)
        if m[metric] > best_score:
            best_score = m[metric]
            best_thr = thr
    return best_thr, best_score


def print_metrics(metrics, name="Test"):
    print(f"\n--- {name} ---")
    print(f"  Confusion matrix:      predicted 0   predicted 1")
    print(f"    actual 0 (empty)     {metrics['TN']:>11}   {metrics['FP']:>11}")
    print(f"    actual 1 (present)   {metrics['FN']:>11}   {metrics['TP']:>11}")
    print(f"  Accuracy:          {metrics['accuracy']:.3f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"  Precision:         {metrics['precision']:.3f}")
    print(f"  Recall:            {metrics['recall']:.3f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.3f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC AUC:           {metrics['roc_auc']:.3f}")


# =============================================================================
# 5. TIME SERIES CROSS-VALIDATION
# =============================================================================
def expanding_window_cv(df, feature_names, layers_dims, n_folds=4,
                        val_days=3, outer_test_days=3, embargo_hours=6,
                        learning_rate=0.5, num_iterations=3000):
    """
    Expanding-window time series cross-validation.

    For each fold:
      - Training: everything from data start up to a cutoff (expands each fold)
      - [embargo gap]
      - Inner validation: val_days (used to tune the threshold)
      - [embargo gap]
      - Outer test: outer_test_days (used to score the fold)

    Returns per-fold metrics and the average best threshold.
    """
    embargo = pd.Timedelta(hours=embargo_hours)
    val_window = pd.Timedelta(days=val_days)
    test_window = pd.Timedelta(days=outer_test_days)

    # Total window needed per fold: embargo + val + embargo + test
    fold_window = embargo + val_window + embargo + test_window

    # The held-out test set (last TEST_DAYS) is never touched in CV
    cv_end = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
    cv_data = df[df["time"] <= cv_end].copy()

    cv_start = cv_data["time"].min()
    cv_duration = cv_end - cv_start
    usable = cv_duration - fold_window  # time available for expanding the training set

    fold_results = []
    fold_splits = []

    for fold in range(n_folds):
        # Training end grows with each fold
        train_end = cv_start + (usable * (fold + 1) / n_folds)
        val_start = train_end + embargo
        val_end = val_start + val_window
        test_start = val_end + embargo
        test_end = test_start + test_window

        train_df = cv_data[cv_data["time"] <= train_end]
        val_df = cv_data[(cv_data["time"] >= val_start) & (cv_data["time"] < val_end)]
        test_df = cv_data[(cv_data["time"] >= test_start) & (cv_data["time"] < test_end)]

        if len(train_df) < 100 or len(val_df) < 10 or len(test_df) < 10:
            print(f"  Fold {fold + 1}: skipped (not enough data)")
            continue

        fold_splits.append({
            "fold": fold + 1,
            "train_start": train_df["time"].min(), "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
            "test_start": test_start, "test_end": test_end,
            "train_n": len(train_df), "val_n": len(val_df), "test_n": len(test_df),
        })

        # Normalize: fit scaler on training only
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(train_df[feature_names].values).T
        X_val = np.clip(scaler.transform(val_df[feature_names].values).T, 0, 1)
        X_test = np.clip(scaler.transform(test_df[feature_names].values).T, 0, 1)
        Y_train = train_df["presence"].values.reshape(1, -1).astype(float)
        Y_val = val_df["presence"].values.reshape(1, -1).astype(float)
        Y_test = test_df["presence"].values.reshape(1, -1).astype(float)

        print(f"\n  Fold {fold + 1}: train n={len(train_df)}, val n={len(val_df)}, test n={len(test_df)}")

        # Train
        parameters, costs = L_layer_model(
            X_train, Y_train, layers_dims,
            learning_rate=learning_rate, num_iterations=num_iterations, print_cost=False)
        print(f"    Final cost: {costs[-1]:.6f}")

        # Tune threshold on validation set
        val_proba = predict_proba(X_val, parameters)
        best_thr, best_val_score = find_best_threshold(Y_val, val_proba, metric="macro_f1")
        print(f"    Best threshold (from validation): {best_thr:.2f}  (val macro F1: {best_val_score:.3f})")

        # Score on outer test with tuned threshold
        test_proba = predict_proba(X_test, parameters)
        test_preds = (test_proba > best_thr).astype(int)
        metrics = compute_metrics(Y_test, test_preds, test_proba)
        metrics["threshold"] = best_thr
        fold_results.append(metrics)

        print(f"    Outer test — Acc: {metrics['accuracy']:.3f}  Bal.Acc: {metrics['balanced_accuracy']:.3f}"
              f"  Macro F1: {metrics['macro_f1']:.3f}  ROC AUC: {metrics['roc_auc']:.3f}")

    return fold_results, fold_splits


# =============================================================================
# 6. PLOTTING
# =============================================================================
def plot_cv_folds(fold_splits, cv_end):
    """Visualize the expanding training window like your friend's diagram."""
    fig, ax = plt.subplots(figsize=(12, 3))
    colors = {"train": "#4285F4", "val": "#F4A235", "test": "#34A853"}
    bar_height = 0.6

    for i, s in enumerate(fold_splits):
        y = len(fold_splits) - 1 - i
        # Training bar
        ax.barh(y, (s["train_end"] - s["train_start"]).total_seconds() / 86400,
                left=mdates.date2num(s["train_start"]),
                height=bar_height, color=colors["train"], edgecolor="white")
        # Validation bar
        ax.barh(y, (s["val_end"] - s["val_start"]).total_seconds() / 86400,
                left=mdates.date2num(s["val_start"]),
                height=bar_height, color=colors["val"], edgecolor="white")
        # Test bar
        ax.barh(y, (s["test_end"] - s["test_start"]).total_seconds() / 86400,
                left=mdates.date2num(s["test_start"]),
                height=bar_height, color=colors["test"], edgecolor="white")

    ax.set_yticks(range(len(fold_splits)))
    ax.set_yticklabels([f"Fold {s['fold']}  (train n={s['train_n']})"
                        for s in reversed(fold_splits)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.autofmt_xdate()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["train"], label="Expanding training"),
        Patch(facecolor=colors["val"], label="Inner validation"),
        Patch(facecolor=colors["test"], label="Outer scoring"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", ncol=3)
    ax.set_title(f"Expanding training window  •  embargo gaps omitted from all sections")
    plt.tight_layout()
    return fig


def plot_final_evaluation(test_df, Y_test, proba_test, threshold, metrics):
    """Plot the held-out test: probability vs actual, confusion matrix, and metrics bar."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Home occupancy model  |  chronological evaluation", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.94, f"Observed vs predicted occupancy  •  all {len(Y_test.ravel())} test readings",
             ha="center", fontsize=10, color="grey")

    # Top: time series
    ax1 = fig.add_axes([0.06, 0.52, 0.90, 0.38])
    times = test_df["time"].values
    ax1.plot(times, proba_test.ravel(), color="#4285F4", linewidth=0.8, label="Predicted P(Home)")
    ax1.plot(times, Y_test.ravel(), color="#333333", linewidth=0.8, alpha=0.7,
             drawstyle="steps-post", label="Actual occupancy (1=Home)")
    ax1.axhline(threshold, color="#DB4437", linestyle="--", linewidth=1,
                label=f"Classification threshold = {threshold:.2f}")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Home probability / actual status")
    ax1.set_xlabel("Date (held-out test only)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Bottom left: confusion matrix
    ax2 = fig.add_axes([0.06, 0.06, 0.40, 0.35])
    cm = np.array([[metrics["TN"], metrics["FP"]],
                    [metrics["FN"], metrics["TP"]]])
    ax2.imshow(cm, cmap="Blues", aspect="auto")
    labels_y = ["Away", "Home"]
    labels_x = ["Away", "Home"]
    total_away = metrics["TN"] + metrics["FP"]
    total_home = metrics["FN"] + metrics["TP"]
    for i in range(2):
        for j in range(2):
            total = total_away if i == 0 else total_home
            pct = cm[i, j] / total * 100 if total > 0 else 0
            actual_label = "Away" if i == 0 else "Home"
            ax2.text(j, i, f"{cm[i, j]}\n{pct:.0f}% of actual {actual_label}",
                     ha="center", va="center", fontsize=9,
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels_x)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(labels_y)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.set_title("Confusion matrix  •  held-out test", fontsize=10)

    # Bottom right: metrics bar chart
    ax3 = fig.add_axes([0.55, 0.06, 0.40, 0.35])
    baseline_acc = max(np.mean(Y_test.ravel()), 1 - np.mean(Y_test.ravel()))
    metric_names = ["Accuracy", "Baseline accuracy", "Balanced accuracy", "Macro F1", "ROC AUC"]
    metric_vals = [metrics["accuracy"], baseline_acc, metrics["balanced_accuracy"],
                   metrics["macro_f1"], metrics["roc_auc"] or 0]
    colors = ["#4285F4", "#9E9E9E", "#4285F4", "#4285F4", "#34A853"]
    bars = ax3.barh(range(len(metric_names)), metric_vals, color=colors, height=0.6)
    ax3.set_yticks(range(len(metric_names)))
    ax3.set_yticklabels(metric_names)
    ax3.set_xlim(0, 1.0)
    ax3.set_xlabel("Score (higher is better)")
    ax3.set_title("Model performance vs majority baseline", fontsize=10)
    ax3.invert_yaxis()
    for bar, val in zip(bars, metric_vals):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=9)

    plt.savefig("DNN_final_evaluation.png", dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    df = load_data(DATA_FILE)
    if USE_TIME_FEATURES:
        df = add_time_features(df)

    feature_names = [c for c in df.columns if c not in ("time", "presence")]
    n_features = len(feature_names)
    print(f"Features used ({n_features}): {feature_names}")

    layers_dims = LAYERS_DIMS.copy()
    layers_dims[0] = n_features

    # Split: held-out test = last TEST_DAYS, rest used for CV + final training
    train_all, test_final = split_train_test(df, TEST_DAYS)
    print(f"Total samples: {len(df)}")
    print(f"CV + training pool: {len(train_all)}  |  Held-out test: {len(test_final)}")
    print(f"Share of 'present' in training pool: {train_all['presence'].mean():.2%}")
    print(f"\nArchitecture: {layers_dims}")
    print(f"Learning rate: {LEARNING_RATE},  Iterations: {NUM_ITERATIONS}")

    # ----- CROSS-VALIDATION -----
    print(f"\n{'='*60}")
    print(f"EXPANDING-WINDOW CROSS-VALIDATION ({N_FOLDS} folds)")
    print(f"  Val window: {VAL_DAYS} days,  Outer test: {OUTER_TEST_DAYS} days,  Embargo: {EMBARGO_HOURS}h")
    print(f"{'='*60}")

    cv_end = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
    fold_results, fold_splits = expanding_window_cv(
        df, feature_names, layers_dims,
        n_folds=N_FOLDS, val_days=VAL_DAYS, outer_test_days=OUTER_TEST_DAYS,
        embargo_hours=EMBARGO_HOURS, learning_rate=LEARNING_RATE,
        num_iterations=NUM_ITERATIONS)

    if fold_results:
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        avg_thr = np.mean([r["threshold"] for r in fold_results])
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]:
            vals = [r[key] for r in fold_results if r[key] is not None]
            print(f"  {key:20s}:  {np.mean(vals):.3f}  (+/- {np.std(vals):.3f})")
        print(f"  {'avg threshold':20s}:  {avg_thr:.2f}")

        # Plot the CV folds diagram
        fig_folds = plot_cv_folds(fold_splits, cv_end)
    else:
        avg_thr = 0.5
        print("\nNo CV folds completed. Using default threshold 0.5")

    # ----- FINAL MODEL (train on all training data, evaluate on held-out test) -----
    print(f"\n{'='*60}")
    print(f"FINAL MODEL (trained on full training pool, tested on held-out week)")
    print(f"  Using threshold from CV: {avg_thr:.2f}")
    print(f"{'='*60}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(train_all[feature_names].values).T
    X_test = np.clip(scaler.transform(test_final[feature_names].values).T, 0, 1)
    Y_train = train_all["presence"].values.reshape(1, -1).astype(float)
    Y_test = test_final["presence"].values.reshape(1, -1).astype(float)

    parameters, costs = L_layer_model(
        X_train, Y_train, layers_dims,
        learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS, print_cost=True)

    proba_test = predict_proba(X_test, parameters)
    preds_test = (proba_test > avg_thr).astype(int)
    final_metrics = compute_metrics(Y_test, preds_test, proba_test)
    print_metrics(final_metrics, f"Held-out test (threshold={avg_thr:.2f})")

    # ----- PLOTS -----
    # Learning curve
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel("Cost J")
    plt.xlabel("Iterations (per hundreds)")
    plt.title(f"Learning curve (lr={LEARNING_RATE}, layers={layers_dims})")

    # Final evaluation dashboard
    fig_eval = plot_final_evaluation(test_final, Y_test, proba_test, avg_thr, final_metrics)

    plt.show()
    print(f"\nFinal evaluation plot saved to: DNN_final_evaluation.png")


if __name__ == "__main__":
    main()
