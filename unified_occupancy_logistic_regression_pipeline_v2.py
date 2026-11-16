"""
Unified Occupancy Detection with Logistic Regression (MJ2507)
=============================================================
This script implements a unified logistic regression pipeline for occupancy detection
using sensor data. It supports multiple sampling intervals, expanding-window cross-validation,
and threshold tuning to optimize balanced accuracy while maintaining parity between home and away recall."""

import copy
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. SETTINGS
# =============================================================================
DATA_FILE = "NEW Cleaned final data for models.xltx"  # Supports .xlsx or .csv
INTERVALS_MINUTES = [5, 10, 15]                       # Intervals to evaluate
PLOT_INTERVAL = 10                                    # Which interval to generate full plots for

TEST_DAYS = 7                 # Held-out test window (last N days)
NUM_ITERATIONS = 30000        # Gradient descent steps
LEARNING_RATE = 0.005          # Step size alpha (tuned for Z-score normalized inputs)

# Expanding-Window Cross-Validation Settings
N_FOLDS = 4
EMBARGO_HOURS = 24             # Buffer gap to prevent temporal leakage
VAL_DAYS = 3                  # Inner validation window for threshold tuning
OUTER_TEST_DAYS = 3           # Outer scoring window per fold


# =============================================================================
# 2. DATA LOADING & FEATURE ENGINEERING
# =============================================================================
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load Excel or CSV file, clean headers, parse European decimals, and sort chronologically."""
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path, sep=None, engine="python", decimal=",")
        else:
            df = pd.read_excel(file_path, sheet_name=0)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    df.columns = df.columns.astype(str).str.strip()

    time_col = None
    presence_col = None
    for col in df.columns:
        c_low = col.lower()
        if ("timestep" in c_low or "time" in c_low or "date" in c_low) and time_col is None:
            time_col = col
        if ("presence" in c_low or "occupancy" in c_low) and presence_col is None:
            presence_col = col

    if time_col is None or presence_col is None:
        raise ValueError(f"Could not find Timestep or Presence columns. Found: {list(df.columns)}")

    df = df.rename(columns={time_col: "time", presence_col: "presence"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Convert string/object numeric columns (handling comma decimals)
    for col in df.columns:
        if col not in ("time", "presence") and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df["presence"] = df["presence"].astype(int)
    return df


def prepare_interval_dataset(raw_df: pd.DataFrame, interval_minutes: int):
    """
    Resample data to `interval_minutes` and engineer physical + cyclical features.
    """
    df_res = raw_df.set_index("time")
    rule = f"{interval_minutes}min"

    numeric_cols = [c for c in df_res.columns if c != "presence"]
    agg_dict = {col: "mean" for col in numeric_cols}
    agg_dict["presence"] = lambda x: int(x.mean() >= 0.5)

    df_res = df_res.resample(rule).agg(agg_dict).dropna().reset_index()

    # Map core sensors & metadata columns flexibly
    feature_keywords = {
        "CO2": ["co2"],
        "Light": ["light"],
        "IndoorTemp": ["indoor temperature", "indoor temp"],
        "OutdoorTemp": ["outdoor temperature", "outdoor temp"],
        "Humidity": ["relative humidity", "humidity"],
        "WeekdayWeekend": ["weekday(0)/weekend(1)", "weekend"],
        "DayNight": ["day(0)/night(1)", "day/night"],
        "WeekdayNum": ["weekday number"]
    }

    rename_map = {}
    selected_features = []
    for short_name, keywords in feature_keywords.items():
        for col in df_res.columns:
            if any(kw in col.lower() for kw in keywords):
                rename_map[col] = short_name
                selected_features.append(short_name)
                break

    df_res = df_res.rename(columns=rename_map)

    # 1. Physical Derivative Feature: Rate of change of CO2 (detects decay when people leave)
    if "CO2" in df_res.columns:
        df_res["CO2_diff"] = df_res["CO2"].diff().fillna(0.0)
        selected_features.append("CO2_diff")

    # 2. Cyclical Time Features (smooth 24-hour clock encoding)
    hour = df_res["time"].dt.hour + df_res["time"].dt.minute / 60.0
    df_res["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_res["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    selected_features.extend(["hour_sin", "hour_cos"])

    # Fallback weekend flag if not present in Excel metadata
    if "WeekdayWeekend" not in selected_features:
        df_res["WeekdayWeekend"] = (df_res["time"].dt.dayofweek >= 5).astype(float)
        selected_features.append("WeekdayWeekend")

    keep_cols = ["time", "presence"] + selected_features
    df_res = df_res[keep_cols].dropna().reset_index(drop=True)
    return df_res, selected_features


def split_train_test(df: pd.DataFrame, test_days: int):
    """Chronological split: last `test_days` become the held-out test set."""
    split_time = df["time"].max() - pd.Timedelta(days=test_days)
    train = df[df["time"] <= split_time].copy()
    test = df[df["time"] > split_time].copy()
    return train, test


# =============================================================================
# 3. COURSE-COMPLIANT LOGISTIC REGRESSION (FROM SCRATCH)
# =============================================================================
def sigmoid(z):
    """Compute the sigmoid of z safely."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def initialize_with_zeros(dim):
    """Initialize weight vector w of shape (dim, 1) and scalar bias b = 0.0."""
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    """
    Forward and backward propagation for logistic regression.
    X -- (num_features, m)
    Y -- (1, m)
    """
    m = X.shape[1]

    # Forward pass
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    A = np.clip(A, 1e-12, 1.0 - 1e-12)
    cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))

    # Backward pass
    dZ = A - Y
    dw = (1.0 / m) * np.dot(X, dZ.T)
    db = (1.0 / m) * np.sum(dZ)

    grads = {"dw": dw, "db": db}
    return grads, float(np.squeeze(cost))


def optimize(w, b, X, Y, num_iterations=10000, learning_rate=0.05, print_cost=False):
    """Run gradient descent to learn w and b."""
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]

        if i % 500 == 0:
            costs.append(cost)
            if print_cost and i % 5000 == 0:
                print(f"    Iteration {i:>6}: cost = {cost:.6f}")

    params = {"w": w, "b": b}
    return params, grads, costs


def predict_proba(w, b, X):
    """Compute predicted probabilities P(y=1 | X)."""
    w = w.reshape(X.shape[0], 1)
    return sigmoid(np.dot(w.T, X) + b)


# =============================================================================
# 4. METRICS & VALIDATION THRESHOLD TUNING
# =============================================================================
def compute_roc_auc(y_true, y_scores):
    """Compute ROC AUC from scratch."""
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = fp = 0
    auc = prev_fpr = prev_tpr = 0.0
    for val in y_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return auc


def compute_metrics(Y_true, Y_pred, Y_proba=None):
    """Compute Accuracy, Balanced Accuracy, Precision, Home Recall, Away Recall, Macro F1, ROC AUC."""
    y, p = Y_true.ravel(), Y_pred.ravel()
    TP = int(np.sum((p == 1) & (y == 1)))
    TN = int(np.sum((p == 0) & (y == 0)))
    FP = int(np.sum((p == 1) & (y == 0)))
    FN = int(np.sum((p == 0) & (y == 1)))

    accuracy = (TP + TN) / len(y) if len(y) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_home = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    recall_away = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity
    balanced_acc = (recall_home + recall_away) / 2.0

    f1_pos = (2 * precision * recall_home / (precision + recall_home)) if (precision + recall_home) > 0 else 0.0
    prec_neg = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    f1_neg = (2 * prec_neg * recall_away / (prec_neg + recall_away)) if (prec_neg + recall_away) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0

    roc_auc = compute_roc_auc(y, Y_proba.ravel()) if Y_proba is not None else None

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall_home": recall_home,
        "recall_away": recall_away,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }


def find_best_threshold(Y_true, Y_proba, metric="balanced_accuracy", max_recall_gap=0.08):
    """
    Tune decision threshold on validation data to maximize Balanced Accuracy
    while enforcing that Home Recall and Away Recall stay within `max_recall_gap`
    of each other (preventing one class from dominating the other).
    """
    best_thr = 0.5
    best_score = -1.0
    fallback_thr = 0.5
    fallback_score = -1.0

    for thr in np.arange(0.15, 0.85, 0.01):
        preds = (Y_proba > thr).astype(int)
        m = compute_metrics(Y_true, preds)
        score = m[metric]

        # Track unconstrained best as fallback (in case a validation fold lacks one class)
        if score > fallback_score:
            fallback_score = score
            fallback_thr = thr

        # Enforce parity between Home Recall and Away Recall
        if abs(m["recall_home"] - m["recall_away"]) <= max_recall_gap:
            if score > best_score:
                best_score = score
                best_thr = thr

    if best_score < 0:
        return float(fallback_thr), float(fallback_score)

    return float(best_thr), float(best_score)


# =============================================================================
# 5. EXPANDING-WINDOW TIME SERIES CROSS-VALIDATION
# =============================================================================
def expanding_window_cv(df, feature_names, n_folds=4,
                        val_days=3, outer_test_days=3, embargo_hours=6,
                        learning_rate=0.05, num_iterations=20000):
    """
    Expanding-window time-series CV with embargo gaps.
    Tunes threshold on inner validation set to maximize Balanced Accuracy.
    """
    embargo = pd.Timedelta(hours=embargo_hours)
    val_window = pd.Timedelta(days=val_days)
    test_window = pd.Timedelta(days=outer_test_days)
    fold_window = embargo + val_window + embargo + test_window

    cv_end = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
    cv_data = df[df["time"] <= cv_end].copy()
    cv_start = cv_data["time"].min()
    usable = (cv_end - cv_start) - fold_window

    fold_results = []
    fold_splits = []

    for fold in range(n_folds):
        train_end = cv_start + (usable * (fold + 1) / n_folds)
        val_start = train_end + embargo
        val_end = val_start + val_window
        test_start = val_end + embargo
        test_end = test_start + test_window

        train_df = cv_data[cv_data["time"] <= train_end]
        val_df = cv_data[(cv_data["time"] >= val_start) & (cv_data["time"] < val_end)]
        test_df = cv_data[(cv_data["time"] >= test_start) & (cv_data["time"] < test_end)]

        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        fold_splits.append({
            "fold": fold + 1,
            "train_start": train_df["time"].min(), "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
            "test_start": test_start, "test_end": test_end,
            "train_n": len(train_df), "val_n": len(val_df), "test_n": len(test_df),
        })

        # Z-score standardization fitted strictly on training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_names].values).T
        X_val = scaler.transform(val_df[feature_names].values).T
        X_test = scaler.transform(test_df[feature_names].values).T

        Y_train = train_df["presence"].values.reshape(1, -1).astype(float)
        Y_val = val_df["presence"].values.reshape(1, -1).astype(float)
        Y_test = test_df["presence"].values.reshape(1, -1).astype(float)

        w, b = initialize_with_zeros(X_train.shape[0])
        params, _, _ = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)
        w, b = params["w"], params["b"]

        # Tune threshold on validation set for Balanced Accuracy
        val_proba = predict_proba(w, b, X_val)
        best_thr, _ = find_best_threshold(Y_val, val_proba, metric="balanced_accuracy", max_recall_gap=0.08)

        # Evaluate on outer fold test window
        test_proba = predict_proba(w, b, X_test)
        test_preds = (test_proba > best_thr).astype(int)
        metrics = compute_metrics(Y_test, test_preds, test_proba)
        metrics["threshold"] = best_thr
        fold_results.append(metrics)

        print(f"  Fold {fold+1} | Thr: {best_thr:.2f} | Acc: {metrics['accuracy']:.3f} | "
              f"Bal.Acc: {metrics['balanced_accuracy']:.3f} | HomeRec: {metrics['recall_home']:.3f} | "
              f"AwayRec: {metrics['recall_away']:.3f} | MacroF1: {metrics['macro_f1']:.3f}")

    return fold_results, fold_splits


# =============================================================================
# 6. DIAGNOSTIC PLOTTING
# =============================================================================
def plot_evaluation_dashboard(test_df, Y_test, proba_test, threshold, metrics, interval_min):
    """Generate the chronological prediction vs. actual dashboard + confusion matrix."""
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Unified Occupancy Logistic Regression ({interval_min}-min interval) | Chronological Evaluation",
                 fontsize=13, fontweight="bold")

    # Top panel: Time series probabilities & actual states
    ax1 = fig.add_axes([0.06, 0.52, 0.90, 0.38])
    times = test_df["time"].values
    ax1.plot(times, proba_test.ravel(), color="#2563eb", linewidth=1.0, label="Predicted P(Home)")
    ax1.fill_between(times, Y_test.ravel(), step="post", color="#94a3b8", alpha=0.35, label="Actual Occupancy (1=Home)")
    ax1.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.2,
                label=f"Tuned Decision Threshold = {threshold:.2f}")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Probability / Occupancy")
    ax1.legend(loc="upper right", ncol=3, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Bottom left: Confusion matrix
    ax2 = fig.add_axes([0.06, 0.08, 0.38, 0.34])
    cm = np.array([[metrics["TN"], metrics["FP"]],
                   [metrics["FN"], metrics["TP"]]])
    ax2.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            row_sum = cm[i].sum()
            pct = (cm[i, j] / row_sum * 100) if row_sum > 0 else 0
            label = "Away" if i == 0 else "Home"
            ax2.text(j, i, f"{cm[i, j]}\n({pct:.1f}% of {label})",
                     ha="center", va="center", fontsize=9,
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Predicted Away (0)", "Predicted Home (1)"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Actual Away (0)", "Actual Home (1)"])
    ax2.set_title("Confusion Matrix (Held-Out Test)", fontsize=10, fontweight="bold")

    # Bottom right: Key metrics comparison bar chart
    ax3 = fig.add_axes([0.53, 0.08, 0.43, 0.34])
    names = ["Accuracy", "Balanced Acc", "Recall (Home)", "Recall (Away)", "Macro F1", "ROC AUC"]
    vals = [
        metrics["accuracy"], metrics["balanced_accuracy"],
        metrics["recall_home"], metrics["recall_away"],
        metrics["macro_f1"], metrics["roc_auc"] or 0.0
    ]
    colors = ["#3b82f6", "#10b981", "#6366f1", "#f59e0b", "#8b5cf6", "#14b8a6"]
    bars = ax3.barh(range(len(names)), vals, color=colors, height=0.6)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names)
    ax3.set_xlim(0, 1.08)
    ax3.invert_yaxis()
    ax3.set_title("Held-Out Test Performance Breakdown", fontsize=10, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax3.text(v + 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.1%}", va="center", fontsize=9)

    out_file = f"Unified_LR_Evaluation_{interval_min}min.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved evaluation dashboard to '{out_file}'.")


# =============================================================================
# 7. MAIN EXECUTION LOOP ACROSS INTERVALS
# =============================================================================
def main():
    print(f"Loading raw dataset from '{DATA_FILE}'...")
    raw_df = load_and_clean_data(DATA_FILE)
    print(f"Raw rows loaded: {len(raw_df)} | Date range: {raw_df['time'].min()} to {raw_df['time'].max()}")

    summary_table = []

    for interval in INTERVALS_MINUTES:
        print("\n" + "=" * 75)
        print(f" EVALUATING {interval}-MINUTE SAMPLING INTERVAL")
        print("=" * 75)

        df_int, feature_names = prepare_interval_dataset(raw_df, interval)
        train_all, test_final = split_train_test(df_int, TEST_DAYS)

        print(f"Features ({len(feature_names)}): {feature_names}")
        print(f"Samples -> CV/Train pool: {len(train_all)} | Held-out test: {len(test_final)} "
              f"(Test Home Share: {test_final['presence'].mean():.1%})")

        # 1. Expanding-Window Cross-Validation & Threshold Tuning
        print("\nRunning Expanding-Window Cross-Validation...")
        cv_results, _ = expanding_window_cv(
            df_int, feature_names,
            n_folds=N_FOLDS, val_days=VAL_DAYS, outer_test_days=OUTER_TEST_DAYS,
            embargo_hours=EMBARGO_HOURS, learning_rate=LEARNING_RATE,
            num_iterations=NUM_ITERATIONS
        )

        if cv_results:
            tuned_threshold = float(np.mean([r["threshold"] for r in cv_results]))
            cv_bal_acc = float(np.mean([r["balanced_accuracy"] for r in cv_results]))
            print(f"  -> CV Mean Balanced Accuracy: {cv_bal_acc:.3f} | Optimal Tuned Threshold: {tuned_threshold:.2f}")
        else:
            tuned_threshold = 0.50
            print("  -> Insufficient data for CV folds; defaulting threshold to 0.50")

        # 2. Final Model Training on Full Training Pool
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_all[feature_names].values).T
        X_test = scaler.transform(test_final[feature_names].values).T
        Y_train = train_all["presence"].values.reshape(1, -1).astype(float)
        Y_test = test_final["presence"].values.reshape(1, -1).astype(float)

        w, b = initialize_with_zeros(X_train.shape[0])
        params, _, costs = optimize(
            w, b, X_train, Y_train,
            num_iterations=NUM_ITERATIONS, learning_rate=LEARNING_RATE, print_cost=False
        )
        w, b = params["w"], params["b"]

        # 3. Held-Out Test Evaluation using CV-Tuned Threshold
        proba_test = predict_proba(w, b, X_test)
        preds_test = (proba_test > tuned_threshold).astype(int)
        final_metrics = compute_metrics(Y_test, preds_test, proba_test)

        print(f"\nHeld-Out 7-Day Test Results ({interval}-min, Threshold={tuned_threshold:.2f}):")
        print(f"  Accuracy          : {final_metrics['accuracy']*100:.2f}%")
        print(f"  Balanced Accuracy : {final_metrics['balanced_accuracy']*100:.2f}%")
        print(f"  Precision         : {final_metrics['precision']*100:.2f}%")
        print(f"  Recall (Home = 1) : {final_metrics['recall_home']*100:.2f}%")
        print(f"  Recall (Away = 0) : {final_metrics['recall_away']*100:.2f}%")
        print(f"  Macro F1          : {final_metrics['macro_f1']*100:.2f}%")
        print(f"  ROC AUC           : {final_metrics['roc_auc']:.3f}")

        if interval == PLOT_INTERVAL:
            print("\nLearned Feature Weights (Z-score standardized):")
            for fname, weight in sorted(zip(feature_names, w.ravel()), key=lambda x: -abs(x[1])):
                print(f"  {fname:<18}: {weight:+.4f}")
            plot_evaluation_dashboard(test_final, Y_test, proba_test, tuned_threshold, final_metrics, interval)

        summary_table.append({
            "interval": interval,
            "threshold": tuned_threshold,
            "accuracy": final_metrics["accuracy"],
            "balanced_acc": final_metrics["balanced_accuracy"],
            "precision": final_metrics["precision"],
            "recall_home": final_metrics["recall_home"],
            "recall_away": final_metrics["recall_away"],
            "macro_f1": final_metrics["macro_f1"],
            "roc_auc": final_metrics["roc_auc"]
        })

    # Print Final Comparison Table
    print("\n" + "#" * 98)
    print(" UNIFIED MODEL SUMMARY ACROSS SAMPLING INTERVALS (HELD-OUT 7-DAY TEST)")
    print("#" * 98)
    print(f"{'Interval':<9} | {'Thr':<5} | {'Accuracy':<9} | {'Bal. Acc':<9} | {'Precision':<9} | {'Rec(Home)':<9} | {'Rec(Away)':<9} | {'Macro F1':<8} | {'AUC':<5}")
    print("-" * 98)
    for r in summary_table:
        print(f"{r['interval']:<6} min | {r['threshold']:<5.2f} | {r['accuracy']*100:7.2f}%  | "
              f"{r['balanced_acc']*100:7.2f}%  | {r['precision']*100:7.2f}%  | "
              f"{r['recall_home']*100:7.2f}%  | {r['recall_away']*100:7.2f}%  | "
              f"{r['macro_f1']*100:7.2f}%  | {r['roc_auc']:5.3f}")
    print("#" * 98)
    plt.show()


if __name__ == "__main__":
    main()