
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

def evaluate_models():

    start = time.time()
    print("Loading test data...", flush=True)

    # Load only a sample for visualization
    X_test = pd.read_csv("data/X_test.csv").sample(
        n=20000, random_state=42
    )

    y_test = pd.read_csv("data/y_test.csv").iloc[
        X_test.index, 0
    ]

    print("Loading saved models...", flush=True)

    encoder = joblib.load("models/label_encoder.pkl")
    y_test_encoded = encoder.transform(y_test)

    rf_model = joblib.load("models/random_forest.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")

    print("Generating Random Forest predictions...", flush=True)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    print("Generating XGBoost predictions...", flush=True)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    os.makedirs("figures", exist_ok=True)

    # Confusion matrix comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pred, title in [
        (axes[0], rf_pred, "Random Forest"),
        (axes[1], xgb_pred, "XGBoost")
    ]:
        cm = confusion_matrix(y_test_encoded, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("figures/confusion_matrix_comparison.png", dpi=200)
    plt.close()

    print("Confusion matrix saved.", flush=True)

    # ROC curve comparison
    plt.figure(figsize=(8, 6))

    for prob, name in [
        (rf_prob, "Random Forest"),
        (xgb_prob, "XGBoost")
    ]:
        fpr, tpr, _ = roc_curve(y_test_encoded, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/roc_curve_comparison.png", dpi=200)
    plt.close()

    print("ROC curve saved.", flush=True)
    print(f"Evaluation completed in {time.time()-start:.2f} seconds.")

if __name__ == "__main__":
    evaluate_models()
