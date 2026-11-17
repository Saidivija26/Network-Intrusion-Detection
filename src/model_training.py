import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_random_forest():
    print("Loading scenario training and testing datasets...")

    X_train = pd.read_csv("data/scenario_X_train.csv")
    X_test = pd.read_csv("data/scenario_X_test.csv")

    y_train = pd.read_csv("data/scenario_y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("data/scenario_y_test.csv").iloc[:, 0]

    # Encode target labels.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    print("Label order:", list(encoder.classes_))

    # Find the encoded label for Attack.
    attack_matches = [
        i
        for i, label in enumerate(encoder.classes_)
        if str(label).strip().lower() in {"attack", "attacks", "malicious"}
    ]

    if len(attack_matches) != 1:
        raise ValueError(
            "Could not identify exactly one Attack label. "
            f"Labels found: {list(encoder.classes_)}"
        )

    if len(encoder.classes_) != 2:
        raise ValueError(
            "This script requires binary labels: Attack and Benign/Normal. "
            f"Labels found: {list(encoder.classes_)}"
        )

    attack_index = attack_matches[0]
    other_index = next(i for i in range(2) if i != attack_index)

    print(f"Attack label: {encoder.classes_[attack_index]}")

    # Split training data to create validation data.
    # Keep the test set untouched until final evaluation.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_train_encoded,
    )

    # Select features using training data only.
    print("\nSelecting features...")

    selector = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        threshold="median",
    )
    selector.fit(X_fit, y_fit)

    selected_features = X_train.columns[selector.get_support()].tolist()

    if not selected_features:
        raise ValueError("Feature selection did not select any features.")

    print(f"Selected {len(selected_features)} of {X_train.shape[1]} features:")
    print(selected_features)

    # Apply the fitted selector to each dataset.
    X_fit_selected = selector.transform(X_fit)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_fit_selected, y_fit)

    print("Model training completed!")

    # Find the model probability column for Attack.
    attack_column = list(model.classes_).index(attack_index)

    # Compare thresholds using validation data only.
    val_attack_probabilities = model.predict_proba(X_val_selected)[:, attack_column]
    y_val_attack = (y_val == attack_index).astype(int)

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    validation_results = []

    print("\nValidation threshold comparison:")

    for candidate in thresholds:
        val_pred_attack = (
            val_attack_probabilities >= candidate
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val_attack,
            val_pred_attack,
            labels=[0, 1],
        ).ravel()

        recall = tp / (tp + fn) if (tp + fn) else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0

        validation_results.append(
            {
                "threshold": candidate,
                "recall": recall,
                "false_alarm_rate": false_alarm_rate,
            }
        )

        print(
            f"Threshold {candidate:.2f}: "
            f"attack recall={recall:.4f}, "
            f"false alarm rate={false_alarm_rate:.4f}, "
            f"missed attacks={fn}, false alarms={fp}"
        )

    # Select the highest threshold that maintains at least 98% attack recall.
    eligible = [
        result
        for result in validation_results
        if result["recall"] >= 0.98
    ]

    if eligible:
        selected = max(eligible, key=lambda result: result["threshold"])
    else:
        selected = max(
            validation_results,
            key=lambda result: (result["recall"], result["threshold"]),
        )

    threshold = selected["threshold"]

    print(
        f"\nSelected threshold: {threshold:.2f} "
        f"(validation attack recall={selected['recall']:.4f})"
    )

    # Evaluate on the untouched test data.
    attack_probabilities = model.predict_proba(X_test_selected)[:, attack_column]
    predicted_attack = (attack_probabilities >= threshold).astype(int)
    y_test_attack = (y_test_encoded == attack_index).astype(int)

    y_pred_encoded = [
        attack_index if is_attack else other_index
        for is_attack in predicted_attack
    ]

    print("\nFinal test results:")
    print(f"Threshold: {threshold:.2f}")
    print("Accuracy:", accuracy_score(y_test_encoded, y_pred_encoded))

    print("\nClassification report:")
    print(
        classification_report(
            y_test_encoded,
            y_pred_encoded,
            labels=[other_index, attack_index],
            target_names=[
                str(encoder.classes_[other_index]),
                str(encoder.classes_[attack_index]),
            ],
            zero_division=0,
        )
    )

    # Confusion matrix order: [[TN, FP], [FN, TP]].
    cm = confusion_matrix(
        y_test_attack,
        predicted_attack,
        labels=[0, 1],
    )
    tn, fp, fn, tp = cm.ravel()

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    print(f"True negatives (benign correctly identified): {tn}")
    print(f"False positives (false alarms): {fp}")
    print(f"False negatives (missed attacks): {fn}")
    print(f"True positives (attacks detected): {tp}")

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0

    print(f"False positive rate: {false_positive_rate:.4f}")
    print(f"False negative rate: {false_negative_rate:.4f}")
    print(
        "Attack ROC-AUC:",
        roc_auc_score(y_test_attack, attack_probabilities),
    )
    print(
        "Attack PR-AUC:",
        average_precision_score(y_test_attack, attack_probabilities),
    )

    # Save the model, encoder, feature selector, and selected threshold.
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")
    joblib.dump(encoder, "models/label_encoder.pkl")
    joblib.dump(selector, "models/feature_selector.pkl")

    with open("models/attack_threshold.txt", "w", encoding="utf-8") as file:
        file.write(str(threshold))

    print("\nModel, encoder, selector, and threshold saved successfully!")


if __name__ == "__main__":
    train_random_forest()