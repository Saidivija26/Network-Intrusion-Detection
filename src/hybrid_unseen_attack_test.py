import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/IoT_Intrusion_cleaned.csv"
BENIGN_LABEL = "BenignTraffic"
HELD_OUT_ATTACK = "MITM-ArpSpoofing"


def print_results(title, y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(f"\n{title}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[tn, fp], [fn, tp]])
    print(f"False alarms: {fp}")
    print(f"Missed attacks: {fn}")
    print(f"Attacks detected: {tp}")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            zero_division=0,
        )
    )


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    benign = df[df["label"] == BENIGN_LABEL].copy()
    held_out = df[df["label"] == HELD_OUT_ATTACK].copy()
    known_attacks = df[
        (df["label"] != BENIGN_LABEL)
        & (df["label"] != HELD_OUT_ATTACK)
    ].copy()

    if benign.empty:
        raise ValueError(f"No benign records found: {BENIGN_LABEL}")
    if held_out.empty:
        raise ValueError(f"No held-out attack records found: {HELD_OUT_ATTACK}")
    if known_attacks.empty:
        raise ValueError("No other attack records found for training.")

    # Keep some benign records separate for the final test.
    benign_train, benign_test = train_test_split(
        benign,
        test_size=0.20,
        random_state=42,
    )

    # Create a benign-majority training set with about 9 benign
    # records for every known attack record.
    attack_count = min(len(known_attacks), len(benign_train) // 9)
    attack_train = known_attacks.sample(
        n=attack_count,
        random_state=42,
    )

    # The final test set contains benign traffic and only the held-out
    # attack family.
    held_out_count = min(len(held_out), len(benign_test) // 9)
    attack_test = held_out.sample(
        n=held_out_count,
        random_state=42,
    )

    train_df = pd.concat(
        [benign_train, attack_train],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.concat(
        [benign_test, attack_test],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = train_df.drop(columns=["label"])
    y_train = (train_df["label"] != BENIGN_LABEL).astype(int)

    X_test = test_df.drop(columns=["label"])
    y_test = (test_df["label"] != BENIGN_LABEL).astype(int)

    print(f"\nAttack family held out from training: {HELD_OUT_ATTACK}")
    print("\nTraining counts:")
    print(y_train.map({0: "Benign", 1: "Attack"}).value_counts())
    print("\nTest counts:")
    print(y_test.map({0: "Benign", 1: "Attack"}).value_counts())

    # Keep validation data separate from model fitting and final testing.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
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

    X_fit_selected = selector.transform(X_fit)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    print(
        f"Selected {selector.get_support().sum()} "
        f"of {X_train.shape[1]} features."
    )

    # Supervised Random Forest trained on known attack families.
    print("\nTraining supervised Random Forest...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    classifier.fit(X_fit_selected, y_fit)

    attack_column = list(classifier.classes_).index(1)
    val_attack_probability = classifier.predict_proba(
        X_val_selected
    )[:, attack_column]

    # Select the supervised threshold using known attacks in validation.
    candidate_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    threshold_results = []

    print("\nValidation threshold comparison:")
    for threshold in candidate_thresholds:
        val_prediction = (
            val_attack_probability >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            val_prediction,
            labels=[0, 1],
        ).ravel()

        recall = tp / (tp + fn) if (tp + fn) else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0

        threshold_results.append(
            {
                "threshold": threshold,
                "recall": recall,
                "false_alarm_rate": false_alarm_rate,
            }
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"attack recall={recall:.4f}, "
            f"false alarm rate={false_alarm_rate:.4f}"
        )

    eligible = [
        result
        for result in threshold_results
        if result["recall"] >= 0.98
    ]

    if eligible:
        chosen_threshold = max(
            eligible,
            key=lambda result: result["threshold"],
        )["threshold"]
    else:
        chosen_threshold = max(
            threshold_results,
            key=lambda result: (result["recall"], result["threshold"]),
        )["threshold"]

    print(f"\nSelected supervised threshold: {chosen_threshold:.2f}")

    # Train the anomaly detector using benign training traffic only.
    print("\nTraining benign-only Isolation Forest...")
    benign_fit = X_fit.loc[y_fit == 0]

    anomaly_detector = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    anomaly_detector.fit(benign_fit)

    # Choose the anomaly cutoff using benign validation records only.
    benign_val = X_val.loc[y_val == 0]
    benign_val_scores = anomaly_detector.decision_function(benign_val)

    # The lowest 1% of benign validation scores define unusual traffic.
    anomaly_cutoff = float(pd.Series(benign_val_scores).quantile(0.01))
    print(f"Benign-only anomaly cutoff: {anomaly_cutoff:.6f}")

    # Evaluate both systems on the untouched held-out-family test set.
    test_attack_probability = classifier.predict_proba(
        X_test_selected
    )[:, attack_column]

    supervised_prediction = (
        test_attack_probability >= chosen_threshold
    ).astype(int)

    test_anomaly_score = anomaly_detector.decision_function(X_test)
    anomaly_prediction = (test_anomaly_score <= anomaly_cutoff).astype(int)

    # Hybrid alert: alert if either model flags the connection.
    hybrid_prediction = (
        (supervised_prediction == 1) | (anomaly_prediction == 1)
    ).astype(int)

    print_results(
        "Supervised Random Forest on unseen attack",
        y_test,
        supervised_prediction,
    )
    print_results(
        "Benign-only anomaly detector on unseen attack",
        y_test,
        anomaly_prediction,
    )
    print_results(
        "Combined hybrid system on unseen attack",
        y_test,
        hybrid_prediction,
    )

    print(
        "\nSupervised Attack ROC-AUC:",
        roc_auc_score(y_test, test_attack_probability),
    )
    print(
        "Anomaly score ROC-AUC:",
        roc_auc_score(y_test, -test_anomaly_score),
    )


if __name__ == "__main__":
    main()