import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/IoT_Intrusion_cleaned.csv"
BENIGN_LABEL = "BenignTraffic"
HELD_OUT_ATTACK = "MITM-ArpSpoofing"


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    benign = df[df["label"] == BENIGN_LABEL].copy()
    held_out = df[df["label"] == HELD_OUT_ATTACK].copy()
    known_attacks = df[
        (df["label"] != BENIGN_LABEL)
        & (df["label"] != HELD_OUT_ATTACK)
    ].copy()

    if benign.empty:
        raise ValueError(f"No rows found for benign label '{BENIGN_LABEL}'.")
    if held_out.empty:
        raise ValueError(
            f"No rows found for held-out attack '{HELD_OUT_ATTACK}'."
        )
    if known_attacks.empty:
        raise ValueError("No other attack types found for training.")

    # Keep benign records separate for training and final testing.
    benign_train, benign_test = train_test_split(
        benign,
        test_size=0.20,
        random_state=42,
    )

    # Use only known attacks for training. Aim for about 9 benign
    # records per attack record in the training data.
    train_attack_count = min(
        len(known_attacks),
        max(1, len(benign_train) // 9),
    )
    attack_train = known_attacks.sample(
        n=train_attack_count,
        random_state=42,
    )

    # Test on the held-out attack family plus benign test records.
    test_attack_count = min(
        len(held_out),
        max(1, len(benign_test) // 9),
    )
    attack_test = held_out.sample(
        n=test_attack_count,
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

    # Binary target: 0 = Benign, 1 = Attack.
    y_train = (train_df["label"] != BENIGN_LABEL).astype(int)
    y_test = (test_df["label"] != BENIGN_LABEL).astype(int)

    X_train = train_df.drop(columns=["label"])
    X_test = test_df.drop(columns=["label"])

    print(f"\nAttack family held out from training: {HELD_OUT_ATTACK}")
    print("\nTraining class counts:")
    print(y_train.map({0: "Benign", 1: "Attack"}).value_counts())
    print("\nTest class counts:")
    print(y_test.map({0: "Benign", 1: "Attack"}).value_counts())

    # Make validation data from training data for threshold selection.
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

    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} of {X_train.shape[1]} features.")

    X_fit_selected = selector.transform(X_fit)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    # Train using known attack families only.
    print("\nTraining model on known attack families...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_fit_selected, y_fit)

    attack_column = list(model.classes_).index(1)

    # Choose a threshold using validation data containing known attacks.
    val_attack_probabilities = model.predict_proba(X_val_selected)[:, attack_column]
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    validation_results = []

    print("\nValidation threshold comparison:")
    for threshold in thresholds:
        val_pred = (val_attack_probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_val,
            val_pred,
            labels=[0, 1],
        ).ravel()

        recall = tp / (tp + fn) if (tp + fn) else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0

        validation_results.append(
            {
                "threshold": threshold,
                "recall": recall,
                "false_alarm_rate": false_alarm_rate,
            }
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"attack recall={recall:.4f}, "
            f"false alarm rate={false_alarm_rate:.4f}, "
            f"missed attacks={fn}, false alarms={fp}"
        )

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

    chosen_threshold = selected["threshold"]
    print(f"\nSelected threshold: {chosen_threshold:.2f}")

    # Final evaluation: the held-out attack family was not used for training.
    test_attack_probabilities = model.predict_proba(X_test_selected)[:, attack_column]
    test_predictions = (
        test_attack_probabilities >= chosen_threshold
    ).astype(int)

    print(f"\nResults for unseen attack family: {HELD_OUT_ATTACK}")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"False alarms: {fp}")
    print(f"Missed unseen attacks: {fn}")
    print(f"Unseen attacks detected: {tp}")

    print(
        "Unseen attack ROC-AUC:",
        roc_auc_score(y_test, test_attack_probabilities),
    )
    print(
        "Unseen attack PR-AUC:",
        average_precision_score(y_test, test_attack_probabilities),
    )


if __name__ == "__main__":
    main()