import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = "data/IoT_Intrusion_cleaned.csv"
BENIGN_LABEL = "BenignTraffic"
FEEDBACK_ATTACK = "MITM-ArpSpoofing"
FEEDBACK_FRACTION = 0.25


def train_and_evaluate(training_data, test_data, title):
    X = training_data.drop(columns=["label"])
    y = (training_data["label"] != BENIGN_LABEL).astype(int)

    X_test = test_data.drop(columns=["label"])
    y_test = (test_data["label"] != BENIGN_LABEL).astype(int)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Select features using fitting data only.
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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_fit_selected, y_fit)

    # Choose a threshold using validation data only.
    attack_column = list(model.classes_).index(1)
    val_probabilities = model.predict_proba(X_val_selected)[:, attack_column]

    threshold_results = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        val_predictions = (val_probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_val, val_predictions, labels=[0, 1]
        ).ravel()

        recall = tp / (tp + fn) if (tp + fn) else 0
        threshold_results.append((threshold, recall))

    eligible = [item for item in threshold_results if item[1] >= 0.98]
    if eligible:
        chosen_threshold = max(eligible, key=lambda item: item[0])[0]
    else:
        chosen_threshold = max(threshold_results, key=lambda item: item[1])[0]

    test_probabilities = model.predict_proba(X_test_selected)[:, attack_column]
    test_predictions = (test_probabilities >= chosen_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{title}")
    print(f"Selected threshold: {chosen_threshold:.2f}")
    print(f"Selected features: {selector.get_support().sum()} of {X.shape[1]}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[tn, fp], [fn, tp]])
    print(f"False alarms: {fp}")
    print(f"Missed MITM attacks: {fn}")
    print(f"MITM attacks detected: {tp}")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            zero_division=0,
        )
    )


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    benign = df[df["label"] == BENIGN_LABEL].copy()
    mitm = df[df["label"] == FEEDBACK_ATTACK].copy()
    known_attacks = df[
        (df["label"] != BENIGN_LABEL)
        & (df["label"] != FEEDBACK_ATTACK)
    ].copy()

    if benign.empty or mitm.empty or known_attacks.empty:
        raise ValueError("Could not find benign, MITM, and other attack records.")

    # Keep benign training and test records separate.
    benign_train, benign_test = train_test_split(
        benign, test_size=0.20, random_state=42
    )

    # Split MITM records into analyst feedback and untouched MITM test data.
    mitm_feedback, mitm_test_pool = train_test_split(
        mitm, test_size=0.20, random_state=42
    )

    # Use a 9:1 benign-to-attack ratio in both training experiments.
    attack_quota = len(benign_train) // 9
    feedback_count = max(1, round(attack_quota * FEEDBACK_FRACTION))
    known_count = attack_quota - feedback_count

    # Baseline training: no MITM examples.
    baseline_attacks = known_attacks.sample(
        n=attack_quota, random_state=42
    )
    baseline_train = pd.concat(
        [benign_train, baseline_attacks],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    # Retrained model: replace 25% of attack training records with
    # analyst-confirmed MITM examples.
    known_for_retraining = known_attacks.sample(
        n=known_count, random_state=43
    )
    feedback_examples = mitm_feedback.sample(
        n=feedback_count, random_state=44
    )
    retraining_attacks = pd.concat(
        [known_for_retraining, feedback_examples],
        ignore_index=True,
    )
    retrained_train = pd.concat(
        [benign_train, retraining_attacks],
        ignore_index=True,
    ).sample(frac=1, random_state=45).reset_index(drop=True)

    # Test both models on the same benign records and MITM records
    # that neither model has seen.
    mitm_test_count = min(
        len(mitm_test_pool),
        max(1, len(benign_test) // 9),
    )
    mitm_test = mitm_test_pool.sample(
        n=mitm_test_count, random_state=46
    )
    final_test = pd.concat(
        [benign_test, mitm_test],
        ignore_index=True,
    ).sample(frac=1, random_state=47).reset_index(drop=True)

    print(f"\nMITM feedback examples used: {feedback_count}")
    print(f"Separate MITM test examples: {mitm_test_count}")
    print("The test examples are kept out of both models' training data.")

    train_and_evaluate(
        baseline_train,
        final_test,
        "Baseline model (no MITM feedback)",
    )
    train_and_evaluate(
        retrained_train,
        final_test,
        "Retrained model (includes MITM feedback)",
    )


if __name__ == "__main__":
    main()
    