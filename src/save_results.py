
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import train_test_split


def save_binary_results():

    print("\nSaving Binary Classification Results...")

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]

    encoder = joblib.load("models/label_encoder.pkl")
    model = joblib.load("models/random_forest.pkl")

    y_true = encoder.transform(y_test)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        "results/binary_classification_report.csv"
    )

    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv("results/binary_confusion_matrix.csv")

    print("Binary results saved.")


def save_multiclass_results():

    print("\nSaving Multi-class Classification Results...")

    df = pd.read_csv(
        "data/IoT_Intrusion_cleaned.csv",
        low_memory=False
    )

    X = df.drop(columns=["label"])
    y = df["label"]

    encoder = joblib.load(
        "models/multiclass_label_encoder.pkl"
    )

    model = joblib.load(
        "models/multiclass_random_forest.pkl"
    )

    y_encoded = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        "results/multiclass_classification_report.csv"
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=range(len(encoder.classes_))
    )

    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv("results/multiclass_confusion_matrix.csv")

    print("Multi-class results saved.")


def main():

    os.makedirs("results", exist_ok=True)

    save_binary_results()
    save_multiclass_results()

    print("\nAll evaluation results saved successfully!")


if __name__ == "__main__":
    main()
