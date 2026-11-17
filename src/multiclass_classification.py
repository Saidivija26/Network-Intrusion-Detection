
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/IoT_Intrusion_cleaned.csv"


def train_multiclass_model():

    print("Loading cleaned dataset...", flush=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Separate features and original attack labels
    X = df.drop(columns=["label"])
    y = df["label"]

    print("\nNumber of attack categories:", y.nunique())
    print("\nClass distribution:")
    print(y.value_counts())

    # Encode attack categories
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining Random Forest multiclass model...", flush=True)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nModel training completed!", flush=True)

    # Predictions
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and encoder
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    joblib.dump(model, "models/multiclass_random_forest.pkl")
    joblib.dump(encoder, "models/multiclass_label_encoder.pkl")

    print("\nMulticlass model saved successfully!")


if __name__ == "__main__":
    train_multiclass_model()
