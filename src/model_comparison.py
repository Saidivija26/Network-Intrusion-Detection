
import pandas as pd
import joblib
import os

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def train_xgboost():

    print("Loading training and testing datasets...")

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    y_train = pd.read_csv("data/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    print("\nTraining XGBoost model...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train_encoded)

    print("\nXGBoost training completed!")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test_encoded, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test_encoded,
        y_pred,
        target_names=encoder.classes_
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_encoded, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test_encoded, y_prob))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost.pkl")

    print("\nXGBoost model saved successfully!")


if __name__ == "__main__":
    train_xgboost()
