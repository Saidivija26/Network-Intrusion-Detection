
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

def feature_importance():

    print("Loading Random Forest model...")

    model = joblib.load("models/random_forest.pkl")

    # Load feature names
    X_train = pd.read_csv("data/X_train.csv", nrows=5)
    feature_names = X_train.columns

    # Extract importance
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance", ascending=False
    )

    print("\nTop 15 Important Features:")
    print(importance_df.head(15).to_string(index=False))

    # Plot top 15 features
    top_features = importance_df.head(15).sort_values(
        by="Importance"
    )

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title("Top 15 Important Features - Random Forest")
    plt.xlabel("Feature Importance")
    plt.ylabel("Network Features")
    plt.tight_layout()

    plt.savefig(
        "figures/feature_importance.png",
        dpi=300
    )

    plt.show()

    print("\nFeature importance graph saved!")


if __name__ == "__main__":
    feature_importance()
