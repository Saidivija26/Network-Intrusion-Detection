
import pandas as pd
import os

DATA_PATH = "data/IoT_Intrusion.csv"
OUTPUT_PATH = "data/IoT_Intrusion_cleaned.csv"


def preprocess_dataset():
    print("Loading original dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Original dataset shape:", df.shape)

    # Check missing values
    print("\nMissing values:", df.isnull().sum().sum())

    # Remove exact duplicate rows
    duplicates = df.duplicated().sum()
    print("\nDuplicate records found:", duplicates)

    df = df.drop_duplicates().copy()

    print("Shape after removing duplicates:", df.shape)
    print("Records removed:", duplicates)

    # Display updated class distribution
    print("\nUpdated attack category distribution:")
    print(df["label"].value_counts())

    print("\nUpdated number of categories:", df["label"].nunique())

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned dataset saved successfully!")
    print("File:", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_dataset()
