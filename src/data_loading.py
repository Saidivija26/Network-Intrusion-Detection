
import pandas as pd
import os

# Dataset path
DATA_PATH = "data/IoT_Intrusion.csv"

def load_dataset():
    print("Loading dataset...")

    if not os.path.exists(DATA_PATH):
        print("Dataset file not found!")
        return None

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst 5 records:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    load_dataset()
