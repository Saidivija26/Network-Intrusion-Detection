import pandas as pd
from sklearn.model_selection import train_test_split

# Start with the cleaned dataset already in your project.
df = pd.read_csv("data/IoT_Intrusion_cleaned.csv")

# In this dataset, BenignTraffic is the normal traffic label.
benign = df[df["label"] == "BenignTraffic"]
attacks = df[df["label"] != "BenignTraffic"]

if benign.empty or attacks.empty:
    raise ValueError("Could not find both BenignTraffic and attack records.")

# Keep about 9 benign records for each attack record.
attack_count = min(len(attacks), round(len(benign) / 9))
attack_sample = attacks.sample(n=attack_count, random_state=42)

# Combine the records and create binary labels.
scenario = pd.concat([benign, attack_sample], ignore_index=True)
scenario["binary_label"] = scenario["label"].apply(
    lambda label: "Benign" if label == "BenignTraffic" else "Attack"
)
scenario = scenario.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and test sets.
X = scenario.drop(columns=["label", "binary_label"])
y = scenario["binary_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Use new filenames so your existing data files are preserved.
X_train.to_csv("data/scenario_X_train.csv", index=False)
X_test.to_csv("data/scenario_X_test.csv", index=False)
y_train.to_frame(name="binary_label").to_csv(
    "data/scenario_y_train.csv", index=False
)
y_test.to_frame(name="binary_label").to_csv(
    "data/scenario_y_test.csv", index=False
)

print("Scenario class counts:")
print(scenario["binary_label"].value_counts())
print("\nNew scenario train/test files saved in the data folder.")