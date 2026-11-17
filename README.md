# Network Intrusion Detection Using Machine Learning

This project uses machine learning to classify network traffic as **Benign** or **Attack**. It also evaluates how well the detector handles attack families that were excluded from training, and explores whether anomaly detection and analyst feedback can improve detection of an unfamiliar family.

> **Important:** Results below are experiments on the included CICIoT2023-derived dataset. They are not a guarantee of performance on live network traffic. The scenario data is sampled and uses random row splits, so results should be interpreted as a project evaluation rather than production validation.

## Dataset

The project expects the original CSV at `data/IoT_Intrusion.csv`. Its target column is `label`; `BenignTraffic` is treated as normal traffic and the other labels are attack families. The preprocessing script removes exact duplicate rows and writes `data/IoT_Intrusion_cleaned.csv`.

The assignment scenario script creates a binary, benign-majority dataset by retaining all benign rows and sampling attack rows to target roughly **9 benign records for every attack record**. It writes separate train and test CSV files named `scenario_X_train.csv`, `scenario_X_test.csv`, `scenario_y_train.csv`, and `scenario_y_test.csv` in `data/`.

The dataset itself may not be included in a GitHub upload. Place the CSV at the path above before running the scripts. Do not commit a large dataset unless your instructor specifically requires it and the repository can accept it.

## Project workflow

1. Inspect and clean the data.
2. Create the binary assignment scenario.
3. Train a feature-selected Random Forest and choose its decision threshold using validation data.
4. Evaluate the final model on a separate test set.
5. Test a held-out attack family, compare an Isolation Forest hybrid, and simulate retraining after analyst feedback.

## Main scripts

| Script | Purpose |
|---|---|
| `src/data_loading.py` | Loads and prints information about the original dataset. |
| `src/preprocessing.py` | Removes exact duplicate rows and writes the cleaned dataset. |
| `src/create_scenario_data.py` | Creates the benign-majority binary train/test scenario files. |
| `src/model_training.py` | Selects features, trains a binary Random Forest, tunes a threshold on validation data, evaluates on the test data, and saves the model artifacts. |
| `src/unseen_attack_test.py` | Holds one attack family out of supervised training and evaluates it. The default family is `MITM-ArpSpoofing`. |
| `src/hybrid_unseen_attack_test.py` | Compares the supervised model, benign-only Isolation Forest, and their combined alerts on a held-out family. |
| `src/feedback_retraining_test.py` | Simulates adding analyst-confirmed MITM examples to training, then compares the baseline and retrained models on separate MITM examples. |
| `src/multiclass_classification.py` | Optional experiment to predict the original traffic label / attack family. |

Other scripts in `src/` support the earlier binary model comparison and evaluation workflow. The primary scenario workflow below uses the `scenario_*.csv` files.

## Setup

Run these commands from the project’s root folder, where `README.md` and `requirements.txt` are located. The commands are for Windows PowerShell.

```powershell
py -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -r requirements.txt
```

## Run the primary experiment

First make sure `data/IoT_Intrusion.csv` exists. Then run the commands from the project root in this order:

```powershell
python src/data_loading.py
python src/preprocessing.py
python src/create_scenario_data.py
python src/model_training.py
```

The model training script saves these files under `models/`:

- `random_forest.pkl`
- `label_encoder.pkl`
- `feature_selector.pkl`
- `attack_threshold.txt`

Run the generalization and feedback experiments after preprocessing:

```powershell
python src/unseen_attack_test.py
python src/hybrid_unseen_attack_test.py
python src/feedback_retraining_test.py
```

To try a different held-out attack family in `unseen_attack_test.py` or `hybrid_unseen_attack_test.py`, change the `HELD_OUT_ATTACK` value near the top of that script to a label present in the dataset, save the file, and rerun it. For example, use `DDoS-ICMP_Flood`, `Mirai-greeth_flood`, or `MITM-ArpSpoofing`. The scripts use one held-out family per run.

The optional multiclass experiment uses the cleaned dataset:

```powershell
python src/multiclass_classification.py
```

## Results from the reported runs

### Binary scenario model

The scenario contained **24,476 Benign** and **2,720 Attack** records before its 80/20 split. The feature selector retained **23 of 46 input features** in the reported run. On its 5,440-record test set, the model selected threshold **0.40** using validation data and produced:

| Measure | Result |
|---|---:|
| True negatives (Benign correctly identified) | 4,895 |
| False alarms | 1 |
| Missed attacks | 6 |
| Attacks detected | 538 |
| False positive rate | 0.0002 |
| False negative rate | 0.0110 |
| Attack ROC-AUC | 0.9986 |
| Attack PR-AUC | 0.9964 |

### Held-out-family experiments

These experiments use separate script splits; they are not the same test set as the binary scenario above.

| Experiment | Attacks detected | Attacks missed | False alarms |
|---|---:|---:|---:|
| DDoS-ICMP_Flood held out | 544 / 544 | 0 | 6 |
| Mirai-greeth_flood held out | 542 / 544 | 2 | 0 |
| MITM-ArpSpoofing held out, supervised baseline | 0 / 544 | 544 | 0 |
| MITM-ArpSpoofing held out, hybrid anomaly detector | 24 / 544 | 520 | 38 |
| MITM-ArpSpoofing after simulated analyst-feedback retraining | 454 / 544 | 90 | 33 |

The baseline’s complete failure on the held-out MITM family shows that high scores on other attack families do not guarantee generalization to every new family. In the feedback experiment, 544 MITM records were used as simulated analyst feedback, while a separate set of 544 MITM records remained outside both models’ training data. The retrained model improved detection on that separate test set, but still missed attacks and generated false alarms.

The DDoS and Mirai values came from runs where `HELD_OUT_ATTACK` was changed to those families. The committed scripts default to `MITM-ArpSpoofing`.

## Evaluation terminology

For attack detection, **Attack** is the positive class. The confusion matrix is printed in this order:

```text
[[TN, FP],
 [FN, TP]]
```

- **TN:** benign traffic correctly identified.
- **FP:** benign traffic incorrectly flagged as an attack (false alarm).
- **FN:** attack traffic missed by the detector.
- **TP:** attack traffic correctly detected.
- **False positive rate:** `FP / (FP + TN)`.
- **False negative rate:** `FN / (FN + TP)`.
- **ROC-AUC / PR-AUC:** threshold-independent ranking metrics; PR-AUC is especially useful when classes are imbalanced.

Thresholds are chosen from validation data, then the selected threshold is evaluated on test data. Results can change if the dataset, split, code, or library versions change.

## Repository structure

```text
Network-Intrusion-Detection/
├── data/              # Input dataset and generated train/test CSV files
├── figures/           # Generated visualizations
├── models/            # Saved model artifacts (created by scripts)
├── results/           # Saved result files (created by scripts)
├── src/               # Python source code
├── requirements.txt
└── README.md
```

## Limitations and future work

- The scenario deliberately samples attack records to create a benign-majority training set; this is not necessarily the real-world traffic prevalence.
- Random row splitting may put closely related traffic records in both training and test data. A time-based or device-based evaluation would give a stronger test of generalization.
- The held-out-family experiments test the selected attack family in this dataset; they do not establish detection of arbitrary future attacks.
- The feedback script is a controlled retraining simulation, not a deployed analyst-feedback or continuous-learning system.
- Future work can test group/time-based splits, additional datasets, calibrated thresholds, explainability, and live traffic integration.
