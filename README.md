
# Network Intrusion Detection Using Machine Learning

## 1. Project Overview

This project presents a machine learning-based Network Intrusion Detection System (NIDS) designed to identify malicious network traffic and distinguish between benign and attack activities.

The system uses machine learning algorithms to analyze network traffic features and classify potential cyberattacks.

Two classification approaches are implemented:
- Binary Classification
- Multi-class Classification

## 2. Dataset Description

**Dataset:** CICIoT2023-derived IoT Intrusion Dataset

The dataset contains network traffic information with 47 columns, including 46 input features and one target label.

**Dataset details:**
- Records initially loaded: 1,048,575
- Features: 46
- Target column: `label`
- Attack categories: 34
- Records after duplicate removal: 983,019
- Missing values: 0

The dataset includes benign traffic and multiple attack categories such as DDoS, DoS, Mirai, Reconnaissance, Spoofing, and Injection attacks.

## 3. Project Modules

### Module 1: Data Loading
- Load the network intrusion dataset.
- Inspect dataset dimensions.
- Display column information.
- Identify missing values.

### Module 2: Data Preprocessing
- Identify duplicate records.
- Remove duplicate records.
- Analyze attack category distribution.
- Create binary labels for Attack and Benign traffic.
- Prepare training and testing datasets.

### Module 3: Binary Classification
Two machine learning algorithms are implemented:

1. Random Forest
2. XGBoost

The models classify network traffic into:
- Attack
- Benign

### Module 4: Multi-class Classification

A Random Forest classifier is trained to identify the original 34 traffic categories.

This approach identifies the specific attack type instead of only predicting whether traffic is malicious.

### Module 5: Model Evaluation

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

Feature importance analysis is also performed to identify influential network traffic features.

## 4. Technologies Used

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Imbalanced-learn
- Joblib
- Visual Studio Code

## 5. Project Structure

```text
Network_Intrusion_Detection/
│
├── data/
├── figures/
├── Models/
├── notebooks/
├── results/
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── main.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── model_evaluation.py
│   ├── multiclass_classification.py
│   └── save_results.py
│
├── requirements.txt
└── README.md
```

## 6. Experimental Results

### Binary Classification

The binary classification experiment compares Random Forest and XGBoost models.

Random Forest:
- ROC-AUC: 0.99959
- Confusion Matrix:
  - True Negatives: 191175
  - False Positives: 534
  - False Negatives: 104
  - True Positives: 4791

XGBoost:
- ROC-AUC: 0.99950
- Confusion Matrix:
  - True Negatives: 191281
  - False Positives: 428
  - False Negatives: 317
  - True Positives: 4578

### Multi-class Classification

The Random Forest multi-class model was trained using 786,415 records and tested using 196,604 records.

Evaluation results:
- Accuracy: 99% (rounded)
- Macro Precision: 0.77
- Macro Recall: 0.81
- Macro F1-score: 0.78
- Weighted F1-score: 0.99

Performance varies across attack categories, particularly for rare attack classes.

## 7. Visualizations

The project generates the following visualizations:

- Attack category distribution
- Confusion matrix comparison
- ROC curve comparison
- Random Forest feature importance

All generated figures are stored in the `figures/` directory.

## 8. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
cd Network_Intrusion_Detection
```

### Step 2: Create virtual environment

```bash
py -3.12 -m venv venv
```

### Step 3: Activate virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run data loading

```bash
python src/data_loading.py
```

### Step 6: Run preprocessing

```bash
python src/preprocessing.py
```

### Step 7: Run model training

```bash
python src/model_training.py
```

### Step 8: Run multi-class classification

```bash
python src/multiclass_classification.py
```

### Step 9: Run evaluation

```bash
python src/model_evaluation.py
```

### Step 10: Save evaluation results

```bash
python src/save_results.py
```

## 9. Conclusion

This project demonstrates the application of machine learning techniques for network intrusion detection.

The binary classification experiments compare Random Forest and XGBoost, while the multi-class classification experiment identifies different categories of network attacks.

The experimental results demonstrate promising detection performance, while also highlighting the challenges associated with imbalanced attack categories and rare network events.

## 10. Future Enhancements

- Improve detection of rare attack categories.
- Explore advanced ensemble learning techniques.
- Develop a real-time intrusion monitoring dashboard.
- Investigate explainable AI techniques.
- Evaluate the system on additional unseen network traffic datasets.
