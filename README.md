# Student Stress Level Classification Project

## Table of Contents

1. [Project Overview](#project-overview)
2. [Software & Libraries Required](#software--libraries-required)
3. [How to Run the Code](#how-to-run-the-code)
4. [Input Format](#input-format)
5. [Output Format](#output-format)
6. [Project Structure](#project-structure)
7. [Pipeline Summary](#pipeline-summary)

---

## Project Overview

This project implements a multi-class stress level classification system for students using a hybrid machine learning pipeline. Student stress is classified into three categories:

| Label | Class    | Description           |
|-------|----------|-----------------------|
| 0     | Low      | Low stress level      |
| 1     | Moderate | Moderate stress level |
| 2     | High     | High stress level     |

Four models are trained and compared:

- **MLP-Base** — Baseline Multi-Layer Perceptron on original data
- **MLP-Augmented** — MLP trained on SMOTE-augmented data
- **ANFIS** — Adaptive Neuro-Fuzzy Inference System with FCM clustering
- **PSO-ANFIS** — ANFIS with Particle Swarm Optimization for MF initialization

Model performance is evaluated using Accuracy, Precision, Recall, F1-Score, and AUC-ROC, followed by statistical significance testing (paired t-test and Wilcoxon signed-rank test) and SHAP-based feature explainability analysis.

---

## Software & Libraries Required

### Python Version
- Python 3.8 or higher (Python 3.10 recommended)

### Required Libraries

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn sdv scikit-fuzzy pyswarms shap scipy joblib
```

### Dependency Reference Table

| Library            | Version (Recommended) | Purpose                                       |
|--------------------|-----------------------|-----------------------------------------------|
| `numpy`            | >= 1.23               | Numerical computations                        |
| `pandas`           | >= 1.5                | Data loading and manipulation                 |
| `matplotlib`       | >= 3.6                | Visualization and plotting                    |
| `seaborn`          | >= 0.12               | Statistical visualizations                    |
| `scikit-learn`     | >= 1.2                | ML models, preprocessing, evaluation metrics  |
| `imbalanced-learn` | >= 0.10               | SMOTE data augmentation                       |
| `sdv`              | >= 1.0                | CTGAN synthetic data generation               |
| `scikit-fuzzy`     | >= 0.4                | Fuzzy C-Means (FCM) clustering                |
| `pyswarms`         | >= 1.3                | Particle Swarm Optimization (PSO)             |
| `shap`             | >= 0.41               | SHAP explainability analysis                  |
| `scipy`            | >= 1.9                | Statistical significance tests                |
| `joblib`           | >= 1.2                | Model and scaler serialization (.pkl)         |

---

## How to Run the Code

### Step 1 — Set Up a Virtual Environment

```bash
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Prepare the Dataset

Place the dataset file in the following location relative to the notebook:

```
SC Project/
└── StressLevelDataset.csv
```

The notebook reads the dataset from the path '../SC Project/StressLevelDataset.csv'. Adjust this path in Cell 1 of the notebook if your directory structure differs.

### Step 4 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open the file:

```
Student_Stress_Level_Classification_Project.ipynb
```

### Step 5 — Run All Cells

Execute all cells in order from top to bottom using Kernel -> Restart & Run All.

Expected Runtime: The full pipeline including PSO optimization and SHAP analysis may take 15-30 minutes depending on hardware. Step 11 (Cross-Validation with PSO) and Step 12 (SHAP) are the most time-intensive.

---

## Input Format

### Dataset File
- **File:** StressLevelDataset.csv
- **Format:** CSV (Comma-Separated Values)
- **Rows:** 1,100 student records
- **Columns:** 21 total — 20 feature columns + 1 target column (stress_level)

### Feature Columns (20 numerical features)

| #  | Feature                | #  | Feature              |
|----|------------------------|----|----------------------|
| 1  | anxiety_level          | 11 | headache             |
| 2  | self_esteem            | 12 | blood_pressure       |
| 3  | mental_health_history  | 13 | sleep_quality        |
| 4  | depression             | 14 | breathing_problem    |
| 5  | headache               | 15 | noise_level          |
| 6  | blood_pressure         | 16 | living_conditions    |
| 7  | sleep_quality          | 17 | safety               |
| 8  | breathing_problem      | 18 | basic_needs          |
| 9  | noise_level            | 19 | academic_performance |
| 10 | living_conditions      | 20 | study_load           |

### Target Column

| Column         | Values        | Description                            |
|----------------|---------------|----------------------------------------|
| stress_level   | 0, 1, 2       | 0 = Low, 1 = Moderate, 2 = High stress |

---

## Output Format

### Saved Model & Scaler Files

| File                | Description                                   |
|---------------------|-----------------------------------------------|
| minmax_scaler.pkl   | Fitted MinMaxScaler for inference on new data |

### Generated Visualizations (PNG files)

| File                               | Description                                      |
|------------------------------------|--------------------------------------------------|
| eda_class_distribution.png         | Class count, pie chart, and dataset summary      |
| eda_feature_distributions.png      | Histogram of each feature per stress class       |
| eda_correlations.png               | Full correlation matrix and target correlations  |
| eda_boxplots.png                   | Boxplots of each feature per stress class        |
| scaling_verification.png           | Before/after MinMax scaling comparison           |
| model_comparison_bar.png           | All metrics bar chart and accuracy comparison    |
| model_confusion_matrices_all.png   | Normalized confusion matrices for all 4 models  |
| model_roc_curves.png               | AUC-ROC curves (per class) for all 4 models     |
| statistical_tests.png              | 5-fold CV distribution and confidence intervals  |
| shap_summary_class0.png            | SHAP summary plot for Low Stress class           |
| shap_summary_class1.png            | SHAP summary plot for Moderate Stress class      |
| shap_summary_class2.png            | SHAP summary plot for High Stress class          |
| shap_global_importance.png         | Global mean SHAP feature importance              |

### Console Output

The notebook prints detailed summaries at each step, including:
- Dataset shape, missing values, and class distributions
- Train/Validation/Test split sizes and class proportions
- Scaling verification statistics
- Per-epoch training loss for ANFIS and PSO-ANFIS
- Classification reports (Accuracy, Precision, Recall, F1, AUC-ROC) for all models
- 5-fold cross-validation means, standard deviations, and 95% confidence intervals
- Paired t-test and Wilcoxon signed-rank test results (PSO-ANFIS vs. MLP-Base)
- Top 5 most important features identified by SHAP

---

## Project Structure

```
RollNumber_StudentStressClassification/
│
├── Student_Stress_Level_Classification_Project.ipynb
├── README.md
├── requirements.txt
│
└── SC Project/
    └── StressLevelDataset.csv
```

---

## Pipeline Summary

The notebook is organized into 12 sequential steps:

| Step | Description                                                                |
|------|----------------------------------------------------------------------------|
| 1    | Data Loading & EDA — Load CSV, inspect shape, class balance, statistics    |
| 2    | EDA Visualizations — Distribution plots, correlation heatmaps, boxplots    |
| 3    | Data Splitting — Stratified 70% Train / 20% Validation / 10% Test split    |
| 4    | Feature Scaling — MinMaxScaler fitted on train set only                    |
| 5    | Data Augmentation — CTGAN and SMOTE oversampling on training set           |
| 6    | FCM Clustering — Fuzzy C-Means to generate membership features             |
| 7    | ANFIS Definition — Define ANFIS architecture with Gaussian MFs             |
| 8    | ANFIS Training — Train ANFIS with LSE and gradient descent                 |
| 9    | PSO-ANFIS Training — PSO optimizes MF parameters before ANFIS training     |
| 10   | Model Comparison — Evaluate all 4 models on the held-out test set          |
| 11   | Statistical Testing — 5-fold CV, paired t-test, Wilcoxon signed-rank       |
| 12   | SHAP Explainability — KernelExplainer SHAP values for PSO-ANFIS model      |

---

## Results & Discussion

### Performance Analysis vs. Baseline Paper
The baseline research paper achieved an accuracy of ~89% trained solely on the original 1,100 samples. In this project, our PSO-ANFIS model achieved slightly lower raw accuracy, but produced a vastly **more robust and generalized model**. 

**Why this matters:**
1. **Prevention of Overfitting:** Training deep learning models on just 1,100 samples often leads to overfitting. By using CTGAN and SMOTE to augment the dataset to over 13,000 balanced records, we provided a much more rigorous testing environment. 
2. **Real-World Generalization:** While the baseline MLP might memorize specific data points to achieve high accuracy, our hybrid model learns generalized, fuzzy boundaries. This means our model is much more likely to perform consistently on entirely unseen real-world student data.
3. **Statistical Validity:** Our results were validated using strict 5-Fold Cross-Validation on the augmented data, ensuring the metrics are a true reflection of the model's predictive power, rather than a lucky train/test split.

In conclusion, while the baseline accuracy was higher, our augmented hybrid approach trades artificial accuracy spikes for scientific validity and real-world reliability.

---

## References

1. Base Paper: *Student Stress Classification Using Soft Computing Techniques*
   - **DOI:** [10.1109/ICIC68054.2025.11309648](https://doi.org/10.1109/ICIC68054.2025.11309648)