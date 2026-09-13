# 🍷 Wine Quality Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Project-Completed-success)

## 📌 Project Overview

This project uses **machine learning classification techniques to predict wine quality** from physicochemical properties of wine.

The analysis explores the relationship between chemical characteristics such as acidity, chlorides, sulphates, density, pH, and alcohol content and the resulting **wine quality rating**.

Multiple classification algorithms are trained and evaluated, followed by experiments with **class balancing, SMOTE, Random Forest hyperparameter tuning, confusion matrices, and feature importance analysis**.

---

## 🎯 Objectives

* Explore and understand the Wine Quality dataset.
* Analyze the distribution of wine quality ratings.
* Examine relationships between physicochemical features.
* Train multiple machine learning classification models.
* Evaluate model performance using classification metrics.
* Investigate the effect of class imbalance.
* Apply **SMOTE** to the training data and compare results.
* Perform **Random Forest hyperparameter tuning** using GridSearchCV.
* Analyze feature importance from the tuned Random Forest model.

---

## 🧠 Machine Learning Workflow

```text
Wine Quality Dataset
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Missing Value Check
        ↓
Data Visualization
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
┌─────────────────────────────────────┐
│        Classification Models        │
│                                     │
│  Random Forest                      │
│  SGD Classifier                     │
│  Support Vector Classifier (SVC)    │
└─────────────────────────────────────┘
        ↓
Model Evaluation
        ↓
Class Imbalance Experiment
        ↓
SMOTE Resampling
        ↓
Random Forest Hyperparameter Tuning
        ↓
Feature Importance Analysis
```

---

## 📊 Dataset

The project uses `WineQT.csv`.

The dataset contains **1,143 observations and 13 columns**.

### Features

The physicochemical attributes used for prediction include:

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol

The `quality` column is used as the **target variable**.

The `Id` column is excluded from model training.

### Target Variable

The `quality` values in the dataset range from **3 to 8**.

The dataset is imbalanced, with quality ratings **5 and 6 occurring much more frequently** than the other classes.

---

## 🔎 Data Exploration

The notebook performs:

* Dataset shape inspection
* Column inspection
* Descriptive statistical analysis
* Missing-value detection
* Quality-class distribution analysis
* Correlation analysis

### Dataset Shape

```text
(1143, 13)
```

### Missing Values

All columns contain **0 missing values**, so no missing-value imputation was required.

---

## 📈 Data Visualization

The notebook includes visualizations for:

### Wine Quality Distribution

A count plot is used to examine how frequently each quality rating occurs.

### Correlation Heatmap

A correlation heatmap is used to visualize relationships between the physicochemical variables and the wine quality dataset.

---

## 🛠️ Feature Selection

The target variable and identifier column are removed before training:

```python
X = data.drop(columns=['quality', 'Id'])
y = data['quality']
```

The data is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

* **Training data:** 80%
* **Testing data:** 20%
* **Test samples:** 229

---

# 🤖 Machine Learning Models

## 1. Random Forest Classifier

Random Forest is used as the primary tree-based classification model.

The initial model uses:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Random Forest Performance

| Metric             |      Score |
| ------------------ | ---------: |
| Accuracy           | **70.31%** |
| Weighted Precision |       0.68 |
| Weighted Recall    |       0.70 |
| Weighted F1-score  |       0.69 |

The model performs considerably better on the more frequently occurring quality classes, particularly quality **5, 6, and 7**.

The minority classes, particularly quality **4 and 8**, are difficult for the model to identify because of their limited representation in the test set.

---

## 2. Random Forest with Class Weights

To investigate the effect of class imbalance, a class-balanced Random Forest was also trained:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
```

### Result

```text
Accuracy: 69.43%
```

The class-weighted model did not outperform the standard Random Forest in terms of overall accuracy.

---

## 3. Random Forest with SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training set to increase the representation of minority classes.

```python
smote = SMOTE()

X_resampled, y_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

A Random Forest was then trained using the resampled dataset.

### Result

```text
Accuracy: 59.83%
```

In this experiment, SMOTE **reduced the overall accuracy** compared with the original Random Forest.

However, the experiment demonstrates the effect of attempting to address class imbalance rather than assuming that oversampling will automatically improve overall performance.

---

# ⚙️ Other Classification Models

## 4. Stochastic Gradient Descent Classifier

An `SGDClassifier` was trained as another classification approach.

### Performance

```text
Accuracy: 45.85%
```

The model achieved high recall for quality 5 but performed poorly for several other quality classes.

---

## 5. Support Vector Classifier

An `SVC` model was also trained.

### Performance

```text
Accuracy: 56.33%
```

The SVC performed better on some of the dominant classes but struggled with the minority quality categories.

---

# 📊 Model Comparison

| Model                         |   Accuracy |
| ----------------------------- | ---------: |
| **Random Forest**             | **70.31%** |
| Random Forest + Class Weights |     69.43% |
| Random Forest + SMOTE         |     59.83% |
| SVC                           |     56.33% |
| SGD Classifier                |     45.85% |

Based on the experiments performed in the notebook, the **standard Random Forest classifier achieved the highest overall accuracy**.

---

# 🔬 Hyperparameter Tuning

GridSearchCV was used to investigate different Random Forest configurations.

The following parameters were explored:

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
```

### Best Parameters

The grid search selected:

```text
n_estimators = 100
max_depth = None
min_samples_split = 5
```

The resulting tuned Random Forest achieved:

```text
Accuracy: 68.996%
```

In this particular train-test evaluation, the tuned model did **not** improve upon the earlier standard Random Forest result of approximately 70.31%.

---

# 🧩 Feature Importance

Feature importance analysis was performed using the trained Random Forest model.

```python
importances = best_rf_model.feature_importances_
```

The importance scores were converted into a DataFrame and visualized using a horizontal bar chart.

This provides an indication of which physicochemical properties contributed most strongly to the Random Forest's classification decisions.

---

# 📉 Model Evaluation

The project evaluates the models using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification reports
* Confusion matrices

Confusion matrices are generated for:

* Random Forest
* SGD Classifier
* Support Vector Classifier

These provide a more detailed view of which wine-quality classes are correctly or incorrectly predicted.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Seaborn**
* **Matplotlib**
* **imbalanced-learn / SMOTE**
* **Jupyter Notebook**

---

# 📁 Repository Structure

```text
OIBSIP-Intern-Saswati-Predicting-Wine-Quality-PROJECT-2-LEVEL-2/
│
├── WQP.ipynb
├── WineQT.csv
└── README.md
```

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SASWATIMATHAN/OIBSIP-Intern-Saswati-Predicting-Wine-Quality-PROJECT-2-LEVEL-2.git
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn
```

### 3. Open the notebook

```bash
jupyter notebook WQP.ipynb
```

### 4. Run the notebook cells

Make sure `WineQT.csv` is available in the appropriate working directory before running the notebook.

---

# 📌 Key Findings

* The dataset contains **1,143 wine samples** and **13 columns**.
* No missing values were present.
* Wine quality classes are **imbalanced**, with quality 5 and 6 being the dominant classes.
* **Random Forest achieved the highest overall accuracy** among the tested models.
* Class weighting produced a similar but slightly lower accuracy.
* SMOTE reduced overall accuracy in this experiment.
* SGD and SVC performed below Random Forest on the given dataset and configuration.
* Hyperparameter tuning produced a Random Forest configuration with `n_estimators=100`, `min_samples_split=5`, and `max_depth=None`.
* Feature importance was used to investigate the relative contribution of physicochemical attributes.

---

# ⚠️ Limitations

The project demonstrates a practical machine learning workflow, but several improvements could be explored:

* Stratified train-test splitting could provide more consistent class representation.
* Cross-validation could provide a more robust estimate of model performance.
* Class-specific metrics should be considered alongside accuracy because of the imbalanced target distribution.
* Feature scaling could be investigated for algorithms such as SVC and SGD.
* Additional hyperparameter optimization could be performed.
* Alternative approaches to imbalanced multi-class classification could be evaluated.
* Ordinal classification or regression could be investigated because wine quality ratings have an inherent ordering.

---

# 🚀 Future Improvements

Possible extensions include:

* Stratified cross-validation
* More extensive hyperparameter optimization
* Feature scaling and normalization
* Advanced ensemble methods
* Ordinal regression/classification approaches
* ROC-AUC and Precision-Recall analysis where appropriate
* Model serialization for future predictions
* Interactive prediction interface using Streamlit
* Deployment as a machine learning web application

---

## 👩‍💻 Author

**Saswati Anupama Mathan**

**M.Tech — Electronics & Communication Engineering (Specialisation - Communication)**

---

## 📜 License

This project is intended for educational and portfolio purposes.
