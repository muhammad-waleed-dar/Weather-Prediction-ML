# Weather Prediction Using Machine Learning

---

## Project Overview
This project applies Machine Learning techniques to predict whether it will rain tomorrow based on historical daily weather observations from Australia. This repository contains **Phase 1: Dataset Selection & Data Preprocessing**, which focuses on preparing a clean, machine-learning-ready dataset. This project applies Machine Learning to predict whether it will rain tomorrow (`RainTomorrow`) based on historical daily weather observations from Australia. It is being built as a multi-phase project covering the full ML lifecycle — from data preprocessing to model training, evaluation, and deployment.


**Motivation:** I previously built a Smart Weather Monitoring Station using Arduino and C++. This project extends that interest into the Machine Learning domain by using historical weather data to make predictions rather than just recording live sensor readings.

---
 
## Project Roadmap
 
| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Dataset Selection, EDA & Data Preprocessing | ✅ Complete |
| **Phase 2** | Model Training (Logistic Regression, Decision Tree, Random Forest) | ⏳ Upcoming |
| **Phase 3** | Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC) | ⏳ Upcoming |
| **Phase 4** | Hyperparameter Tuning | ⏳ Upcoming |
| **Phase 5** | Feature Importance Analysis | ⏳ Upcoming |
| **Phase 6** | Model Saving / Deployment (optional) | ⏳ Upcoming |
 
*This README will be updated as each phase is completed to reflect the latest project state.*
 
---
 

## 1. Dataset Selection
| Detail | Info |
|--------|------|
| **Source** | Kaggle — Rain in Australia Dataset |
| **Link** | https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package |
| **File** | weatherAUS.csv |
| **Justification** | Dataset contains real-world weather observations with missing values, class imbalance, and categorical features — ideal for practicing complete preprocessing pipeline |

---

## 2. Data Understanding
| Property | Value |
|----------|-------|
| **Rows** | 145,460 |
| **Columns** | 23 |
| **Target Variable** | RainTomorrow (Binary: 0 = No Rain, 1 = Rain) |
| **Feature Types** | 16 numerical (float64), 7 categorical (object) |
| **Duplicates** | 0 |
| **Missing Values** | Present in multiple columns (e.g. Sunshine: 69,835, Cloud9am: 55,888) |
| **Class Distribution** | No Rain: ~78%, Rain: ~22% (imbalanced) |

### Feature Descriptions
| Column | Description |
|--------|-------------|
| Date | Date of observation |
| Location | Weather station location |
| MinTemp | Minimum temperature (°C) |
| MaxTemp | Maximum temperature (°C) |
| Rainfall | Rainfall recorded (mm) |
| Evaporation | Pan evaporation (mm) |
| Sunshine | Hours of bright sunshine |
| WindGustDir | Direction of strongest wind gust |
| WindGustSpeed | Speed of strongest wind gust (km/h) |
| WindDir9am | Wind direction at 9am |
| WindDir3pm | Wind direction at 3pm |
| WindSpeed9am | Wind speed at 9am (km/h) |
| WindSpeed3pm | Wind speed at 3pm (km/h) |
| Humidity9am | Humidity at 9am (%) |
| Humidity3pm | Humidity at 3pm (%) |
| Pressure9am | Atmospheric pressure at 9am (hPa) |
| Pressure3pm | Atmospheric pressure at 3pm (hPa) |
| Cloud9am | Cloud cover at 9am (oktas) |
| Cloud3pm | Cloud cover at 3pm (oktas) |
| Temp9am | Temperature at 9am (°C) |
| Temp3pm | Temperature at 3pm (°C) |
| RainToday | Did it rain today? (Yes/No) |
| RainTomorrow | **TARGET** — Will it rain tomorrow? (Yes/No) |

---

## 3. Data Preprocessing (13 Steps)

| Step | Description |
|------|-------------|
| 1 | Dataset loading and shape verification |
| 2 | Basic exploration: info(), describe(), columns, dtypes, head(), tail(), nunique() |
| 3 | Duplicate record check — 0 duplicates found, no removal needed |
| 4 | Null value detection — missing values found in 20 out of 23 columns |
| 5 | Null value handling — mode for categorical columns, median for numerical columns |
| 6 | Label Encoding — converted 7 categorical columns to numeric |
| 7 | MinMax Normalization — scaled all numerical features to [0,1] range (target excluded) |
| 8 | Correlation matrix — visualized feature relationships using heatmap |
| 9 | Dropped 'Date' column — not relevant for ML model training |
| 10 | Outlier detection and removal — IQR method applied on Rainfall column (before/after boxplots) |
| 11 | Class imbalance analysis — No Rain: 95,420 vs Rain: 18,228 |
| 12 | Feature/Target split — X (21 features) and y (RainTomorrow) |
| 13 | SMOTE-Tomek applied — balanced classes to 95,257 each |

### Justification for SMOTE-Tomek
The dataset was heavily imbalanced (~78% No Rain vs ~22% Rain). Training a model on imbalanced data would result in biased predictions favoring the majority class. SMOTE-Tomek was applied to oversample the minority class (Rain) while cleaning borderline samples, resulting in a balanced dataset of 95,257 samples per class.

---

## 4. Libraries & Tools
| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | Pandas, NumPy |
| Preprocessing & ML | Scikit-learn |
| Class Imbalance | Imbalanced-learn (SMOTE-Tomek) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook, VS Code |
| Version Control | Git & GitHub |

---

## 5. Repository Structure

```
Weather-Prediction-ML/
├── Phase1-EDA/
│   ├── EDA.ipynb
│   ├── weatherAUS.csv
│   └── report.pdf
├── Phase2-/    (empty for now)
├── Phase3-/       (empty for now)
└── README.md

```

---

## 6. How to Run
1. Clone the repo:
```bash
git clone https://github.com/muhammad-waleed-dar/Weather-Prediction-ML.git
```
2. Open `EDA.ipynb` in VS Code or Jupyter Notebook
3. Ensure `weatherAUS.csv` is in the same folder
4. Run all cells sequentially

---

## 7. Challenges Encountered
- **High missing values** in Sunshine (48%) and Cloud columns (38-41%) — handled using median/mode imputation
- **Severe class imbalance** (~78% No Rain) — resolved using SMOTE-Tomek
- **Mixed data types** — categorical columns required Label Encoding before model training
