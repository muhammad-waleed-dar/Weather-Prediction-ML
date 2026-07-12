# Weather Prediction Using Machine Learning

---

## Project Overview
This project applies Machine Learning techniques to predict whether it will rain tomorrow based on historical daily weather observations from Australia. This project applies Machine Learning to predict whether it will rain tomorrow (`RainTomorrow`) based on historical daily weather observations from Australia. It is being built as a multi-phase project covering the full ML lifecycle — from data preprocessing to model training, evaluation, and deployment.

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
| **Phase 6** | Model Saving / Deployment | ⏳ Upcoming |
 
*This README will be updated as each phase is completed to reflect the latest project state.*

---
 
## Dataset
| Detail | Info |
|--------|------|
| **Source** | [Kaggle — Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) |
| **File** | `weatherAUS.csv` |
| **Size** | 145,460 rows × 23 columns |
| **Target Variable** | `RainTomorrow` (Binary: 0 = No Rain, 1 = Rain) |
| **Date Range** | 2007 – 2017 (10 years, multiple Australian locations) |
| **Problem Type** | Binary Classification |
 
### Why this dataset?
`weatherAUS.csv` contains real-world weather observations with missing values, categorical features, and class imbalance — making it well-suited for practicing a complete, realistic preprocessing and modeling pipeline rather than a "clean" toy dataset.
 
### Feature Descriptions
| Column | Description |
|--------|-------------|
| Date | Date of observation |
| Location | Weather station location |
| MinTemp / MaxTemp | Minimum / Maximum temperature (°C) |
| Rainfall | Rainfall recorded (mm) |
| Evaporation | Pan evaporation (mm) |
| Sunshine | Hours of bright sunshine |
| WindGustDir / WindGustSpeed | Direction / speed of strongest wind gust |
| WindDir9am / WindDir3pm | Wind direction at 9am / 3pm |
| WindSpeed9am / WindSpeed3pm | Wind speed at 9am / 3pm (km/h) |
| Humidity9am / Humidity3pm | Humidity at 9am / 3pm (%) |
| Pressure9am / Pressure3pm | Atmospheric pressure at 9am / 3pm (hPa) |
| Cloud9am / Cloud3pm | Cloud cover at 9am / 3pm (oktas) |
| Temp9am / Temp3pm | Temperature at 9am / 3pm (°C) |
| RainToday | Did it rain today? (Yes/No) |
| RainTomorrow | **Target** — Will it rain tomorrow? (Yes/No) |
 
---
 
## Phase 1: EDA & Data Preprocessing ✅ (focuses on preparing a clean, machine-learning-ready dataset)
 
**Notebook:** [`Phase1-EDA/EDA.ipynb`](./Phase1-EDA/EDA.ipynb)
 
13-step preprocessing pipeline:
 
| Step | Description |
|------|-------------|
| 1 | Dataset loading and shape verification |
| 2 | Basic exploration: `info()`, `describe()`, `head()`, `tail()`, `nunique()` |
| 3 | Duplicate record check — 0 duplicates found |
| 4 | Null value detection — missing values in 21/23 columns |
| 5 | Null value handling — mode for categorical, median for numerical |
| 6 | Label Encoding — 7 categorical columns converted to numeric |
| 7 | MinMax Normalization — numeric features scaled to [0,1] (target excluded) |
| 8 | Correlation matrix — feature relationships visualized via heatmap |
| 9 | Dropped `Date` column — not directly useful in raw form |
| 10 | Outlier detection & removal — IQR method on `Rainfall` |
| 11 | Class imbalance analysis — No Rain: 95,420 vs Rain: 18,228 |
| 12 | Feature/Target split — X (21 features), y (`RainTomorrow`) |
| 13 | SMOTE-Tomek — balanced dataset to 95,257 samples per class |
 
**Why SMOTE-Tomek?** The dataset was imbalanced (~84% No Rain vs ~16% Rain after cleaning). A model trained on this would be biased toward predicting "No Rain." SMOTE-Tomek oversamples the minority class with synthetic, interpolated samples while removing ambiguous borderline samples — resulting in a cleaner, balanced dataset.
 
---
 
## Libraries & Tools
| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | Pandas, NumPy |
| ML & Preprocessing | Scikit-learn |
| Class Imbalance | Imbalanced-learn (SMOTE-Tomek) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook, VS Code |
| Version Control | Git & GitHub |

---
 
## Repository Structure
```
Weather-Prediction-ML/
├── Phase1-EDA/
│   ├── EDA.ipynb
│   ├── weatherAUS.csv
│   └── report.pdf
├── Phase2-ModelTraining/     (upcoming)
├── Phase3-Evaluation/        (upcoming)
├── .gitignore
└── README.md
```
*Structure will grow as each phase is added — this section is updated per phase.*

---
 
## Challenges Encountered
- **High missing values** — `Sunshine` (48%) and `Cloud9am`/`Cloud3pm` (38–41%) had severe gaps, handled via median/mode imputation to preserve dataset size rather than dropping rows/columns outright.
- **Severe class imbalance** — ~84% No Rain vs ~16% Rain (post-cleaning), resolved using SMOTE-Tomek.
- **Mixed data types** — 7 categorical columns required Label Encoding before any numeric operations or modeling could proceed.

