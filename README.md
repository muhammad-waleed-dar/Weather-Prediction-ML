# Weather Prediction Using Machine Learning
 
---

## Project Overview 
This project applies Machine Learning to predict whether it will rain tomorrow (`RainTomorrow`) based on historical daily weather observations from Australia. It is being built as a multi-phase project covering the full ML lifecycle — from data preprocessing to model training, evaluation, and deployment.

**Motivation:** I previously built a Smart Weather Monitoring Station using Arduino and C++. This project extends that interest into the Machine Learning domain by using historical weather data to make predictions rather than just recording live sensor readings.

---
  
## Project Roadmap 
 
| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Dataset Selection, EDA & Data Preprocessing | ✅ Complete |
| **Phase 2** | Data Visualization & Feature Engineering | ✅ Complete |
| **Phase 3** | Model Training & Evaluation (Logistic Regression, Decision Tree, Random Forest) | ✅ Complete |
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
| 7 | Correlation matrix — feature relationships visualized via heatmap |
| 8 | Dropped `Date` column — not directly useful in raw form |
| 9 | Outlier detection & removal — IQR method on `Rainfall` |
| 10 | MinMax Normalization — numeric features scaled to [0,1] (target excluded) |
| 11 | Class imbalance analysis — No Rain: 95,420 vs Rain: 18,228 |
| 12 | Feature/Target split — X (21 features), y (`RainTomorrow`) |
| 13 | SMOTE-Tomek — balanced dataset to 95,251 samples per class |
 
**Why outlier removal before scaling?** MinMaxScaler sets its [0,1] range from each column's min/max. Removing extreme Rainfall outliers (e.g. 371mm) *before* scaling keeps those bounds reflective of everyday values rather than being stretched by rare storm events.
 
**Why SMOTE-Tomek?** The dataset was imbalanced (~84% No Rain vs ~16% Rain after cleaning). A model trained on this would be biased toward predicting "No Rain." SMOTE-Tomek oversamples the minority class with synthetic, interpolated samples while removing ambiguous borderline samples — resulting in a cleaner, balanced dataset.

---

## Phase 2: Feature Engineering & Exploratory Data Visualization ✅
 
**Notebook:** [`Phase2-Visualization/Visualization_FeatureEngineering.ipynb`](./Phase2-Visualization/Visualization_FeatureEngineering.ipynb)
 
This notebook reloads the raw dataset (rather than continuing from Phase 1's encoded output) so plots use readable labels — actual month names, `Yes`/`No`, location names — instead of Label-Encoded integers. Encoding happens later, only when data is handed to a model. Every visualization includes a title, axis labels, and a brief written interpretation, per the task requirements.
 
### Engineered Features
 
| Feature | Formula / Source | Why |
|---------|-------------------|-----|
| `Month`, `Season` | Extracted from `Date` (parsed with `format='%d/%m/%Y'`) | Weather is seasonal — Phase 1 dropped `Date` entirely, losing this signal |
| `TempRange` | `MaxTemp - MinTemp` | A large daily swing behaves differently than a stable day |
| `HumidityChange` | `Humidity3pm - Humidity9am` | Direction of humidity change through the day can signal incoming weather |
| `PressureChange` | `Pressure3pm - Pressure9am` | Falling pressure is a classic precursor to rain |
 
**Redundant feature removed:** `Date` itself is dropped after `Month`/`Season` are extracted from it — its useful signal is preserved, but the unusable raw string is not carried forward.
  
### Feature Selection
`SelectKBest` with `mutual_info_classif` ranks features by mutual information with `RainTomorrow`, complementing the linear Pearson correlation shown in the heatmap. **Bug fixed:** an earlier version ran this on numeric columns only, which silently excluded `RainToday`, `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `Month`, and `Season` — all still raw strings at that point. Given the strong persistence effect `RainToday` shows (see below), leaving it out was a real gap. All categorical columns are now Label-Encoded specifically for this step and passed in via `discrete_features`, so `mutual_info_classif` treats them correctly as categorical rather than continuous.
 
### Visualizations
 
| # | Plot | Type | Purpose |
|---|------|------|---------|
| 1 | RainTomorrow class distribution | Count plot | Confirms class imbalance |
| 2 | Rainfall distribution | Histogram + KDE | Univariate shape / right-skew check |
| 3 | Pressure3pm by RainTomorrow | Box plot | Tests "low pressure precedes rain" |
| 4 | Humidity3pm by RainTomorrow | Box plot | Tests humidity as a rain predictor |
| 5 | Rain probability by Season | Bar plot | Validates the new `Season` feature |
| 6 | Humidity3pm vs Pressure3pm | Scatter (hue) | Bivariate class separation |
| 7 | Humidity3pm by RainTomorrow | Violin plot | Full density shape, not just quartiles |
| 8 | Pressure3pm density by class | Overlaid KDE | Where the two classes diverge most |
| 9 | RainToday vs RainTomorrow | Count plot + crosstab | Tests weather "persistence" |
| 10 | Top 10 Locations by record count | Bar plot | Checks category balance across `Location` |
| 11 | Rain rate by calendar month | Line plot | Confirms a real seasonal cycle, not a random pattern |
| 12 | Pairwise feature relationships | Pair plot | Fast multi-feature visual sanity check |
| 13 | Correlation heatmap (with new features) | Heatmap | Linear relationship check |
| 14 | Feature importance via Mutual Information | Bar plot | Non-linear feature selection, required by task brief — now includes categorical features |

### Mutual Information Ranking (Top 10)

After fixing the categorical-column exclusion bug, the top 10 features by MI score are:

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | Humidity3pm | 0.1147 |
| 2 | TempRange | 0.0671 |
| 3 | Sunshine | 0.0599 |
| 4 | Cloud3pm | 0.0586 |
| 5 | Rainfall | 0.0549 |
| 6 | RainToday | 0.0421 |
| 7 | Cloud9am | 0.0421 |
| 8 | HumidityChange | 0.0385 |
| 9 | Humidity9am | 0.0379 |
| 10 | Pressure9am | 0.0297 |

This ranking complements the Pearson correlation heatmap by capturing non-linear relationships.

### Key Findings
- `Humidity3pm` is the strongest predictor of `RainTomorrow` by correlation (0.44) — this heatmap result is unaffected by the Mutual Information fix below.
- `TempRange` is the 2nd-strongest feature by correlation (-0.34) — a genuinely useful engineered feature, not a weak one as initially assumed before the notebook was actually run.
- `RainToday` shows strong persistence with `RainTomorrow` — rain rate jumps from 15.6% (no rain today) to 46.4% (rain today). It was previously missing from the Mutual Information ranking entirely; re-run Step 4 after the fix to see its actual score alongside the other features.
- `Season`/`Month` show a real seasonal cycle — Winter has the highest rain probability (26.1%), Summer the lowest (20.3%).
- `PressureChange` and `MinTemp` show the weakest correlation with the target (0.08 each) — worth checking against the updated Mutual Information ranking before deciding whether to drop either in Phase 3.
- Class counts in this notebook (110,316 No / 31,877 Yes) differ from Phase 1's (95,420 / 18,228) because this notebook works from the dataset before Rainfall outlier removal — both are correct for their respective pipeline stage.
- **Note:** the Mutual Information ranking changed after fixing the categorical-column exclusion bug — read the freshly re-run output rather than relying on any specific scores quoted earlier in this project's development.

---
 
## Phase 3: Model Training & Evaluation ✅
 
**Notebook:** [`Phase3-ModelTraining_Evaluation/ModelTraining_Evaluation.ipynb`](./Phase3-ModelTraining_Evaluation/ModelTraining_Evaluation.ipynb)
 
This notebook reloads the raw dataset and reproduces Phase 2's cleaning and engineered
features (`Month`, `Season`, `TempRange`, `HumidityChange`, `PressureChange`) plus Phase 1's
IQR outlier removal on `Rainfall`, so it runs standalone.
 
**Task sheet requirements (Classification track):**
- Models: Logistic Regression, Decision Tree, Random Forest — the boosting/XGBoost-style
  models considered earlier in this project's development were intentionally left out, since
  the official task sheet restricts non-taught techniques without prior mentor approval
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Cross-validation to assess robustness and generalization
- Model comparison and written justification for the final model choice
- Submission: GitHub repository link only, deadline **28 July 2026**

### Leakage-Free Pipeline Order
 
Unlike Phase 1/2, this phase involves real train/test evaluation, so preventing data leakage
between train and test data is a genuine requirement here (not applicable to Phase 1, which
never split data):
 
1. Load raw data, reproduce Phase 2 cleaning/feature engineering + Phase 1 outlier removal
2. Label-encode categorical columns
3. **Split train/test first** (80/20, stratified) — before any fitting
4. Scale with `MinMaxScaler`, fit on train only, transform both
5. Apply **SMOTE-Tomek to the training set only** — the test set is left at its natural
   ~84/16 imbalance so evaluation metrics reflect real-world performance
6. Train Logistic Regression, Decision Tree, and Random Forest on the resampled training set
7. Evaluate all three on the untouched test set (Accuracy, Precision, Recall, F1, Confusion Matrix)
8. `StratifiedKFold` cross-validation (5-fold) — chosen over plain K-Fold because the target
   is imbalanced, so a plain fold split risks folds with very few "Rain" rows; SMOTE-Tomek is
   refit fresh inside each fold via an `imblearn` pipeline to avoid cross-fold leakage
9. Compare all three models and select the final one

### Model Roles
 
| Model | Role |
|-------|------|
| Logistic Regression | Baseline — linear, expected to underperform on non-linear weather thresholds |
| Decision Tree | Captures non-linear interactions, prone to overfitting on a single tree |
| Random Forest | Primary/ensemble model — averages many trees, reduces overfitting vs a single tree |

### Dataset Split (this notebook's pipeline)
 
Running this notebook's cleaning + IQR outlier removal on `Rainfall` produces 113,648 rows
(95,420 No Rain / 18,228 Rain — matching Phase 1's counts), split 80/20 stratified into:
 
| Split | Rows | Rain % |
|-------|------|--------|
| Train | 90,918 | 16.0% |
| Test | 22,730 | 16.0% |
 
SMOTE-Tomek on the training set only: `{No Rain: 76,336 → 76,209, Rain: 14,582 → 76,209}`
(before/after resampling, on the 90,918-row training split).
 
### Results (Test Set, Rain Class)
 
| Model | Accuracy | Precision | Recall | F1-Score | Mean F1 (CV) |
|-------|----------|-----------|--------|----------|--------------|
| Logistic Regression | 0.7742 | 0.3916 | 0.7364 | 0.5113 | 0.5148 |
| Decision Tree | 0.7893 | 0.3813 | 0.5038 | 0.4341 | 0.4274 |
| **Random Forest** | **0.8645** | **0.5804** | 0.5595 | **0.5698** | **0.5643** |
 
### Confusion Matrices (Test Set, 22,730 rows)
 
| Model | TN (No Rain correct) | FP (false alarm) | FN (missed rain) | TP (rain caught) |
|-------|----|----|----|----|
| Logistic Regression | 14,912 | 4,172 | 961 | 2,685 |
| Decision Tree | 16,103 | 2,981 | 1,809 | 1,837 |
| Random Forest | 17,609 | 1,475 | 1,606 | 2,040 |
 
### Final Model Selection: Random Forest
 
**Random Forest** is selected as the final model. It has the best F1-Score (0.5698) and
Precision (0.5804) among the three, and its cross-validation F1 (0.5643) sits close to its
test F1 (0.5698) — a gap of ~0.006, indicating stable, non-overfit performance.
 
**Logistic Regression** shows the highest Recall (0.7364) — it catches more actual rainy
days — but at a real cost: Precision of only 0.3916 means most of its rain predictions are
false alarms (4,172 FP vs Random Forest's 1,475).
 
**Decision Tree** performs worst on every metric. Its CV F1 (0.4274) and test F1 (0.4341) are
close to each other (gap ~0.007), so it isn't overfitting — it's just a consistently weaker
model than the ensemble, likely because a single tree overfits to training-set noise in ways
that don't generalize, without the variance-averaging benefit Random Forest gets from many trees.
 
**Trade-off acknowledged:** for a weather-warning use case, missing rain (a false negative) is
arguably costlier than a false alarm. Logistic Regression misses fewer rainy days (961 vs
Random Forest's 1,606), but at the cost of vastly more false alarms (4,172 vs 1,475). Random
Forest is chosen here because it has the best overall F1 balance and the most stable
cross-validated performance, but this recall/precision trade-off is worth stating explicitly
in a viva.
 
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
├── weatherAUS.csv
├── Phase1-EDA/
│   ├── EDA.ipynb
│   └── report.pdf
├── Phase2-Visualization/
│   └── Visualization_FeatureEngineering.ipynb
├── Phase3-ModelTraining_Evaluation/
│   └── ModelTraining_Evaluation.ipynb
├── .gitignore
└── README.md
```
*Structure will grow as each phase is added — this section is updated per phase.*
 
---
 
---
 
## How to Run
```bash
git clone https://github.com/muhammad-waleed-dar/Weather-Prediction-ML.git
cd Weather-Prediction-ML
```
**Phase 1 (EDA & Preprocessing):** open `Phase1-EDA/EDA.ipynb` in VS Code or Jupyter Notebook (ensure `weatherAUS.csv` is in the same folder), then run all cells sequentially
 
**Phase 2 (Visualization & Feature Engineering):** copy `weatherAUS.csv` into `Phase2-Visualization/`, open `Visualization_FeatureEngineering.ipynb`, then run all cells sequentially. This notebook reloads and re-cleans the raw dataset independently of Phase 1's output, so it can be run standalone.
 
**Phase 3 (Model Training & Evaluation):** open `Phase3-ModelTraining_Evaluation/ModelTraining_Evaluation.ipynb` (it loads `weatherAUS.csv` via `../weatherAUS.csv` from the repo root), then run all cells sequentially. This notebook also reloads and re-cleans the raw dataset independently, so it can be run standalone.
 
---

## Challenges Encountered  
- **High missing values** — `Sunshine` (48%) and `Cloud9am`/`Cloud3pm` (38–41%) had severe gaps, handled via median/mode imputation to preserve dataset size rather than dropping rows/columns outright.
- **Severe class imbalance** — ~84% No Rain vs ~16% Rain (post-cleaning), resolved using SMOTE-Tomek.
- **Mixed data types** — 7 categorical columns required Label Encoding before any numeric operations or modeling could proceed.

