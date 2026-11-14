# Heart Failure Prediction

Predict whether a patient has heart disease from 11 routine clinical measurements, and serve the model as a web service.

> Educational project. Not a medical tool and not medical advice.

## Problem description

### Background

Heart disease is the leading cause of death worldwide. When a patient has chest pain or other warning signs, the standard way to confirm coronary artery disease is a coronary angiography: a thin tube is guided into the heart's arteries, and a dye shows on X-ray whether any of them are narrowed. It is accurate, but it is invasive, expensive and needs a specialist team, so it can't be offered to everyone.

### The problem today

Deciding who needs an angiography is a judgment call. Before that step, doctors already have cheaper results:

- a basic check-up: age, sex, blood pressure, cholesterol, blood sugar and a resting ECG
- an exercise stress test: the patient's highest heart rate, whether exercise brings on chest pain, and how the ECG changes

But weighing 11 results together is hard, and two doctors can read the same results differently. Every patient in this dataset was sent for an angiography (that is where the label comes from), and **45% of them turned out not to have heart disease**. Those patients went through an invasive, expensive procedure they didn't need, while patients who really were sick waited in the same queue.

### The goal

Use the results that are already collected to estimate each patient's risk **before** the angiography decision. The model answers one question: given these 11 measurements, what is the probability that this patient has heart disease? Here, "heart disease" means the angiography found at least one major artery narrowed by more than 50%.

### How the model helps

The model learns from 918 past patients whose angiography results are known. It finds which combinations of measurements usually go with heart disease. For a new patient, it turns the 11 separate results into one consistent risk score that a clinic can act on:

- **Prioritize:** high-risk patients move to the front of the angiography queue.
- **Support referrals:** a general doctor without a cardiologist on site gets a second opinion on whether to refer the patient.
- **Avoid unnecessary procedures:** for low-risk patients, the doctor may decide that non-invasive follow-up is enough.
- **Consistency:** every patient is assessed the same way.

### How it is used

1. The patient has a check-up and an exercise stress test.
2. The clinic's software sends the 11 results to the prediction web service (`POST /predict`).
3. The service returns the probability of heart disease and a yes/no flag. The flag's threshold is set so that few sick patients are missed (see below).
4. The doctor combines this with everything else they know about the patient and decides what to do next.

### What the model is not

It doesn't diagnose, and it doesn't replace an angiography or a cardiologist. It was trained on 918 patients from a few hospitals in the US, Hungary and Switzerland, in data donated in 1988. A real clinic would need to retrain it and check it on its own patients before using it.

### Type of problem

Binary classification. The target is `HeartDisease` (1 = heart disease, 0 = normal). 55% of the 918 patients have heart disease, so the two classes are fairly balanced.

### How success is measured

- **ROC AUC** to compare models, since it doesn't depend on any one decision threshold.
- **Recall** for heart disease when choosing the threshold. Missing a sick patient is much worse than sending a healthy one for an extra test. The original Statlog study counted a missed case as 5 times as costly as a false alarm.
- **Baseline to beat:** saying "heart disease" for every patient is right 55% of the time, with a ROC AUC of 0.5.

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (fedesoriano, Kaggle, 2021; Open Database License). It combines five public heart disease datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland, Hungary, Switzerland, Long Beach VA and Statlog) and removes the duplicates, which leaves 918 patients.

The CSV is committed at [`data/heart.csv`](data/heart.csv). To download it again (no Kaggle login needed):

```bash
mkdir -p data
curl -L -o data/heart.zip https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction
unzip data/heart.zip -d data && rm data/heart.zip
```

| Column | Meaning | Values |
|---|---|---|
| `Age` | Age of the patient | years |
| `Sex` | Sex of the patient | `M`, `F` |
| `ChestPainType` | Type of chest pain | `TA` typical angina, `ATA` atypical angina, `NAP` non-anginal pain, `ASY` no symptoms |
| `RestingBP` | Blood pressure at rest | mm Hg |
| `Cholesterol` | Serum cholesterol | mg/dl |
| `FastingBS` | Fasting blood sugar above 120 mg/dl | `1` yes, `0` no |
| `RestingECG` | Electrocardiogram at rest | `Normal`, `ST` (ST-T wave abnormality), `LVH` (left ventricular hypertrophy) |
| `MaxHR` | Highest heart rate reached during an exercise test | beats per minute |
| `ExerciseAngina` | Chest pain brought on by exercise | `Y`, `N` |
| `Oldpeak` | ST depression during exercise compared with rest | number |
| `ST_Slope` | Slope of the ST segment at peak exercise | `Up`, `Flat`, `Down` |
| **`HeartDisease`** | **Target:** heart disease diagnosed | `1` yes, `0` no |

In the notebook, the column names are converted to lowercase with underscores (for example `ChestPainType` → `chest_pain_type`), and the text values to lowercase.

## EDA summary

The full analysis is in [`notebook.ipynb`](notebook.ipynb), sections 4 (EDA) and 6 (feature importance).

- **Target:** 508 patients (55%) have heart disease and 410 (45%) don't, so the classes are fairly balanced.
- **Hidden missing values:** the CSV has no empty cells, but 172 patients (19%) have a cholesterol of 0 and 1 patient has a resting blood pressure of 0, which is impossible. We mark them as missing. After the train/validation/test split, we fill them in all three sets with a value calculated from the training data only.
- **Missing cholesterol shows the hospital, not the patient:** 88% of patients with missing cholesterol have heart disease, compared with 48% of the others. All 172 zeros come from the two hospitals that didn't record cholesterol (Switzerland and Long Beach VA), and those hospitals had many more heart disease patients. A model could pick up this shortcut, which wouldn't help a real clinic.
- **Numerical features:** compared with healthy patients, patients with heart disease are older (56 vs 51 years), reach a lower maximum heart rate during the exercise test (128 vs 148) and have a bigger ECG change during exercise (`oldpeak` 1.3 vs 0.4). Resting blood pressure and cholesterol differ only a little.
- **Categorical features:** 79% of the patients are men, and more than half have no chest pain symptoms (`ASY`). Every category has at least 46 patients.

### Feature importance

Calculated on the 80% of patients used for training (`df_full_train`), so the test set stays unseen.

**Categorical features: mutual information.** How much knowing the feature tells us about heart disease. Higher means more useful.

| Feature | Mutual information |
|---|---|
| `st_slope` | 0.184 |
| `chest_pain_type` | 0.152 |
| `exercise_angina` | 0.123 |
| `sex` | 0.046 |
| `fasting_bs` | 0.031 |
| `resting_ecg` | 0.007 |

**Risk ratio:** the heart disease rate in a group divided by the overall rate (54%). Above 1 means higher risk than average. The clearest differences:

- `st_slope`: flat 81% (risk 1.49) vs up 21% (risk 0.39)
- `exercise_angina`: yes 84% (1.55) vs no 35% (0.64)
- `chest_pain_type`: no symptoms (`ASY`) 77% (1.43) vs atypical angina (`ATA`) 12% (0.22). Surprisingly, patients without chest pain symptoms have the highest rate.
- `sex`: men 62% (1.14) vs women 26% (0.47)
- `fasting_bs`: high blood sugar 76% (1.41) vs normal 48% (0.88)
- `resting_ecg` barely matters: 51% to 66% in every group.

**Numerical features: correlation** with heart disease, from −1 to 1. The further from 0, the stronger.

| Feature | Correlation | Meaning |
|---|---|---|
| `max_hr` | −0.40 | lower maximum heart rate → more heart disease |
| `oldpeak` | +0.38 | bigger ECG change during exercise → more heart disease |
| `age` | +0.28 | older → more heart disease |
| `fasting_bs` | +0.24 | high blood sugar → more heart disease |
| `resting_bp` | +0.12 | weak |
| `cholesterol` | +0.08 | weak (measured values only) |

**Missing cholesterol:** a 0/1 flag for "cholesterol was missing" has a correlation of **0.35**. That's much stronger than cholesterol itself and close to the top features. It's the hospital shortcut described above, and we take it into account when we decide how to fill in the missing values.

**Takeaway:** the exercise stress test results (`st_slope`, `exercise_angina`, `oldpeak`, `max_hr`) and the chest pain type carry most of the signal. The resting ECG, resting blood pressure and cholesterol carry little. We keep all 11 features and let the models decide how much to use each one.

## Modeling

Details are in [`notebook.ipynb`](notebook.ipynb), sections 7 to 14. The data is split 60% / 20% / 20% into train (550 patients), validation (184) and test (184), with `random_state=1`.

### Filling in the missing values

We compared three ways to fill the missing cholesterol and blood pressure values, using logistic regression on the validation set. The fill values are calculated from the training data only.

| How the missing values are filled | Validation ROC AUC |
|---|---|
| Leave them as 0 | 0.909 |
| Training median | 0.903 |
| Training median + a "cholesterol was missing" flag | 0.909 |

The hospital shortcut (the zeros, or the flag) adds only 0.006 AUC. That's smaller than the normal variation between cross-validation folds (about ±0.03, see below). **We use the training median.** It costs almost nothing, and the model doesn't rely on a shortcut that wouldn't exist in a real clinic, where cholesterol is always measured.

### Logistic regression

| Model (validation set) | ROC AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Baseline: "heart disease" for everyone | 0.500 | 53% | 53% | 100% |
| Logistic regression, `C=1.0`, threshold 0.5 | **0.903** | 83% | 84% | 84% |

At the default threshold of 0.5, logistic regression finds 82 of the 98 patients with heart disease in the validation set and misses 16.

**Tuning `C`** (the regularization strength) with 5-fold cross-validation on the full training data (80% of patients):

| `C` | 0.001 | 0.01 | 0.1 | 0.5 | 1 | 5 | 10 |
|---|---|---|---|---|---|---|---|
| Mean ROC AUC | 0.787 | 0.886 | 0.913 | 0.914 | 0.913 | 0.913 | 0.913 |

Any `C` from 0.1 to 10 gives practically the same result, with a fold-to-fold variation of about ±0.03. We keep `C=1.0`.

### Tree-based models

Each model was tuned on the validation set, with the same features and the same missing-value fill as logistic regression.

| Model | Parameters tuned | Best values | Validation ROC AUC |
|---|---|---|---|
| Decision tree | `max_depth`, `min_samples_leaf` | 4, 20 | 0.871 |
| Random forest | `n_estimators`, `max_depth`, `min_samples_leaf` | 160, 10, 1 | 0.910 |
| XGBoost | `eta`, `max_depth`, `min_child_weight`, number of rounds | 0.05, 2, 1, 200 | 0.906 |

- **Decision tree:** a tree with no depth limit memorizes the training data and drops to 0.736 on validation. Limiting the depth and the leaf size fixes that.
- **XGBoost:** very shallow trees (`max_depth=2`) work best. With only 550 training patients, deeper trees overfit. For `eta`, 0.05 and 0.1 were tied: a smaller `eta` only needs more rounds (in 5-fold CV, `eta=0.05` with 200 rounds and `eta=0.1` with 100 rounds both score 0.919). We kept 0.05 because its validation curve stays flat near the peak, so the exact number of rounds matters less. Larger values (0.3 and 1.0) peak after a few rounds and then get worse.

### Choosing the final model

The validation set has only 184 patients, so we also compared the tuned models with 5-fold cross-validation on the full training data:

| Model | Validation ROC AUC | 5-fold CV ROC AUC |
|---|---|---|
| Logistic regression | 0.903 | 0.913 ± 0.027 |
| Decision tree | 0.871 | 0.895 ± 0.030 |
| Random forest | **0.910** | 0.918 ± 0.028 |
| XGBoost | 0.906 | **0.919 ± 0.022** |

**We chose XGBoost:** it has the highest and most stable cross-validation score. The differences between logistic regression, random forest and XGBoost are small (within the ±0.03 noise), so logistic regression would also be a reasonable, simpler choice.

### Choosing the decision threshold

The model gives a probability. The threshold turns it into a yes/no flag. Since missing a sick patient is the costly mistake, we chose **the highest threshold that still finds at least 90% of the patients with heart disease** on the validation set: **0.3**.

| Threshold (validation set) | Precision | Recall | Sick patients missed | False alarms |
|---|---|---|---|---|
| 0.5 (default) | 85% | 85% | 15 of 98 | 15 |
| **0.3** | 76% | **92%** | **8 of 98** | 28 |

At 0.3 the model misses about half as many sick patients, in exchange for more false alarms. That's the trade-off described in the problem description.

### Final result on the test set

XGBoost trained on the full training data (734 patients) and scored once on the test set (184 patients):

| Metric | Result |
|---|---|
| ROC AUC | **0.956** |
| Recall at threshold 0.3 | **96%** (106 of 110 patients with heart disease found, 4 missed) |
| Precision at threshold 0.3 | 88% (14 false alarms among 120 flagged patients) |

The test AUC is higher than the cross-validation estimate (0.919). With only 184 patients, a test set can be easier than average by chance, so **about 0.92 is the more realistic expectation for new patients**.

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs the right Python version for you.

```bash
git clone https://github.com/yujiezhou001/heart-failure-prediction.git
cd heart-failure-prediction
uv sync
```

`uv sync` creates a `.venv` folder with the exact library versions from `uv.lock`.

To use the environment, either activate it:

```bash
source .venv/bin/activate
```

or put `uv run` in front of each command (for example `uv run python train.py`), which uses `.venv` without activating it.

To open the notebook, choose the `.venv` kernel in VS Code, or run `uv run --with jupyter jupyter lab`.

## Project structure

```
.
├── data/
│   └── heart.csv       # the dataset
├── notebook.ipynb      # data preparation, EDA and model selection
├── pyproject.toml      # project dependencies
├── uv.lock             # exact versions of every library
└── README.md
```
