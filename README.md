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
├── pyproject.toml      # project dependencies
├── uv.lock             # exact versions of every library
└── README.md
```
