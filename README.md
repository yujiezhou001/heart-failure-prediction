# Heart Failure Prediction

Predict whether a patient has heart disease from 11 routine clinical measurements, and serve the model as a web service.

> Educational project. Not a medical tool and not medical advice.

## Problem description

_Coming in the next step._

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
