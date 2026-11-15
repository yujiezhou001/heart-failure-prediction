#!/usr/bin/env python
"""Train the final heart disease model and save it to model.bin.

The steps are the same as in notebook.ipynb (sections 3, 4.4, 5 and 14):
prepare the data, check the model with 5-fold cross-validation, train it on
the full training data, score it once on the test set, and save it.

Run it with:  uv run python train.py
"""

import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split


# parameters

data_file = 'data/heart.csv'
output_file = 'model.bin'

xgb_params = {
    'eta': 0.05,
    'max_depth': 2,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
num_boost_round = 200

# the highest threshold with at least 90% recall on validation (notebook, section 13)
threshold = 0.3

n_splits = 5


# data preparation

df = pd.read_csv(data_file)

df = df.rename(columns={
    'Age': 'age',
    'Sex': 'sex',
    'ChestPainType': 'chest_pain_type',
    'RestingBP': 'resting_bp',
    'Cholesterol': 'cholesterol',
    'FastingBS': 'fasting_bs',
    'RestingECG': 'resting_ecg',
    'MaxHR': 'max_hr',
    'ExerciseAngina': 'exercise_angina',
    'Oldpeak': 'oldpeak',
    'ST_Slope': 'st_slope',
    'HeartDisease': 'heart_disease',
})

categorical = ['sex', 'chest_pain_type', 'resting_ecg', 'exercise_angina', 'st_slope']
numerical = ['age', 'resting_bp', 'cholesterol', 'fasting_bs', 'max_hr', 'oldpeak']
features = categorical + numerical

for c in categorical:
    df[c] = df[c].str.lower()

# a cholesterol or blood pressure of 0 means "not measured"
df['cholesterol'] = df['cholesterol'].replace(0, np.nan)
df['resting_bp'] = df['resting_bp'].replace(0, np.nan)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# training

def get_fill_values(df):
    # medians from the training data only
    return {
        'cholesterol': df.cholesterol.median(),
        'resting_bp': df.resting_bp.median(),
    }


def prepare_dicts(df, fill_values):
    df = df[features].fillna(fill_values)
    return df.to_dict(orient='records')


def train(df_train, y_train):
    fill_values = get_fill_values(df_train)
    dicts = prepare_dicts(df_train, fill_values)

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    feature_names = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)

    return fill_values, dv, model


def predict(df, fill_values, dv, model):
    dicts = prepare_dicts(df, fill_values)
    X = dv.transform(dicts)

    feature_names = list(dv.get_feature_names_out())
    dX = xgb.DMatrix(X, feature_names=feature_names)
    return model.predict(dX)


# validation

print(f'doing validation with {n_splits}-fold cross-validation')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.heart_disease.values
    y_val = df_val.heart_disease.values

    fill_values, dv, model = train(df_train, y_train)
    y_pred = predict(df_val, fill_values, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc:.3f}')

print('validation results:')
print(f'auc = {np.mean(scores):.3f} +- {np.std(scores):.3f}')


# training the final model

print('training the final model')

y_full_train = df_full_train.heart_disease.values
y_test = df_test.heart_disease.values

fill_values, dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, fill_values, dv, model)

auc = roc_auc_score(y_test, y_pred)

heart_disease_pred = (y_pred >= threshold)
tp = (heart_disease_pred & (y_test == 1)).sum()
fp = (heart_disease_pred & (y_test == 0)).sum()
fn = (~heart_disease_pred & (y_test == 1)).sum()

print(f'test auc = {auc:.3f}')
print(f'at threshold {threshold}: recall = {tp / (tp + fn):.3f}, precision = {tp / (tp + fp):.3f}')


# saving the model

with open(output_file, 'wb') as f_out:
    pickle.dump((fill_values, dv, model, threshold), f_out)

print(f'the model is saved to {output_file}')
