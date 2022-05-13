import pandas as pd 
import numpy as np 
from numpy import linspace

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm
from xgboost import XGBClassifier # XGBoost algorithm

from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow import keras


df_train = pd.read_csv("D:/ADSP/Hertfordshire-Constabulary/data/df_train_final.csv")
df_test = pd.read_csv("D:/ADSP/Hertfordshire-Constabulary/data/df_test_final.csv")

def x_var(df):
    df = df.iloc[:, :-1]
    return df

def y_var(df):
    df = df["outcome_type"]
    return df

x_train = x_var(df_train)
x_test = x_var(df_test)

y_train = y_var(df_train)
y_test = y_var(df_test)

col_to_scale = ["university_crime_distance", "museum_crime_distance", "studios_crime_distance"]

scaler = MinMaxScaler()
x_train[col_to_scale] = scaler.fit_transform(x_train[col_to_scale])

x_test[col_to_scale] = scaler.transform(x_test[col_to_scale])


ModelParams = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(max_depth=5),
        'params': {
            'criterion': ['gini','entropy'],
        }
    },
    "XGBClassifier": {
        "model": XGBClassifier(use_label_encoder=False, boosting='gbdt', 
                               eval_metric='logloss'),
        "params": {'n_estimators': range(6, 10),
        'max_depth': range(3, 8),
        'learning_rate': [.2, .3, .4],
        'colsample_bytree': [.7, .8, .9, 1]}
    }     
}

scores = []

for model_name, mp in ModelParams.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, 
    scoring = "f1", return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })



