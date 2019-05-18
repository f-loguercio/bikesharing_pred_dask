#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  11 09:19:35 2019

@author: federicologuercio
"""

SEED = 1
TARGET_VARIABLE = "cnt"
SPLITS = 4
ESTIMATORS = 50
METRIC = "r2"
TIMESERIES = True

if METRIC == "r2":
    from sklearn.metrics import r2_score as metric_scorer
else:
    from sklearn.metrics import accuracy_score as metric_scorer


import numpy as np
import dask as dd
from sklearn.neighbors import LocalOutlierFactor


def types(df, types, exclude=None):
    types = df.select_dtypes(include=types)
    excluded = [TARGET_VARIABLE]
    if exclude:
        for i in exclude:
            excluded.append(i)
    cols = [col for col in types.columns if col not in excluded]
    return df[cols]


def convert_to_category(df, cols):
    for i in cols:
        df[i] = df[i].astype("category")
    return df


def drop_columns(df, cols):
    return df.drop(df[cols], axis=1)


def numericals(df, exclude=None):
    return types(df, [np.number], exclude)


def categoricals(df, exclude=None):
    return types(df, ["category", object], exclude)


def winsorize_data(df, train_df, cols):
    for col in cols:
        train_df[col] = winsorize(train_df[col], limits=[0.01, 0.01])
        df[df[col] > max(train_df[col])][col] = max(train_df[col])
        df[df[col] < min(train_df[col])][col] = min(train_df[col])
    return df


def lof(df, training_df):
    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
    y_pred = lof.fit_predict(training_df)
    outliers = np.where(y_pred == -1)
    print("Removing " + str(len(outliers[0])) + " records")
    return df.drop(outliers[0])