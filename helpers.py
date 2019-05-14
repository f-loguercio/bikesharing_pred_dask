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


def convert_to_category(df, cols):
    for i in cols:
        df[i] = df[i].astype("category")
    return df


def drop_columns(df, cols):
    return df.drop(df[cols], axis=1)
