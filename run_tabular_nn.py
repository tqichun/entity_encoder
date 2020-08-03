#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
from joblib import load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from tabular_nn import train_tabular_nn

X_train, y_train, X_test, y_test, cat = load(
    "/home/tqc/PycharmProjects/automl/ASKL2.0_experiments/automl_dataset/126025.bz2")
nan_cnt = np.count_nonzero(pd.isna(pd.concat([X_train, X_test])), axis=0)
cat = np.array(cat)
cat_na_mask = (nan_cnt > 0) & cat
num_na_mask = (nan_cnt > 0) & (~cat)
cat_imputer = SimpleImputer(strategy="constant", fill_value="NA").fit(X_train.loc[:, cat_na_mask])
# num_imputer = SimpleImputer(strategy="median").fit(X_train.loc[:, num_na_mask])
X_train.loc[:, cat_na_mask] = cat_imputer.transform(X_train.loc[:, cat_na_mask])
X_test.loc[:, cat_na_mask] = cat_imputer.transform(X_test.loc[:, cat_na_mask])
# X_train.loc[:, num_na_mask] = num_imputer.transform(X_train.loc[:, num_na_mask])
# X_test.loc[:, num_na_mask] = num_imputer.transform(X_test.loc[:, num_na_mask])
ordinal_encoder = OrdinalEncoder(dtype="int").fit(X_train.loc[:, cat])
transformer = StandardScaler().fit(X_train.loc[:, ~cat])
X_train.loc[:, cat] = ordinal_encoder.transform(X_train.loc[:, cat])
X_train.loc[:, ~cat] = transformer.transform(X_train.loc[:, ~cat])
X_test.loc[:, cat] = ordinal_encoder.transform(X_test.loc[:, cat])
X_test.loc[:, ~cat] = transformer.transform(X_test.loc[:, ~cat])
cat_indexes = np.arange(len(cat))[cat]
label_encoder = LabelEncoder().fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
X_train = X_train.values
X_test = X_test.values

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
# rf = RandomForestClassifier(n_estimators=1000, random_state=0)
# rf.fit(X_train, y_train)
# print(rf.score(X_test, y_test))

# lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.01)
# lgbm.fit(X_train, y_train)
# print(lgbm.score(X_test, y_test))
# exit(0)

train_tabular_nn(
    X_train, y_train, cat_indexes,
    X_valid=X_test, y_valid=y_test,
    nn_params={
        "use_bn": False,
        "dropout_output": 0.3,
        "dropout_hidden": 0.3,
        "layers": (320, 160),
        "af_hidden": "tanh"
    },
    batch_size=1024,
    epoch=32,
    lr=0.01,
    optimizer="adam"
)
