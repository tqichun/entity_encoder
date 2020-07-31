#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import lightgbm
import numpy as np
import pandas as pd
from joblib import load
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

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
X_train = X_train
X_test = X_test


def calc_balanced_sample_weight( y_train: np.ndarray):
    unique, counts = np.unique(y_train, return_counts=True)
    # This will result in an average weight of 1!
    cw = 1 / (counts / np.sum(counts)) / len(unique)

    sample_weights = np.ones(y_train.shape)

    for i, ue in enumerate(unique):
        mask = y_train == ue
        sample_weights[mask] *= cw[i]
    return sample_weights

# from lightgbm import LGBMClassifier

from wrap_lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split

# 测试多分类
X,y=load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
lgbm = LGBMClassifier(n_estimators=5000, verbose=100)
lgbm.fit(X_train, y_train, X_test, y_test)
print(lgbm.score(X_test, y_test))
# 测试回归
X,y=load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
lgbm = LGBMRegressor(n_estimators=5000, verbose=100)
lgbm.fit(X_train, y_train, X_test, y_test)
print(lgbm.score(X_test, y_test))

# 测试没有验证集
lgbm = LGBMClassifier(n_estimators=100, verbose=1)
lgbm.fit(X_train, y_train)
print(lgbm.score(X_test, y_test))

# 测试热启动
lgbm = LGBMClassifier(verbose=16)
# 0.8764 1618
# 0.8749 1557
for n_estimator in [ 128, 512, 2048, 4096]:
    lgbm.n_estimators = n_estimator
    lgbm.fit(X_train, y_train, X_test, y_test)
    acc = lgbm.score(X_test, y_test)
    print(f"n_estimator = {n_estimator}, accuracy = {acc:.4f}")

# 测试category
lgbm = LGBMClassifier(n_estimators=2000, verbose=100)
sample_weight=calc_balanced_sample_weight(y_train)
lgbm.fit(X_train, y_train, X_test, y_test, categorical_feature=cat_indexes.tolist())
print(lgbm.score(X_test, y_test))


# 测试样本权重
lgbm = LGBMClassifier(n_estimators=2000, verbose=1)
sample_weight=calc_balanced_sample_weight(y_train)
lgbm.fit(X_train, y_train, X_test, y_test, sample_weight=sample_weight)
print(lgbm.score(X_test, y_test))





param = dict(
    # n_estimators=2000,
    boosting_type="gbdt",
    objective="binary",
    learning_rate=0.01,
    max_depth=31,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=0,
    # cat_smooth=35,
    lambda_l1=0.1,
    lambda_l2=0.2,
    subsample_for_bin=40000,
    # min_data_in_leaf=4,
    min_child_weight=0.01  # min_child_weight
)
lgbm = lightgbm.train(
    param,
    lightgbm.Dataset(X_train, y_train),
    num_boost_round=2000,
    valid_sets=lightgbm.Dataset(X_test, y_test),
    early_stopping_rounds=250,
    # learning_rates=0.01,
    verbose_eval=100
)
y_prob = lgbm.predict(X_test)
y_pred = np.round(y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
print(f"roc_auc = {roc_auc:.3f}, acc = {acc:.5f}")
exit(0)
