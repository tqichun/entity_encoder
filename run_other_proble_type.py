#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from entity_embedding_nn import train_entity_embedding_nn

df = pd.read_csv("train.csv")
y = df.pop("class").values
X = df.values
test_df = pd.read_csv("train.csv")
y_test = test_df.pop("class").values
X_test = test_df.values

n_uniques = (X.max(axis=0) + 1).astype("int")
print(f"samples : {X.shape[0]}, features : {X.shape[1]}")
print(f"n_uniques(category cardinals) : {list(n_uniques)}")


def callback_multiclass(i, entity_embedding_nn):
    _, _, y_prob = entity_embedding_nn(X_test)
    y_prob = y_prob.detach().numpy()
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"epoch {i}, acc = {acc:.3f}")


train_entity_embedding_nn(
    X, y,
    callback=callback_multiclass, epoch=50, n_class=3
)


def callback_regression(i, entity_embedding_nn):
    _, _, y_pred = entity_embedding_nn(X_test)
    y_pred = y_pred.detach().numpy()
    r2 = r2_score(y_test, y_pred)
    print(f"epoch {i}, r2 = {r2:.3f}")


train_entity_embedding_nn(
    X, y,
    callback=callback_regression, epoch=25, n_class=1
)
