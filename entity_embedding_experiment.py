#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def callback(i, entity_embedding_nn):
    _, _, y_prob = entity_embedding_nn(X_test)
    y_prob = y_prob.detach().numpy()
    y_pred = np.round(y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"epoch {i}, roc_auc = {roc_auc:.3f}, accuracy = {accuracy}")


entity_embedding_nn = train_entity_embedding_nn(
    X, y, callback=callback, epoch=25, lr=1e-2)

rf = RandomForestClassifier(random_state=0)
eb_rf = deepcopy(rf)
lr = LogisticRegression(random_state=0)
ohe_lr = Pipeline([
    ("ohe", OneHotEncoder(sparse=False)),
    ("lr", deepcopy(lr))
])

X_embeds, _, _ = entity_embedding_nn(X)
X_embeds = [X_embed.detach().numpy() for X_embed in X_embeds]
X_embeds = np.concatenate(X_embeds, axis=1)

X_test_embeds, _, _ = entity_embedding_nn(X_test)
X_test_embeds = [X_embed.detach().numpy() for X_embed in X_test_embeds]
X_test_embeds = np.concatenate(X_test_embeds, axis=1)

rf.fit(X, y)
rf_acc = rf.score(X_test, y_test)
print(f"RF score :\t {rf_acc}")

eb_rf.fit(X_embeds, y)
eb_rf_acc = eb_rf.score(X_test_embeds, y_test)
print(f"Embeds+RF score:\t {eb_rf_acc}")

ohe_lr.fit(X, y)
ohe_lr_acc = ohe_lr.score(X_test, y_test)
print(f"OHE+LR score :\t {ohe_lr_acc}")

lr.fit(X_embeds, y)
lr_acc = eb_rf.score(X_test_embeds, y_test)
print(f"Embeds+LR score:\t {lr_acc}")
