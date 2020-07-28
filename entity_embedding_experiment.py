#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from itertools import chain

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn
from torch.nn.functional import binary_cross_entropy

class EntityEmbeddingNN(nn.Module):
    def __init__(
            self,
            n_uniques: np.ndarray,
            A=10, B=5,
            dropout1=0.1,
            dropout2=0.1,
            n_class=2
    ):
        super(EntityEmbeddingNN, self).__init__()
        self.n_class = n_class
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.n_uniques = n_uniques
        self.A = A
        self.B = B
        exp_ = np.exp(-n_uniques * 0.05)
        self.embed_dims = (5 * (1 - exp_) + 1).astype("int64")
        sum_ = np.log(self.embed_dims).sum()
        self.n_layer1 = min(1000,
                            int(A * (n_uniques.size ** 0.5) * sum_ + 1))
        self.n_layer2 = int(self.n_layer1 / B) + 2
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(n_unique), int(embed_dim))
            for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)
        ])
        self.layer1 = nn.Sequential(
            nn.Linear(self.embed_dims.sum(), self.n_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.n_layer1, self.n_layer2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout2)
        )
        # regression
        if n_class == 1:
            self.output = nn.Linear(self.n_layer2, 1)
        # binary classification
        elif n_class == 2:
            self.output = nn.Sequential(
                nn.Linear(self.n_layer2, 1),
                nn.Sigmoid()
            )
        # multi classification
        elif n_class > 2:
            self.output = nn.Sequential(
                nn.Linear(self.n_layer2, n_class),
                nn.Softmax()
            )
        else:
            raise ValueError(f"Invalid n_class : {n_class}")
        self.dense = nn.Sequential(
            self.layer1,
            self.layer2,
        )

        for m in chain(self.dense.modules(), self.output.modules(), self.embeddings.modules()):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, X: np.ndarray):
        embeds = [self.embeddings[i](torch.from_numpy(X[:, i].astype("int64")))
                  for i in range(X.shape[1])]
        features = self.dense(torch.cat(embeds, dim=1))
        outputs = self.output(features)
        return embeds, features, outputs


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from copy import deepcopy
from time import time

df = pd.read_csv("train.csv")
y = df.pop("class").values
y_tensor = torch.from_numpy(y).double()
X = df.values
test_df = pd.read_csv("train.csv")
y_test = test_df.pop("class").values
X_test = test_df.values

n_uniques = (X.max(axis=0) + 1).astype("int")
print(f"samples : {X.shape[0]}, features : {X.shape[1]}")
print(f"n_uniques(category cardinals) : {list(n_uniques)}")
entity_embedding_nn = EntityEmbeddingNN(
    n_uniques
)
# loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(entity_embedding_nn.parameters(), lr=1e-2)

start = time()

for i in range(25):
    optimizer.zero_grad()
    _, _, outputs = entity_embedding_nn(X)
    loss = binary_cross_entropy(outputs.flatten().double(), y_tensor)
    loss.backward()
    optimizer.step()
    _, _, y_prob = entity_embedding_nn(X_test)
    y_prob = y_prob.detach().numpy()
    y_pred = np.round(y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"epoch {i}, roc_auc = {roc_auc:.3f}, accuracy = {accuracy}")

end = time()
print(f"training time = {end - start:.2f}s")


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
