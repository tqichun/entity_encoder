#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from itertools import chain
from logging import getLogger
from time import time
from typing import List, Union

import numpy as np
import torch
from frozendict import frozendict
from sklearn.utils.multiclass import type_of_target
from torch import nn
from torch.nn.functional import binary_cross_entropy, cross_entropy, mse_loss

logger = getLogger(__name__)


class TabularNN(nn.Module):
    def __init__(
            self,
            n_uniques: np.ndarray,
            vector_dim: int,
            cat_indexes: Union[List[int], np.ndarray],
            max_layer_width=2056,
            min_layer_width=32,
            dropout1=0.1,
            dropout2=0.1,
            layers=(256, 128),
            n_class=2
    ):
        super(TabularNN, self).__init__()
        assert len(cat_indexes)==len(n_uniques)
        self.layers = layers
        self.min_layer_width = min_layer_width
        self.max_layer_width = max_layer_width
        self.cat_indexes = np.array(cat_indexes, dtype="int")
        self.n_class = n_class
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.n_uniques = n_uniques
        num_features = len(n_uniques) + vector_dim
        prop_vector_features = vector_dim / num_features
        if vector_dim>0:
            numeric_embed_dim = np.clip(
                round(layers[0] * prop_vector_features * np.log10(vector_dim + 10)),
                min_layer_width, max_layer_width
            )
            self.numeric_block=nn.Sequential(
                nn.Linear(vector_dim, int(numeric_embed_dim)),
                nn.ReLU(inplace=True)
            )
        else:
            numeric_embed_dim=0
            self.numeric_block=None
        if len(n_uniques)>0:
            exp_ = np.exp(-n_uniques * 0.05)
            self.embed_dims = np.round(5 * (1 - exp_) + 1)
            self.embedding_block = nn.ModuleList([
                nn.Embedding(int(n_unique), int(embed_dim))
                for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)
            ])
        else:
            self.embed_dims=np.array([])
            self.embedding_block=None
        after_embed_dim=int(self.embed_dims.sum()+numeric_embed_dim)




        self.dense = nn.Sequential(
            layer1,
            layer2,
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
        for m in chain(self.dense.modules(), self.output.modules(), self.embedding_block.modules()):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, X: np.ndarray):
        embeds = [self.embedding_block[i](torch.from_numpy(X[:, i].astype("int64")))
                  for i in range(X.shape[1])]
        features = self.dense(torch.cat(embeds, dim=1))
        outputs = self.output(features)
        return embeds, features, outputs


def train_tabular_nn(
        X: np.ndarray,
        y: np.ndarray,
        lr=1e-2, epoch=25,
        callback=None,
        n_class=None,
        nn_params=frozendict()
) -> TabularNN:
    # fixme: tricky operate
    n_uniques = (X.max(axis=0) + 1).astype("int")
    if n_class is None:
        if type_of_target(y.astype("float")) == "continuous":
            n_class = 1
        else:
            n_class = np.unique(y).size
    nn_params = dict(nn_params)
    nn_params.update(n_class=n_class)
    entity_embedding_nn = TabularNN(
        n_uniques, **nn_params
    )
    optimizer = torch.optim.Adam(entity_embedding_nn.parameters(), lr=lr)

    start = time()
    if n_class > 2:
        y_tensor = torch.from_numpy(y).long()
    else:
        y_tensor = torch.from_numpy(y).double()
    for i in range(epoch):
        optimizer.zero_grad()
        _, _, outputs = entity_embedding_nn(X)
        if n_class == 2:
            loss = binary_cross_entropy(outputs.flatten().double(), y_tensor)
        elif n_class > 2:
            loss = cross_entropy(outputs.double(), y_tensor)
        elif n_class == 1:
            loss = mse_loss(outputs.flatten().double(), y_tensor)
        else:
            raise ValueError
        loss.backward()
        optimizer.step()
        if callback is not None:
            callback(i, entity_embedding_nn)
    end = time()
    logger.info(f"EntityEmbeddingNN training time = {end - start:.2f}s")
    return entity_embedding_nn
