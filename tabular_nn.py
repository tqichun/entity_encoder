#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from itertools import chain
from logging import getLogger
from time import time
from typing import List, Union, Optional

import numpy as np
import torch
from frozendict import frozendict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.multiclass import type_of_target
from torch import nn
from torch.nn.functional import cross_entropy, mse_loss

logger = getLogger(__name__)


class TabularNN(nn.Module):
    def __init__(
            self,
            n_uniques: np.ndarray,
            vector_dim: int,
            cat_indexes: Union[List[int], np.ndarray],
            max_layer_width=2056,
            min_layer_width=32,
            dropout_hidden=0.1,
            dropout_output=0.2,
            layers=(256, 128),
            n_class=2,
            use_bn=True
    ):
        super(TabularNN, self).__init__()
        self.use_bn = use_bn
        assert len(cat_indexes) == len(n_uniques)
        self.layers = layers
        self.min_layer_width = min_layer_width
        self.max_layer_width = max_layer_width
        self.cat_indexes = np.array(cat_indexes, dtype="int")
        self.n_class = n_class
        self.dropout_output = dropout_output
        self.dropout_hidden = dropout_hidden
        self.n_uniques = n_uniques
        num_features = len(n_uniques) + vector_dim
        prop_vector_features = vector_dim / num_features
        if vector_dim > 0:
            numeric_embed_dim = int(np.clip(
                round(layers[0] * prop_vector_features * np.log10(vector_dim + 10)),
                min_layer_width, max_layer_width
            ))
            self.numeric_block = nn.Sequential(
                nn.Linear(vector_dim, numeric_embed_dim),
                nn.ReLU(inplace=True)
            )
        else:
            numeric_embed_dim = 0
            self.numeric_block = None
        if len(n_uniques) > 0:
            exp_ = np.exp(-n_uniques * 0.05)
            self.embed_dims = np.round(5 * (1 - exp_) + 1).astype("int")
            self.embedding_blocks = nn.ModuleList([
                nn.Embedding(int(n_unique), int(embed_dim))
                for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)
            ])
        else:
            self.embed_dims = np.array([])
            self.embedding_blocks = None
        after_embed_dim = int(self.embed_dims.sum() + numeric_embed_dim)
        deep_net_modules = []
        layers_ = [after_embed_dim] + list(layers)
        layers_len = len(layers_)
        for i in range(1, layers_len):
            in_features = layers_[i - 1]
            out_features = layers_[i]
            dropout_rate = self.dropout_hidden
            block = self.get_block(in_features, out_features, use_bn=self.use_bn, dropout_rate=dropout_rate)
            deep_net_modules.append(block)
        deep_net_modules.append(
            self.get_block(layers_[-1], self.n_class, self.use_bn, dropout_rate=self.dropout_output))
        self.deep_net = nn.Sequential(*deep_net_modules)
        self.wide_net = self.get_block(after_embed_dim, n_class, use_bn=self.use_bn, dropout_rate=self.dropout_output)
        output_modules = []
        if self.n_class > 1:
            output_modules.append(nn.Softmax(dim=1))
        # output_modules.append(nn.Dropout(self.dropout_output))
        self.output_layer = nn.Sequential(*output_modules)

        for m in chain(self.deep_net.modules(), self.wide_net.modules(), self.embedding_blocks.modules(),
                       self.output_layer):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                m.bias.data.zero_()

    def get_block(self, in_features, out_features, use_bn, dropout_rate):
        seq = []
        seq.append(nn.Linear(in_features, out_features))
        if use_bn:
            seq.append(nn.BatchNorm1d(out_features))
        seq.append(nn.ReLU(inplace=True))
        if dropout_rate > 0:
            seq.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*seq)

    def forward(self, X: np.ndarray):
        embeds = []
        if self.embedding_blocks is not None:
            embeds += [self.embedding_blocks[i](torch.from_numpy(X[:, col].astype("int64")))
                       for i, col in enumerate(self.cat_indexes)]
        num_indexed = np.setdiff1d(np.arange(X.shape[1]), self.cat_indexes)
        if self.numeric_block is not None:
            embeds.append(self.numeric_block(torch.from_numpy(X[:, num_indexed].astype("float32"))))
        cat_embeds = torch.cat(embeds, dim=1)
        outputs = self.deep_net(cat_embeds) + self.wide_net(cat_embeds)
        activated = self.output_layer(outputs)
        return activated


def train_tabular_nn(
        X: np.ndarray,
        y: np.ndarray,
        cat_indexes,
        X_valid: Optional[np.ndarray]=None,
        y_valid: Optional[np.ndarray]=None,
        lr=1e-2, epoch=25,
        callback=None,
        n_class=None,
        nn_params=frozendict()
) -> TabularNN:
    # fixme: tricky operate
    cat_indexes = np.array(cat_indexes, dtype="int")
    n_uniques = (X[:, cat_indexes].max(axis=0) + 1).astype("int")
    vector_dim = X.shape[1] - len(cat_indexes)
    if n_class is None:
        if type_of_target(y.astype("float")) == "continuous":
            n_class = 1
        else:
            n_class = np.unique(y).size
    nn_params = dict(nn_params)
    nn_params.update(n_class=n_class)
    tabular_nn: nn.Module = TabularNN(
        n_uniques, vector_dim, cat_indexes,
        **nn_params
    )
    tabular_nn.train(True)
    optimizer = torch.optim.Adam(tabular_nn.parameters(), lr=lr)

    start = time()
    if n_class >= 2:
        y_tensor = torch.from_numpy(y).long()
    else:
        y_tensor = torch.from_numpy(y).double()
    for i in range(epoch):
        # todo : batch validate early_stopping warm_start
        optimizer.zero_grad()
        tabular_nn.train(True)
        outputs = tabular_nn(X)
        if n_class >= 2:
            loss = cross_entropy(outputs.double(), y_tensor)
        elif n_class == 1:
            loss = mse_loss(outputs.flatten().double(), y_tensor)
        else:
            raise ValueError
        loss.backward()
        optimizer.step()
        if X_valid is not None and y_valid is not None:
            y_pred = tabular_nn(X_valid).detach().numpy()
            if n_class == 2:
                y_prob = y_pred[:, 1]
                roc_auc = roc_auc_score(y_valid, y_prob)
                y_pred = np.argmax(y_pred, axis=1)
                acc = accuracy_score(y_valid, y_pred)
                print(f"epoch = {i}, roc_auc = {roc_auc:.3f}, accuracy = {acc:.3f}")
        if callback is not None:
            callback(i, tabular_nn)
    end = time()
    logger.info(f"TabularNN training time = {end - start:.2f}s")
    tabular_nn.eval()
    return tabular_nn
