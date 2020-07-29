#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import functools

import category_encoders.utils as util
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from entity_embedding_nn import train_entity_embedding_nn


class EntityEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cols=None, return_df=True,
            lr=1e-2,
            epoch=25,
            A=10, B=5,
            dropout1=0.1,
            dropout2=0.1,
    ):
        self.lr = lr
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.B = B
        self.A = A
        self.epoch = epoch
        self.return_df = return_df
        self.drop_cols = []
        self.cols = cols
        self._dim = None
        self.feature_names = None

    def fit(self, X, y=None, **kwargs):

        # first check the type
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)
        self.ordinal_encoder = OrdinalEncoder(dtype=np.int)
        # 1. use sklearn's OrdinalEncoder convert categories to int
        self.ordinal_encoder.fit(X[self.cols])
        X_ordinal = self.ordinal_encoder.transform(X[self.cols])
        # 2. train_entity_embedding_nn
        self.entity_embedding_nn = train_entity_embedding_nn(
            X_ordinal,
            y,
            lr=self.lr,
            epoch=self.epoch,
            nn_params={
                "A": self.A,
                "B": self.B,
                "dropout1": self.dropout1,
                "dropout2": self.dropout2,
            }
        )
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)
        index = X.index
        X.index = range(X.shape[0])
        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not self.cols:
            return X if self.return_df else X.values

        # 1. convert X to X_ordinal, and handle unknown categories
        is_known_categories = []
        for i, col in enumerate(self.cols):
            categories = self.ordinal_encoder.categories_[i]
            is_known_category = X[col].isin(categories).values
            if not np.all(is_known_category):
                X.loc[~is_known_category, col] = categories[0]
            is_known_categories.append(is_known_category)
        X_ordinal = self.ordinal_encoder.transform(X[self.cols])
        # 2. embedding by nn, and handle unknown categories by fill 0
        X_embeds, _, _ = self.entity_embedding_nn(X_ordinal)
        X_embeds = [X_embed.detach().numpy() for X_embed in X_embeds]
        for i, is_known_category in enumerate(is_known_categories):
            if not np.all(is_known_category):
                X_embeds[i][~is_known_category, :] = 0
        # 3. replace origin
        get_valid_col_name = functools.partial(self.get_valid_col_name, df=X)
        col2idx = dict(zip(self.cols, range(len(self.cols))))
        result_df_list = []
        cur_columns = []
        for column in X.columns:
            if column in self.cols:
                if len(cur_columns) > 0:
                    result_df_list.append(X[cur_columns])
                    cur_columns = []
                idx = col2idx[column]
                embed = X_embeds[idx]
                new_columns = [f"{column}_{i}" for i in range(embed.shape[1])]
                new_columns = [get_valid_col_name(new_column) for new_column in
                               new_columns]  # fixme Maybe it still exists bug
                embed = pd.DataFrame(embed, columns=new_columns)
                result_df_list.append(embed)
            else:
                cur_columns.append(column)
        if len(cur_columns) > 0:
            result_df_list.append(X[cur_columns])
            cur_columns = []
        X = pd.concat(result_df_list, axis=1)
        X.index = index
        if self.return_df:
            return X
        else:
            return X.values

    def get_valid_col_name(self, col_name, df: pd.DataFrame):
        while col_name in df.columns:
            col_name += "_"
        return col_name


from joblib import load

X_train, y_train, X_test, y_test, cat = load(
    "/home/tqc/PycharmProjects/automl/ASKL2.0_experiments/automl_dataset/126025.bz2")
nan_cnt = np.count_nonzero(pd.isna(pd.concat([X_train, X_test])), axis=0)
X_train = X_train.loc[:, nan_cnt == 0]
X_test = X_test.loc[:, nan_cnt == 0]
cat = pd.Series(cat)
cat = cat[nan_cnt == 0]
cat = pd.Series(cat)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
entity_encoder = EntityEncoder(epoch=10).fit(X_train, y_train)
entity_encoder.transform(X_test)
