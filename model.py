# https://github.com/shenweichen/DeepCTR-Torch

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from layer import FM, DNN, CIN
from utils import SparseFeat, DenseFeat, VarLenSparseFeat
from utils import build_input_features

class Base(tf.keras.Model):
    def __init__(self, linear_feature_columns, dnn_feature_columns, sparse_emb_dim):
        super(Base, self).__init__()

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.sparse_feature_columns = list(
                filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)
                                          ) if len(dnn_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)
                                                 ) if dnn_feature_columns else []
        self.dense_feature_columns = list(
                filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)
                                         ) if len(dnn_feature_columns) else []

        self.embedding_dict = {feat.embedding_name: layers.Embedding(feat.dimension, sparse_emb_dim, embeddings_initializer='normal')
                                for feat in self.sparse_feature_columns+self.varlen_sparse_feature_columns}

        # self.weight = tf.Variable(tf.random.normal(
        #                           [sum(fc.dimension for fc in self.dense_feature_columns), 1],
        #                           stddev=0.0001), trainable=True)
        self.out_bias= tf.Variable(tf.zeros([1,]), trainable=True)


    def input_from_feature_columns(self, x):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            x[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]])
            for feat in self.sparse_feature_columns]
        varlen_sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            x[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]])
            for feat in self.varlen_sparse_feature_columns]

        dense_value_list = [x[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        return sparse_embedding_list, \
               varlen_sparse_embedding_list, \
               dense_value_list

class DeepFM(Base):
    def __init__(self, linear_feature_columns, dnn_feature_columns,
                 sparse_emb_dim, dnn_layers, dropout_rate=0.5):
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns,
                                     sparse_emb_dim)

        self.fm = FM()
        self.dnn= tf.keras.Sequential([
                            DNN(sum(map(lambda x: x.dimension, self.dense_feature_columns)) +
                                  len(self.sparse_feature_columns) * sparse_emb_dim,
                                  dnn_layers, dropout_rate=dropout_rate),
                            layers.Dense(1, use_bias=False, activation='linear')])

    def call(self, x):
        sparse_emb, varlen_emb, dense_emb = self.input_from_feature_columns(x)

        linear_sparse_logit = tf.reduce_sum(
                                tf.concat(sparse_emb, axis=-1), axis=-1, keepdims=False)

        if len(dense_emb):
            linear_dense_logit = tf.matmul(tf.concat(
                                    dense_emb, axis=-1), self.weight)
            logit = tf.squeeze(linear_sparse_logit + linear_dense_logit, -1)

            logit += self.dnn(tf.concat([tf.squeeze(tf.concat(sparse_emb + varlen_emb, -1), 1),
                                        tf.concat(dense_emb, -1)], axis=-1))

        else:
            logit = tf.squeeze(linear_sparse_logit, -1)

            logit += tf.squeeze(self.dnn(tf.concat(
                [tf.squeeze(x, 1) for x in sparse_emb] + [tf.reshape(varlen_emb, [x.shape[0],-1])], 1)), 1)

            logit += self.fm(tf.concat(sparse_emb + varlen_emb, axis=1))

        pred = logit + self.out_bias

        return pred


class xDeepFM(Base):
    def __init__(self, linear_feature_columns, dnn_feature_columns,
                 sparse_emb_dim=8, dnn_layers=(256,256), cin_layers=(256,128,),
                 cin_split_half=True, activation=tf.nn.relu, dropout_rate=0.5):
        super(xDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns,
                                      sparse_emb_dim)

        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_layers) > 0
        if self.use_dnn:
            self.dnn = tf.keras.Sequential([
                            DNN(sum(map(lambda x: x.dimension, self.dense_feature_columns)) +
                                len(self.sparse_feature_columns) * sparse_emb_dim,
                                dnn_layers, dropout_rate=dropout_rate),
                            layers.Dense(1, use_bias=False, activation='linear')])
        self.use_cin = len(cin_layers) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            field_num = len(self.embedding_dict)
            #self.cin = tf.keras.Sequential([])
            self.cin = CIN(field_num, cin_layers, activation, cin_split_half)
            self.cin_linear = layers.Dense(1, use_bias=False, activation='linear')

    def call(self, x):
        sparse_emb, varlen_emb, dense_emb = self.input_from_feature_columns(x)

        logit = tf.reduce_sum(
                    tf.concat(sparse_emb, axis=-1), axis=-1, keepdims=False)

        if self.use_cin:
            cin_input = tf.concat(sparse_emb+varlen_emb, axis=1)
            cin_output = self.cin(cin_input)
            logit += self.cin_linear(cin_output)
        if self.use_dnn:
            dnn_input = tf.concat(
                [tf.squeeze(x, 1) for x in sparse_emb] + [tf.reshape(varlen_emb, [x.shape[0],-1])], 1)
            if dense_emb:
                dnn_input = tf.concat([dnn_input, dense_emb], -1)
            logit += self.dnn(dnn_input)

        pred = tf.squeeze(logit) + self.out_bias

        return pred
