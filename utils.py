# https://github.com/shenweichen/DeepCTR-Torch

from collections import OrderedDict, namedtuple
import numpy as np
import tensorflow as tf

class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float64"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

class VarLenSparseFeat(namedtuple('VarLenFeat',
                                  ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype', 'embedding_name',
                                   'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,
                embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype,
                                                    embedding_name, embedding)


def build_input_features(feature_columns, include_varlen=True,
                         mask_zero=True, prefix='', include_fixlen=True):
    input_features = OrderedDict()
    features = OrderedDict()

    start = 0

    if include_fixlen:
        for feat in feature_columns:
            feat_name = feat.name
            if feat_name in features:
                continue
            if isinstance(feat, SparseFeat):
                features[feat_name] = (start, start + 1)
                start += 1
            elif isinstance(feat, DenseFeat):
                features[feat_name] = (start, start + feat.dimension)
                start += feat.dimension
            elif isinstance(feat,VarLenSparseFeat):
                features[feat_name] = (start, start + feat.maxlen)
                start += feat.maxlen
            else:
                raise TypeError("Invalid feature column type,got",type(feat))

    return features
