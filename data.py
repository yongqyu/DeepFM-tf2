import pandas as pd
import pickle
import tensorflow as tf

from utils import SparseFeat, DenseFeat, VarLenSparseFeat
from config import argparser
args = argparser()

with open(args.dataset_dir+'dataset_fm.pkl', 'rb') as f:
    train_set = pickle.load(f, encoding='latin1')
    test_set = pickle.load(f, encoding='latin1')
    fixlen_feature_names = pickle.load(f, encoding='latin1')
    linear_feature_columns, dnn_feature_columns = pickle.load(f)

def get_dataloader(train_batch_size, test_batch_size):
    train_target = train_set[:,0]
    train_loader = tf.data.Dataset.from_tensor_slices((train_set[:,1:], train_target))
    train_loader = train_loader.shuffle(len(train_set))
    train_loader = train_loader.batch(train_batch_size)

    test_target = test_set[:,0]
    test_loader = tf.data.Dataset.from_tensor_slices((test_set[:,1:], test_target))
    test_loader = test_loader.batch(test_batch_size)

    return train_loader, test_loader, \
           linear_feature_columns, dnn_feature_columns
