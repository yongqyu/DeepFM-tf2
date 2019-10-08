# https://github.com/shenweichen/DeepCTR-Torch

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed'''

import tensorflow as tf

from utils import SparseFeat, DenseFeat, VarLenSparseFeat
from data import get_dataloader
from config import argparser
from model import DeepFM, xDeepFM

# Config
print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

args = argparser()

# Data Load
train_loader, test_loader, \
linear_feature_columns, dnn_feature_columns = \
get_dataloader(args.train_batch_size, args.test_batch_size)

# Loss, Optim
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
loss_metric = tf.keras.metrics.Sum()
auc_metric = tf.keras.metrics.AUC()

# Model
model = xDeepFM(linear_feature_columns, dnn_feature_columns,
                sparse_emb_dim=args.sparse_emb_dim,
                dnn_layers=args.dnn_layers,
                dropout_rate=args.dropout_rate)

# Board
train_summary_writer = tf.summary.create_file_writer(args.log_path)

#@tf.function
def train_one_step(x, y):
    with tf.GradientTape() as tape:
        output = model(x)
        loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=output,
                                                        labels=tf.cast(y, tf.float32)))
    gradient = tape.gradient(loss, model.trainable_variables)
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))

    loss_metric(loss)

# Train
def train(optimizer):
    best_loss= 0.
    best_auc = 0.
    start_time = time.time()
    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(train_loader, start=1):
            train_one_step(x, y)

            if step % args.print_step == 0:
                for x, y in test_loader:
                    output = model(x)
                    auc_metric(y, tf.math.sigmoid(output))

                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' %
                      (epoch, step, loss_metric.result() / args.print_step,
                                    auc_metric.result()))

                if best_auc < (auc_metric.result() / args.print_step):
                    best_loss= loss_metric.result() / args.print_step
                    best_auc = auc_metric.result() / args.print_step
                    model.save_weights(args.model_path+'cp-%d.ckpt'%epoch)
                loss_metric.reset_states()
                auc_metric.reset_states()

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', best_loss, step=epoch)
            tf.summary.scalar('test_auc', best_auc, step=epoch)

        loss_metric.reset_states()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

        print('Epoch %d DONE\tCost time: %.2f' % (epoch, time.time()-start_time))
    print('Best test_auc: ', best_auc)


# Main
if __name__ == '__main__':
    train(optimizer)
