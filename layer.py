# https://github.com/shenweichen/DeepCTR-Torch

import tensorflow as tf
import tensorflow.keras.layers as layers

class FM(tf.keras.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def call(self, input):
        square_of_sum = tf.math.pow(tf.math.reduce_sum(input, 1, keepdims=True), 2)
        sum_of_square = tf.math.reduce_sum(input * input, 1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.math.reduce_sum(cross_term, axis=2, keepdims=False)

        return tf.squeeze(cross_term, -1)


class DNN(tf.keras.layers.Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
    """

    def __init__(self, inputs_dim, hidden_units, activation=tf.nn.relu, dropout_rate=0, use_bn=False):
        super(DNN, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")

        self.linears = [layers.Dense(hidden_units[i]) for i in range(len(hidden_units))]

        if self.use_bn:
            self.bn = [nn.BatchNorm1d(hidden_units[i]) for i in range(len(hidden_units))]

    def call(self, input):
        for i in range(len(self.linears)):

            fc = self.linears[i](input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            input = fc

        return input


class CIN(tf.keras.layers.Layer):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_size, layer_size=(128, 128), activation=tf.nn.relu, split_half=True, l2_reg=1e-5):
        super(CIN, self).__init__()

        self.layer_size = layer_size
        self.activation = activation
        self.split_half = split_half
        self.conv1ds = []

        for i, size in enumerate(layer_size):
            self.conv1ds.append(layers.Conv1D(size, 1, data_format='channels_first'))

    def call(self, input):
        batch_size, field_size, emb_size = input.shape
        hidden_nn_layers = [input]
        final_result = []

        for i, size in enumerate(self.layer_size):
            x = tf.einsum('bhd,bmd->bhmd', hidden_nn_layers[i], hidden_nn_layers[0])
            x = tf.reshape(x, (batch_size, -1, emb_size))
            x = self.conv1ds[i](x)
            if self.activation:
                x = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(x, 2 * [size // 2], 1)
                else:
                    # last layer
                    direct_connect = x
                    next_hidden = 0
            else:
                direct_connect = x
                next_hidden = x

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.reduce_sum(tf.concat(final_result, axis=1), -1)

        return result
