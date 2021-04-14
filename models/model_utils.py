from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, ReLU
from tensorflow.keras.models import Model
import tensorflow as tf

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9


def batch_norm(x, center=True):
    return BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=True,
        gamma_initializer='ones')(x)


def linear_layer(x, units, use_bias=True, use_bn=False):
    x = Dense(units,
              use_bias=use_bias and not use_bn,
              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=.01))(x)
    if use_bn:
        x = batch_norm(x, center=use_bias)
    return x


def projection_head(inputs, ph_layers, return_all=False):
    x = inputs
    outputs = []
    for i, layer_dim in enumerate(ph_layers):
        if i != len(ph_layers) - 1:
            # for the middle layers, use bias and relu for the output.
            dim, bias_relu = layer_dim, True
        else:
            # for the final layer, neither bias nor relu is used.
            dim, bias_relu = layer_dim, False
        x = linear_layer(x, dim, use_bias=bias_relu, use_bn=True)
        x = ReLU(name=f'proj-head-{i}')(x) if bias_relu else x
        outputs.append(x)
    if return_all:
        return outputs
    return x


def add_linear_head(encoder, num_classes, flatten=False, name=None):
    i = encoder.input
    x = encoder.output
    if flatten:
        x = Flatten()(x)
    o = Dense(num_classes, name=name)(x)
    model = Model(inputs=i, outputs=o)
    return model


def remove_layers(model, last_layer_name):
    i = model.layers[0].input
    o = model.get_layer(last_layer_name).output
    return Model(inputs=i, outputs=o)
