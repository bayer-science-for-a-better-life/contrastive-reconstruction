from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Conv2DTranspose,
    MaxPooling2D, Dropout, Input, concatenate, Activation, GlobalAveragePooling2D, Flatten, UpSampling2D, Layer,
    multiply, Lambda
)
from models.model_utils import projection_head
import tensorflow.keras.layers as layers
from enum import Enum, unique
import numpy as np

# batch norm parameters from official simclr repo
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9

ENCODER_OUTPUT_NAME = 'encoder_output'
CONTRASTIVE_OUTPUT = 'con_output'
RECONSTRUCTION_OUTPUT = 'reconst_output'
ATTENTION_UP_LAYER = 'attention_up'


class BaseEnum(Enum):
    @classmethod
    def values(cls):
        list(map(lambda x: x.value, cls))


@unique
class EncoderReduction(BaseEnum):
    """
    Define various methods for reducing the encoder output of shape (w, h, f) to
    """
    GA_POOLING = 'ga_pooling'
    FLATTEN = 'flatten'
    GA_ATTENTION = 'ga_attention'


@unique
class DecoderType(BaseEnum):
    UPSAMPLING = 'upsampling'
    TRANSPOSE = 'transpose'


def conrec_model(input_shape=(256, 256, 1), basemap=32, activation='sigmoid', depth=4,
                 p_dropout=None, batch_normalization=True, projection_dim=128, projection_head_layers=3,
                 skip_connections=None, encoder_reduction=EncoderReduction.GA_POOLING,
                 decoder_type=DecoderType.UPSAMPLING, sc_strength=1):
    def _pool_and_dropout(pool_size, p_dropout, inp):
        """helper fcn to easily add optional dropout"""
        if p_dropout:
            pool = MaxPooling2D(pool_size=pool_size)(inp)
            return Dropout(p_dropout)(pool)
        else:
            return MaxPooling2D(pool_size=pool_size)(inp)

    if skip_connections is None:
        skip_connections = depth - 1

    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    for layer_depth in range(depth):
        x = current_layer
        for _ in range(2):
            x = _create_convolution_block(input_layer=x, n_filters=basemap * (2 ** layer_depth),
                                          kernel=(3, 3), batch_normalization=batch_normalization,
                                          use_bias=True)
        if layer_depth < depth - 1:
            x = Layer(name='sc-' + str(layer_depth))(x)
            skip = _create_convolution_block(input_layer=x,
                                             n_filters=x.shape[-1] * sc_strength,
                                             kernel=(1, 1), batch_normalization=batch_normalization,
                                             use_bias=True)
            levels.append(skip)
            current_layer = _pool_and_dropout(pool_size=(2, 2), p_dropout=p_dropout, inp=x)
        else:
            x = Dropout(p_dropout)(x) if p_dropout else x
            current_layer = x

    reduced = reduce_encoder_output(encoder_output=current_layer, encoder_reduction=encoder_reduction)
    reduced = Layer(name=ENCODER_OUTPUT_NAME)(reduced)

    con_output = add_contrastive_output(input=reduced, projection_dim=projection_dim,
                                        projection_head_layers=projection_head_layers)

    for layer_depth in range(depth - 2, -1, -1):
        if decoder_type == DecoderType.TRANSPOSE:
            x = Conv2DTranspose(basemap * (2 ** layer_depth), (2, 2), strides=(2, 2), padding='same')(current_layer)
        elif decoder_type == DecoderType.UPSAMPLING:
            x = UpSampling2D(size=(2, 2))(current_layer)
        else:
            raise ValueError('Unknown decoder type')
        if skip_connections > layer_depth:
            x = concatenate([x, levels[layer_depth]], axis=3)
        else:
            print('No skip connection')
        for _ in range(2):
            x = _create_convolution_block(input_layer=x, n_filters=basemap * (2 ** layer_depth),
                                          kernel=(3, 3), batch_normalization=batch_normalization,
                                          use_bias=True)
        current_layer = Dropout(p_dropout)(x) if p_dropout else x

    reconstruction_out = Conv2D(input_shape[-1], (1, 1), activation=activation,
                                name=RECONSTRUCTION_OUTPUT)(current_layer)

    return Model(inputs, [reconstruction_out, con_output])


def add_contrastive_output(input, projection_head_layers: int, projection_dim: int):
    mid_dim = input.shape[-1]
    ph_layers = []
    for _ in range(projection_head_layers - 1):
        ph_layers.append(mid_dim)
    if projection_head_layers > 0:
        ph_layers.append(projection_dim)
    contrast_head = projection_head(input, ph_layers=ph_layers)
    con_output = Flatten(name=CONTRASTIVE_OUTPUT)(contrast_head)
    return con_output


def reduce_encoder_output(encoder_output, encoder_reduction):
    if encoder_reduction == EncoderReduction.GA_POOLING:
        reduced = GlobalAveragePooling2D()(encoder_output)
    elif encoder_reduction == EncoderReduction.FLATTEN:
        reduced = Flatten()(encoder_output)
    elif encoder_reduction == EncoderReduction.GA_ATTENTION:
        reduced = Layer()(attention_ga_pooling(encoder_output))
    else:
        raise ValueError()
    return reduced


def _create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3), activation=None,
                              padding='same', strides=(1, 1), use_bias=True):
    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides, use_bias=use_bias)(input_layer)
    if batch_normalization:
        axis = -1
        layer = BatchNormalization(
            axis=axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            fused=True,
            gamma_initializer="ones")(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def attention_ga_pooling(inputs, filters=None, batch_norm=True, dropout=True):
    if filters is None:
        base_shape = inputs.shape[-1]
        filters = [base_shape // 2, base_shape // 4, base_shape // 8]
    attn_layer = inputs
    if dropout:
        attn_layer = Dropout(0.5)(attn_layer)
    for f in filters:
        attn_layer = Conv2D(f, kernel_size=(1, 1), padding='same')(attn_layer)
        if batch_norm:
            attn_layer = BatchNormalization()(attn_layer)
        attn_layer = layers.ReLU()(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    up_c2_w = np.ones((1, 1, 1, inputs.shape[-1]))
    up_c2 = Conv2D(inputs.shape[-1], kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w], name=ATTENTION_UP_LAYER)
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, inputs])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap = Lambda(lambda x: x[0] / x[1])([gap_features, gap_mask])
    return gap
