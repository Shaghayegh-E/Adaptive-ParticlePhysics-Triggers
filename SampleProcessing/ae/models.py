from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Reshape, InputLayer, Lambda, MultiHeadAttention, Add
from keras.models import Sequential, Model


@keras.utils.register_keras_serializable()
def select_first_25(x):
    return x[:, :25]

@keras.utils.register_keras_serializable()
def repeat_last_element(x):
    last = tf.expand_dims(x[:, -1], axis=-1)
    rep = tf.repeat(last, 8, axis=-1)
    return tf.concat([x[:, :-1], rep], axis=-1)


def build_autoencoder_flat(img_shape, code_size) -> tuple[keras.Model, keras.Model]:
    enc = Sequential([InputLayer(img_shape), Flatten(), Dense(code_size)])
    dec = Sequential([InputLayer((code_size,)), Dense(np.prod(img_shape)), Reshape(img_shape)])
    return enc, dec


def build_autoencoder_dense(img_shape, code_size) -> tuple[keras.Model, keras.Model]:
    enc = Sequential([InputLayer(img_shape), Dense(code_size)])
    dec = Sequential([InputLayer((code_size,)), Dense(np.prod(img_shape))])
    return enc, dec


def build_autoencoder_25gate(img_shape, code_size) -> tuple[keras.Model, keras.Model]:
    enc = Sequential([InputLayer(img_shape), Flatten(), Lambda(select_first_25), Dense(code_size)])
    dec = Sequential([InputLayer((code_size,)), Dense(25), Lambda(repeat_last_element), Reshape((4, 8))])
    return enc, dec


def build_attention_autoencoder_jet(img_shape, code_size, num_heads=1) -> keras.Model:
    inp = keras.Input(shape=img_shape)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=img_shape[-1])(inp, inp)
    x = Add()([inp, attn])
    z = Flatten()(x)
    z = Dense(code_size)(z)
    y = Dense(np.prod(img_shape))(z)
    out = Reshape(img_shape)(y)
    return Model(inp, out)
