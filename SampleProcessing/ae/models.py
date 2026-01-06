from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Reshape, InputLayer, Lambda, MultiHeadAttention, Add
from keras.models import Sequential, Model


def select_first_25(x):
    return x[:, :25]


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



####Main autoencoder training ####
def build_autoencoder_data(img_shape, code_size):
    "Adding RELU activations to the AE model. Making it non-linear otherwise AE just PCA as linear encoder and decoder. MSE, Single hidden bottleneck layer AE and data is mean-centered as PCA assumes centering."
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size, activation='relu'))   

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape), activation='relu')) 
    decoder.add(Reshape(img_shape))

    return encoder, decoder