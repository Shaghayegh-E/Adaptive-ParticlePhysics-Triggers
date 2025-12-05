from __future__ import annotations
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable()
def masked_mse_loss(y_true, y_pred):
    valid = tf.logical_and(tf.logical_and(y_true[:, :, 0] >= 0, y_true[:, :, 1] >= 0),
                           y_true[:, :, 2] > 0)
    valid = tf.cast(tf.expand_dims(valid, -1), tf.float32)
    se = tf.square(y_true - y_pred) * valid
    denom = tf.reduce_sum(valid, axis=[1, 2]) + 1e-8
    loss = tf.reduce_sum(se, axis=[1, 2]) / denom
    return tf.math.log1p(loss)


@keras.utils.register_keras_serializable()
def mae_log1p_loss(y_true, y_pred):
    ae = tf.abs(y_true - y_pred)
    denom = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
    loss = tf.reduce_sum(ae, axis=[1, 2]) / denom
    return tf.math.log1p(loss)
