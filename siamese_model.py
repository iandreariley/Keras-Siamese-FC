import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, BatchNormalization, Reshape
from keras.models import Model
from keras import backend as K
from loss import loss_exp

Z_SHAPE = (127, 127, 3)
X_SHAPE = (255, 255, 3)

def conv_layer(filters, kernel_dim, stride_len):
    return [Conv2D(filters, kernel_dim, strides=stride_len,
                  padding='valid', activation='relu', kernel_initializer='glorot_normal')]

def conv_block(filters, kernel_dim, stride_len):
    batch_norm = [BatchNormalization(axis=3)]
    return conv_layer(filters, kernel_dim, stride_len) + batch_norm

def max_pool():
    return [MaxPool2D(pool_size=3, strides=2, padding='valid')]

def alex_net_layers():
    layers = []
    layers += conv_block(48, 11, 2)
    layers += max_pool()
    layers += conv_block(128, 5, 1)
    layers += max_pool()
    layers += conv_block(48, 3, 1)
    layers += conv_block(48, 3, 1)
    layers += [Conv2D(32, 3, strides=1, padding='valid', kernel_initializer='glorot_normal')]
    return layers

def apply_layers(x, layers):
    out = x
    for layer in layers:
        out = layer(out)
    return out

def add_dimension(t):
    return tf.reshape(t, (1,) + t.shape)

def cross_correlation(inputs):
    x = inputs[0]
    x = tf.reshape(x, [1] + x.shape.as_list())
    z = inputs[1]
    z = tf.reshape(z, z.shape.as_list() + [1])
    return tf.nn.convolution(x, z, padding='VALID', strides=(1,1))

def x_corr_map(inputs):
    # Note that dtype MUST be specified, otherwise TF will assert that the input and output structures are the same,
    # which they most certainly are NOT.
    return K.reshape(tf.map_fn(cross_correlation, inputs, dtype=tf.float32, infer_shape=False), shape=(-1,17,17))
    
def x_corr_layer():
    return Lambda(x_corr_map, output_shape=(17, 17))

def make_model(x_shape, z_shape, w_loss=False):
    exemplar = Input(shape=z_shape)
    search = Input(shape=x_shape)
    label_input = Input(shape=(17,17))

    alex_net = alex_net_layers()

    exemplar_features = apply_layers(exemplar, alex_net)
    search_features = apply_layers(search, alex_net)
    score_map = x_corr_layer()([search_features, exemplar_features])
    
    outputs = [score_map]
    inputs = [search, exemplar]
    
    if w_loss:
        loss_layer = loss_exp()([label_input,score_map])
        outputs = outputs + [loss_layer]
        inputs = inputs + [label_input]
    model = Model(inputs=inputs, outputs=outputs)
    return model