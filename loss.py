from keras import backend as K
from keras.layers import Lambda

DEBUG = False

def loss_fn(y_true, y_pred):
    product = -y_true * y_pred
    probs = 1 + K.clip(K.exp(product), 0, 1e6)
    loss = K.log(probs)
    return K.mean(loss, axis=(1,2))

def loss_exp_fn(inputs):
    y_true, y_pred = inputs
    product = -y_true * y_pred
    probs = 1 + K.clip(K.exp(product), 0, 1e6)
    loss = K.log(probs)
    mean_loss = K.mean(K.flatten(loss), axis=-1)
    return product
    
def loss_exp():
    return Lambda(loss_exp_fn)