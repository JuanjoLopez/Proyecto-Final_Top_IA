__all__ = (
    'get_training_model',
    'get_detect_model',
    'WINDOW_SHAPE',
)

import tensorflow as tf
import numpy as np


WINDOW_SHAPE = (64, 128)

### data para el reconocimiento y entrenamiento
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

###
# funciones de activacion
def softmax(a):
    exps = np.exp(a.astype(np.float64))
    return exps / np.sum(exps, axis=-1)[:, np.newaxis]

def sigmoid(a):
  return 1. / (1. + np.exp(-a))


###

def generar_pesos(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def generar_bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def convolution2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)


def max_pooling(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pooling(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def generar_capas_convolution():
    
    x = tf.placeholder(tf.float32, [None, None, None])

    # capa convolucional
    W_convolution1 = generar_pesos([5, 5, 1, 48])
    b_convolution1 = generar_bias([48])
    x_expanded = tf.expand_dims(x, 3)
    h_convolution1 = tf.nn.relu(convolution2d(x_expanded, W_convolution1) + b_convolution1)
    h_pool1 = max_pooling(h_convolution1, ksize=(2, 2), stride=(2, 2))

    #  capa convolucional
    W_convolution2 = generar_pesos([5, 5, 48, 64])
    b_convolution2 = generar_bias([64])

    h_convolution2 = tf.nn.relu(convolution2d(h_pool1, W_convolution2) + b_convolution2)
    h_pool2 = max_pooling(h_convolution2, ksize=(2, 1), stride=(2, 1))

    # capa convolucional
    W_convolution3 = generar_pesos([5, 5, 64, 128])
    b_convolution3 = generar_bias([128])

    h_convolution3 = tf.nn.relu(convolution2d(h_pool2, W_convolution3) + b_convolution3)
    h_pool3 = max_pooling(h_convolution3, ksize=(2, 2), stride=(2, 2))

    return x, h_pool3, [W_convolution1, b_convolution1,
                        W_convolution2, b_convolution2,
                        W_convolution3, b_convolution3]

def get_training_model():
    
    ## perceptron multicapa
    #####
    x_entrada, conv_capa, conv_vars = generar_capas_convolution()
    
    #####
    W_fc1 = generar_pesos([32 * 8 * 128, 2048])
    b_fc1 = generar_bias([2048])

    conv_layer_flat = tf.reshape(conv_capa, [-1, 32 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # capa de salida
    W_fc2 = generar_pesos([2048, 1 + 7 * len(CHARS)])
    b_fc2 = generar_bias([1 + 7 * len(CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2
 # x_entrada es la cap de entrada convolucional
 # y es la salida del perceptron 
 # con vars son los pesos y salidas de las capas de convolucion
 # mas los pesos de las dos capas del perceptron 
    return (x_entrada, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])


def get_detect_model():
   
    x_entrada, conv_capa, conv_vars = generar_capas_convolution()
    
    # 4ta capa 
    W_fc1 = generar_pesos([8 * 32 * 128, 2048])
    W_conv1 = tf.reshape(W_fc1, [8,  32, 128, 2048])
    b_fc1 = generar_bias([2048])
    h_conv1 = tf.nn.relu(convolution2d(conv_capa, W_conv1,
                                stride=(1, 1), padding="VALID") + b_fc1) 
    # Fifth layer
    W_fc2 = generar_pesos([2048, 1 + 7 * len(CHARS)])
    W_conv2 = tf.reshape(W_fc2, [1, 1, 2048, 1 + 7 * len(CHARS)])
    b_fc2 = generar_bias([1 + 7 * len(CHARS)])
    h_conv2 = convolution2d(h_conv1, W_conv2) + b_fc2
# x_entrada es la capa convolucional de entrada
# h conv2 es el resultado de la convolucion 
    return (x_entrada, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])

