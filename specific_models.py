import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

# Necessary parameters are inputs and name

def conv_squeeze(kernel_size, inputs, name=None):
    x = layers.Conv2D(
        1,
        kernel_size,
        padding='same',
        name='final_conv_'+name,
        dtype='float32',
    )(inputs)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32',name=name)(x)
    return outputs