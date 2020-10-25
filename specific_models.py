import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

def conv_squeeze(inputs, kernel_size, name=None):
    x = layers.Conv2D(
        1,
        kernel_size,
        padding='same',
        name='final_conv_'+name,
    )(inputs)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32',name=name)(x)
    return outputs