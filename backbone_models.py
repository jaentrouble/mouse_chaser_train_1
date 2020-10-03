import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

# Get inputs and return outputs
# Output 1000 dim vector (classification)

def hr_5_3_8(inputs):
    x = [inputs]
    x = clayers.HighResolutionModule(
        filters=[8],
        blocks=[3],
        name='HR_0'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16],
        blocks=[3,3],
        name='HR_1'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32],
        blocks=[3,3,3],
        name='HR_2'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_3'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_4'
    )(x)
    x_8_to_16 = clayers.BasicBlock(
        filters=16,
        stride=2,
        name='Downsample_0'
    )(x[0])
    x_16 = layers.Add()([x_8_to_16, x[1]])
    x_16 = layers.BatchNormalization(momentum=clayers.BN_MOMENTUM)(x_16)
    x_16 = layers.ReLU()(x_16)
    x_16_to_32 = clayers.BasicBlock(
        filters=32,
        stride=2,
        name='Downsample_1'
    )(x_16)
    x_32 = layers.Add()([x_16_to_32, x[2]])
    x_32 = layers.BatchNormalization(momentum=clayers.BN_MOMENTUM)(x_32)
    x_32 = layers.ReLU()(x_32)
    x_32_to_64 = clayers.BasicBlock(
        filters=64,
        stride=2,
        name='Downsample_2'
    )(x_32)
    x_64 = layers.Add()([x_32_to_64,x[3]])
    x_64 = layers.BatchNormalization(momentum=clayers.BN_MOMENTUM)(x_64)
    x_64 = layers.ReLU()(x_64)
    x = layers.Conv2D(
        256,
        1,
        padding='same',
    )(x_64)
    x = layers.BatchNormalization(momentum=clayers.BN_MOMENTUM)(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(1000)(x)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs
