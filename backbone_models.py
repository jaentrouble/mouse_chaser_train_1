import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

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
    outputs = clayers.HighResolutionFusion(
        filters=[8],
        name='Fusion_0'
    )(x)[0]
    return outputs

def ehr_112_11(inputs):
    features = clayers.EfficientHRNet_B0(
        filters=[12,22,44,86] ,
        blocks =[2,2,4],      # Model from the Paper has 2x blocks
        name='EffHRNet'
    )(inputs)
    return features