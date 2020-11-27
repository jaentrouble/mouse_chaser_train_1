import tensorflow as tf
from tensorflow import keras
from keras import layers
from hrnet import *

class EfficientHRNet_B0(layers.Layer):
    r"""EfficientHRNet_B0

    Same model as EfficientHRNet paper
    Uses EfficientNet-B0 as a backbone model
    
    Output
    ------
    Output Shape:
        (N,H/2,W/2,Wb1)
    """
    def __init__(
        self,
        filters:list,
        blocks:list,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filters : list of 4 integers
            Number of filters (Wb in paper) per branch.
            Needs to be 4 integers.
        blocks : list of 4 integers
            Number of per each stage. (Refer to the paper)
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.blocks = blocks
        