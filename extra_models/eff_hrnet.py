import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .hrnet import *

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
        blocks : list of 3 integers
            Number of per each stage. (Refer to the paper)
            Note: Here, it refers to Basic block number, 
                  which has 2 conv layers
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.blocks = blocks
        assert len(filters) == 4, 'Filters should be a list of 4 integers'
        assert len(blocks) == 3, 'Blocks should be a list of 3 integers'

    def build(self, input_shape):
        effnet=keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape[1:]
        )
        self.backbone = keras.Model(
            inputs = effnet.input,
            outputs=[
                effnet.get_layer('block2b_add').output,
                effnet.get_layer('block3b_add').output,
                effnet.get_layer('block5c_add').output,
                effnet.get_layer('block7a_project_bn').output,
            ]
        )
        self.branches = []
        self.branch1_layers = []
        for i,bn in enumerate(self.blocks):
            self.branch1_layers.append(
                HighResolutionBranch(
                    self.filters[0],
                    bn,
                    name=f'branch1_stage{i+1}'
                )
            )
        self.branches.append(keras.Sequential(self.branch1_layers))
        self.branch2_layers = []
        for i,bn in enumerate(self.blocks):
            self.branch2_layers.append(
                HighResolutionBranch(
                    self.filters[1],
                    bn,
                    name=f'branch2_stage{i+1}'
                )
            )
        self.branches.append(keras.Sequential(self.branch2_layers))
        self.branch3_layers = []
        for i,bn in enumerate(self.blocks[1:]):
            self.branch3_layers.append(
                HighResolutionBranch(
                    self.filters[2],
                    bn,
                    name=f'branch3_stage{i+2}'
                )
            )
        self.branches.append(keras.Sequential(self.branch3_layers))
        self.branches.append(HighResolutionBranch(
            self.filters[3],
            self.blocks[2],
            name=f'branch4_stage3'
        ))
        self.fusion_layer = HighResolutionFusion(
            [self.filters[0]],
            name='fusion_layer'
        )
        self.deconv_block = keras.Sequential([
            layers.Conv2DTranspose(
                filters=self.filters[0],
                kernel_size=2,
                strides=2
            ),
            BasicBlock(self.filters[0]),
            BasicBlock(self.filters[0]),
        ], name='deconv_block')

    def call(self, inputs):
        backbone_features = self.backbone(inputs)
        branch_outputs = [
            br(ft) for br,ft in zip(self.branches, backbone_features)
        ]
        fused_output = self.fusion_layer(branch_outputs)
        deconv_output = self.deconv_block(fused_output[0])
        return deconv_output

    def get_config(self):
        config = super().get_config()
        config['filters'] = self.filters
        config['blocks'] = self.blocks
        return config
