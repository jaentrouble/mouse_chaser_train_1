import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import backbone_models
import specific_models
from model_trainer import ChaserModel
import numpy as np
import argparse
from tqdm import trange
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('-s','--steps', dest='steps')
parser.add_argument('-n','--name',dest='name')
parser.add_argument('-g','--gpu',dest='gpu', default=False,
                    action='store_true')
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-ea','--eager', dest='eager', 
                    action='store_true',default=False)
args = parser.parse_args()

test_steps = int(args.steps)
# Check Inference speed using only cpu
if args.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    if args.mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
else:
    tf.config.set_visible_devices([],'GPU')

image_tensor_shape = [256,384,3]

backbone_f = backbone_models.mobv3_small
specific_fs = {
    'nose' : specific_models.conv_squeeze_double
}
inputs = keras.Input(image_tensor_shape)

logdir = 'logs/speedcheck/'+args.name


test_model = ChaserModel(inputs, backbone_f, specific_fs)
# For profiling
# Graph drawing
_ = test_model.predict(np.random.random(np.concatenate([[1],image_tensor_shape])))
with tf.profiler.experimental.Profile(logdir):
    for step in range(5):
        with tf.profiler.experimental.Trace('test',step_num=step):
            if args.eager:
                _ = test_model(
                    np.random.random(np.concatenate([[1],image_tensor_shape])),
                    training=False
                )
            else:
                _ = test_model.predict(
                    np.random.random(np.concatenate([[1],image_tensor_shape]))
                )
print('start testing')
st = time()
for _ in trange(test_steps):
    if args.eager:
        _ = test_model(
            np.random.random(np.concatenate([[1],image_tensor_shape])),
            training=False
        )
    else:
        _ = test_model.predict(
            np.random.random(np.concatenate([[1],image_tensor_shape]))
        )
ed = time()
print(f'Took {ed-st:.2f}seconds: average {test_steps/(ed-st):.2f}steps/seconds')