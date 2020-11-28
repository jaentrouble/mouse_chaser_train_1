import tensorflow as tf
from tensorflow import keras
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
args = parser.parse_args()

test_steps = int(args.steps)
# Check Inference speed using only cpu
tf.config.set_visible_devices([],'GPU')

image_tensor_shape = [256,384,3]

backbone_f = backbone_models.ehr_112_11
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
            _ = test_model.predict(
                np.random.random(np.concatenate([[1],image_tensor_shape])))
print('start testing')
st = time()
for _ in trange(test_steps):
    _ = test_model.predict(np.random.random(np.concatenate([[1],image_tensor_shape])))
ed = time()
print(f'Took {ed-st:.2f}seconds: average {test_steps/(ed-st):.2f}steps/seconds')