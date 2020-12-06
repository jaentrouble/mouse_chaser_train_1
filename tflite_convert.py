import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import specific_models
import backbone_models
from model_trainer import ChaserModel, create_train_dataset
import model_lr

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-q','--quantize', dest='quan',
                    action='store_true', default=False)
parser.add_argument('--load',dest='load')
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

train_dir = 'data/mixed/save'

backbone_f = getattr(backbone_models, args.model)
specific_fs={
    'head' : specific_models.conv_squeeze_double,
}
name = args.name
load_model_path = args.load
class_labels = ['head']
img_size = (224,288)
batch_size = 1

inputs = keras.Input((img_size[0],img_size[1],3))
original_model = ChaserModel(inputs, backbone_f, specific_fs)
original_model.compile(
    optimizer='adam',
    loss='mse',
)
original_model.load_weights(load_model_path)
print('loaded from : ' + load_model_path)

lr_f = model_lr.low_lr
lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

dummy_ds = create_train_dataset(
    train_dir,
    class_labels,
    img_size,
    batch_size,
    buffer_size=1000,
    val_data=True
)
# To draw graph
original_model.fit(
    x=dummy_ds,
    epochs=1,
    steps_per_epoch=3,
    callbacks=[lr_callback],
    verbose=1,
)
def representative_data_gen():
    for datum in dummy_ds.take(3000):
        yield [datum[0]]


converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
if args.quan:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()

with open(f'tflite_models/{name}.tflite','wb') as f:
    f.write(tflite_model)