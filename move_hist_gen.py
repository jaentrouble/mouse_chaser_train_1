import matplotlib.pyplot as plt
from model_trainer import *
import tensorflow as tf
from tensorflow.keras import mixed_precision
import cv2
import numpy as np
from pathlib import Path
from tqdm import trange, tqdm
import backbone_models
import specific_models
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-bb','--backbone', dest='backbone')
parser.add_argument('-v','--video',dest='video')
parser.add_argument('-b','--batch',dest='batch')
parser.add_argument('--load',dest='load')
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

vid_dir = Path(args.video)
model_path = Path(args.load)
bb_model = getattr(backbone_models, args.backbone)
sp_model = {
    'nose' : specific_models.conv3_16,
    'tail' : specific_models.conv3_16,
}
test_model = ChaserModel(tf.keras.Input((240,320,3)),bb_model,sp_model)
test_model.load_weights(str(model_path))
original_wh = (640, 480)
model_wh = (320, 240)
batch_size = int(args.batch)
original_hw = (original_wh[1],original_wh[0])
model_hw = (model_wh[1], model_wh[0])

vid_names = os.listdir(vid_dir/'vids')
vid_names.sort()

vid_names = os.listdir(vid_dir/'vids')
vid_names.sort()
mv_per_hr = np.zeros(24)

for v_i in trange(len(vid_names)):
    vid_name = vid_names[v_i]
    hour = int(vid_name.split('_')[2])
    # We don't need original frames; only relative distance is needed
    small_frames = []
    cap = cv2.VideoCapture(str(vid_dir/'vids'/vid_name))
    t = tqdm(unit='frames',leave=False)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            small_frame = cv2.resize(frame, dsize=model_wh)[...,2::-1]
            small_frames.append(small_frame)
            t.update()
        else:
            break
    t.close()
    batch_num = len(small_frames) // batch_size
    poses = np.empty((0,2))
    for i in trange(batch_num, leave=False):
        output = test_model.predict_on_batch(np.array(small_frames[i*batch_size:(i+1)*batch_size]))
        # Only nose
        nose_hms = output['nose']
        nose_poses = np.unravel_index(nose_hms.reshape((nose_hms.shape[0],-1)).argmax(axis=1),nose_hms.shape[1:])
        poses = np.append(poses,np.swapaxes(nose_poses,0,1),axis=0)
    l2_dist = np.sum(np.sqrt(np.sum((poses[1:]-poses[:-1])**2,axis=1)))
    mv_per_hr[hour] += l2_dist

plt.plot(range(24), mv_per_hr)
plt.savefig(str(vid_dir/'output'/'movement.svg'))