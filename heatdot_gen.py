import matplotlib.pyplot as plt
from model_trainer import *
import tensorflow as tf
from tensorflow.keras import mixed_precision
import cv2
import numpy as np
from pathlib import Path
from tqdm import trange
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

hw_ratio = np.divide(original_hw,model_hw)

heatmap = np.zeros(original_hw, dtype=np.float)

vid_names = os.listdir(vid_dir/'vids')
vid_names.sort()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    str(vid_dir/'output'/'heatdot_vid.mp4'),
    fourcc,
    30,
    original_wh,
)

for v_i in trange(len(vid_names)):
    vid_name = vid_names[v_i]
    original_frames = []
    small_frames = []
    cap = cv2.VideoCapture(str(vid_dir/'vids'/vid_name))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            small_frame = cv2.resize(frame, dsize=model_wh)[...,2::-1]
            small_frames.append(small_frame)
            original_frames.append(frame)
        else:
            break
    batch_num = len(original_frames) // batch_size
    for i in trange(batch_num, leave=False):
        output = test_model.predict_on_batch(np.array(small_frames[i*batch_size:(i+1)*batch_size]))
        # Only nose
        nose_hms = output['nose']
        rr, cc = np.unravel_index(nose_hms.reshape((nose_hms.shape[0],-1)).argmax(axis=1),nose_hms.shape[1:])
        rr, cc = np.around(np.multiply([rr,cc],hw_ratio[:,np.newaxis])).astype(np.int)
        for j in range(batch_size):
            original_frame = original_frames[i*batch_size+j]
            heatmap[rr[j],cc[j]] += 1
            norm_hm = ((heatmap/(np.max(heatmap)))*255).astype(np.uint8)
            color_hm = cv2.applyColorMap(norm_hm, cv2.COLORMAP_JET)
            mixed = cv2.addWeighted(original_frame, 0.5, color_hm, 0.5, 0.0)
            writer.write(mixed)
    cap.release()
final_norm_hm = ((heatmap/(np.max(heatmap)))*255).astype(np.uint8)
colored_final_hm = cv2.applyColorMap(final_norm_hm, cv2.COLORMAP_JET)
cv2.imwrite(str(vid_dir/'output'/'heatdot.png'), colored_final_hm)
writer.release()


