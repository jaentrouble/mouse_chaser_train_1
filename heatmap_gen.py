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
import tools

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

def gaussian_heatmap_batch(rr, cc, shape, sigma=10):
    """
    Returns a heat map of a point
    Shape is expected to be (HEIGHT, WIDTH)
    [r,c] should be the point i.e. opposite of pygame notation.

    Parameters
    ----------
    rr : np.array
        batch of row of the points
    cc : np.array
        batch of the points
    shape : tuple of int
        (HEIGHT, WIDTH)

    Returns
    -------
    heatmap : np.array
        shape : (HEIGHT, WIDTH)
    """
    coordinates = np.stack(np.meshgrid(
        np.arange(shape[0],dtype=np.float32),
        np.arange(shape[1],dtype=np.float32),
        indexing='ij',
    ), axis=-1)[np.newaxis,...]
    keypoints = np.stack([rr,cc],axis=-1).reshape((-1,1,1,2))
    heatmaps = np.exp(-(np.sum((coordinates-keypoints)**2,axis=-1))/(2*sigma**2))

    return heatmaps


vid_dir = Path(args.video)
model_path = Path(args.load)
bb_model = getattr(backbone_models, args.backbone)
sp_model = {
    'head' : specific_models.conv_squeeze_double,
}
test_model = ChaserModel(tf.keras.Input((256,384,3)),bb_model,sp_model)
test_model.load_weights(str(model_path))
original_wh = (640, 480)
model_wh = (384, 256)
batch_size = int(args.batch)
original_hw = (original_wh[1],original_wh[0])
model_hw = (model_wh[1], model_wh[0])
heatmap = np.zeros(original_hw, dtype=np.float)
vid_names = os.listdir(vid_dir/'vids')
vid_names.sort()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    str(vid_dir/'output'/'heatmap_vid.mp4'),
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
    last_rc = None
    for i in trange(batch_num, leave=False):
        output = test_model.predict_on_batch(np.array(small_frames[i*batch_size:(i+1)*batch_size]))
        # Only head
        nose_hms = output['head']
        rr, cc = np.unravel_index(nose_hms.reshape((nose_hms.shape[0],-1)).argmax(axis=1),nose_hms.shape[1:])
        rr, cc, last_rc = tools.gravity(rr,cc,last_rc)
        nose_max_hms = gaussian_heatmap_batch(rr,cc, model_hw)
        for nose_max_hm, original_frame in zip(nose_max_hms, original_frames[i*batch_size:(i+1)*batch_size]):
            large_hm = cv2.resize(nose_max_hm, dsize=original_wh)
            heatmap += large_hm
            norm_hm = ((heatmap/(np.max(heatmap)))*255).astype(np.uint8)
            color_hm = cv2.applyColorMap(norm_hm, cv2.COLORMAP_JET)
            mixed = cv2.addWeighted(original_frame, 0.5, color_hm, 0.5, 0.0)
            writer.write(mixed)
    cap.release()
final_norm_hm = ((heatmap/(np.max(heatmap)))*255).astype(np.uint8)
colored_final_hm = cv2.applyColorMap(final_norm_hm, cv2.COLORMAP_JET)
cv2.imwrite(str(vid_dir/'output'/'heatmap.png'), colored_final_hm)
writer.release()


