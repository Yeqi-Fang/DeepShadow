import os
import json
import glob as glob
import numpy as np
import datetime
import utils
import shutil
import tensorboard
import time
import cv2
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import pandas as pd
import utils


TRAIN = True
EPOCHS = 300
star = 10
BH = 1
num_photo = 1000
batch_size = 2
size = 1024
BH_lower = 30
BH_upper = 50
wl = 100e-9
D = 6.5
F = 131.4
noise_radius = 20


now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(f'logs_yolo_stellar/yolov5-{date_string}')
curr_dir = Path(f'logs_yolo_stellar/yolov5-{date_string}')
np.random.seed(2024)

# try:
t1 = time.perf_counter()
data_dir = 'tele_datasets/mixed'

RES_DIR = utils.set_res_dir(TRAIN=TRAIN)
# yolov5s.pt
yaml = os.path.join(data_dir, 'data.yaml')


subprocess.run(f'python train.py --data ../{yaml} --weights yolov5s.pt --img {size} --epochs {EPOCHS} --batch-size {batch_size} --name {RES_DIR} --cache', cwd='yolov5', shell=True)





# !python train.py --data ../tele_datasets/mixed/data.yaml --weights yolov5s.pt --img 1024 --epochs 300 --batch-size 4 --cache


t2 = time.perf_counter()

shutil.move('yolov5/runs/', f'{curr_dir}')

result = pd.read_csv(glob.glob(f'{curr_dir}/**/*/results*.csv', recursive=True)[-1])
result.columns = result.columns.str.strip()
df_sorted = result.sort_values(by='metrics/mAP_0.5', ascending=False)
best = df_sorted.iloc[0, :]


box_loss = best['val/box_loss']
obj_loss = best['val/obj_loss']
cls_loss = best['val/cls_loss']
precision= best['metrics/precision']
recall = best['metrics/recall']
mAP_05 = best['metrics/mAP_0.5']
mAP_0595 = best['metrics/mAP_0.5:0.95']


with open(f"{data_dir}/telescope_config.json", "r") as json_file:
    telescope_config = json.load(json_file)

with open(f"{data_dir}/stars_config.json", "r") as json_file:
    stars_config = json.load(json_file)


a = {
    'Model_name': 'yolov5',
    'Batch_size': batch_size,
    'Resolution': size,
    'date': date_string,
    'Training Epoch': EPOCHS,
    'box_loss' : box_loss,
    'obj_loss' : obj_loss,
    'cls_loss' : cls_loss,
    'precision' : precision,
    'recall' : recall,
    'mAP_0.5' : mAP_05,
    'mAP_0.5:0.95': mAP_0595,
    'No. training': num_photo / 5 *4,
    'No. testing': num_photo / 5,
    'Time': t2 - t1,
    'noise_radius': noise_radius,
}

df = pd.read_csv('logs_yolo/results.csv')
df = pd.concat([df, pd.DataFrame([a])], ignore_index=True)
df.to_csv('logs_yolo/results.csv', index=False)



