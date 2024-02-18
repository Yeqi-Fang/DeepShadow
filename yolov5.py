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
num_photo = 3000
batch_size = 8
size = 1024
BH_lower = 30
BH_upper = 50
wl = 100e-9
D = 6.5
F = 131.4
# np.arange(5e-5, 2e-4, 1e-5)
angular_pixel_size_input_images = np.arange(1.9e-4, 1e-5, 5e-4)
# angular_pixel_size_input_image = 5e-5

for angular_pixel_size_input_image in angular_pixel_size_input_images:
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(f'logs_yolo/yolov5-{date_string}')
    curr_dir = Path(f'logs_yolo/yolov5-{date_string}')
    np.random.seed(2024)


    t1 = time.perf_counter()
    data_dirs = glob.glob(f"tele_datasets/stars{star}_BH{BH}_num{num_photo}_rect_wl{wl:.3e}_*{F}"
                        f"*{angular_pixel_size_input_image:.2e}_BHSize{BH_lower}-{BH_upper}")
    assert len(data_dirs) != 0, 'Empty'
    assert len(data_dirs) == 1, "Please specify more parameters!"
    data_dir = data_dirs[0]
    data_dir


    # Visualize a few training images.
    utils.labels_plot(
        image_paths=f'{data_dir}/train/images/*',
        label_paths=f'{data_dir}/train/labels/*',
        num_samples=2, SHOW=False, SAVE=True, save_dir=curr_dir
    )

    RES_DIR = utils.set_res_dir(TRAIN=TRAIN)
    # yolov5s.pt
    yaml = os.path.join(data_dir, 'data.yaml')
    if TRAIN:
        subprocess.run(f'python train.py --data ../{yaml} --weights yolov5s.pt --img {size} --epochs {EPOCHS} '
                    f'--batch-size {batch_size} --name {RES_DIR} --cache', cwd='yolov5')
    else:
        subprocess.run(f'python train.py --weights yolov5s.pt --data ../{yaml} --img {size}'
                    f'--batch-size {batch_size} --name {RES_DIR} --evolve 1000 --cache', cwd='yolov5', capture_output=True)

    # Function to show validation predictions saved during training.


    try:
        os.mkdir('inference')
    except FileExistsError:
        shutil.rmtree('inference')
        os.mkdir('inference')


    inference_lst = list(np.random.choice(os.listdir(f"{data_dir}/train/images"), 10))


    for i in inference_lst:
        shutil.copy(f'{data_dir}/train/images/{i}', f'inference/{i}')


    utils.inference(RES_DIR, 'inference')

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


    # !cat {data_dir}/tele_config.json


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
        'No. star': star,
        'No. BH': BH,
        'Time': t2 - t1,
        'BH_lower_size': stars_config['BHS_lower_size'],
        'BH_upper_size': stars_config['BH_upper_size'],
        'angular_pixel_size_input_image': telescope_config['angular_pixel_size_input_image'],
        'D': telescope_config['telescope_diameter_m'],
        'F': telescope_config['telescope_focal_length_m'],
        'wavelength': telescope_config['wavelength'],
        'init_size': 3072,
        'CCD_pixel_size': telescope_config['CCD_pixel_size'],
        'CCD_pixel_count': telescope_config['CCD_pixel_count']
    }

    df = pd.read_csv('logs_yolo/results.csv')
    df = pd.concat([df, pd.DataFrame([a])], ignore_index=True)
    df.to_csv('logs_yolo/results.csv', index=False)
