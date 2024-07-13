import uuid

from matplotlib import pyplot as plt
from stars import BH_stars_img
from telescope_simulator import TelescopeSimulator
import json
import cv2
import os
import pandas as pd
from tqdm import tqdm
import glob
import time
import concurrent.futures
import shutil
import numpy as np


num_imgaes = 10
height = 1024
width = 1024
shape = 'rect'
mode = 'train_val'
noise_radius = 0

def generate_image_func(angular_pixel_size_input_image):

    tele_config = dict(
        input_image = r"./stars/BHs.png", telescope_diameter_m = 6.5,
        telescope_focal_length_m = 131.4, angular_pixel_size_input_image = angular_pixel_size_input_image,
        wavelength = 100e-9, CCD_pixel_size = angular_pixel_size_input_image * 131.4 / 206265,
        CCD_pixel_count = 1024, show = False,
    )

    stars_config = dict(
        BHs_path='tele_datasets/224/',num_stars=1, num_BHs=0, stars_lower_size=5, stars_upper_size=15,
        height=height, width=width, bg_color=0, shape=shape, BHS_lower_size=30, BH_upper_size=50
    )

    data_dir = f'tele_datasets/Changing/num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_'\
    f'D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_focal_length_m"]}_'\
    f'AS{tele_config["angular_pixel_size_input_image"]:.2e}_BHSize{stars_config["BHS_lower_size"]}-{stars_config["BH_upper_size"]}'


    if mode == 'train_val':
        try:
            os.mkdir(f'{data_dir}')
            os.mkdir(f'{data_dir}/images')
            os.mkdir(f'{data_dir}/labels')
        except:
            shutil.rmtree(f'{data_dir}')
            os.mkdir(f'{data_dir}')
            os.mkdir(f'{data_dir}/images')
            os.mkdir(f'{data_dir}/labels')


    with open(f'{data_dir}/data.yaml', 'w') as f:
        f.write(f'path: ../{data_dir}\n')
        f.write('train: ./train/images\n')
        f.write('val: ../validation/images\n\n')
        f.write('nc: 2\n')
        f.write("names: ['star', 'BH']\n")


    with open(f"{data_dir}/telescope_config.json", "w") as json_file:
        json.dump(tele_config, json_file)

    with open(f"{data_dir}/stars_config.json", "w") as json_file:
        json.dump(stars_config, json_file)


    df = pd.DataFrame(columns=['images', 'labels'])


    for i in tqdm(range(num_imgaes)):
        img = BH_stars_img(**stars_config)
        img.stars_gen()
        # img.BHs_gen()
        # noise_BHs = img.add_noise(img.stars_BHs_img, radius=noise_radius)
        tele_config['input_image'] = img.stars_BHs_img
        telescope_simulator = TelescopeSimulator(**tele_config)
        code = uuid.uuid4()
        if mode == 'train_val':
            bh_path = f"{data_dir}/images/BHs_{code}.png"
            # bh_path_origin = f"{data_dir}/train/images/BHs_{code}_origin.png"
            txt_path = f"{data_dir}/labels/BHs_{code}.txt"
        elif mode == 'img_lab':
            bh_path = f"{data_dir}/images/BHs_{code}.png"
            txt_path = f"{data_dir}/labels/BHs_{code}.txt"
        else:
            df = pd.concat([df, pd.DataFrame({"images":[f"BHs_{code}.png"], "labels": [f"BHs_{code}.txt"]})])
            bh_path = f"{data_dir}/images/BHs_{code}.png"
            txt_path = f"{data_dir}/labels/BHs_{code}.txt"
        # print(i)
        show=False
        im_array = img.stars_BHs_img
        # plt.imshow(im_array)
        # plt.show()
        intensity_image = telescope_simulator.get_intensity(im_array, show=show)
        conv_image = telescope_simulator.get_convolved_image(im_array, intensity_image, show=show)
        output_img = telescope_simulator.generate_image(conv_image, show=show)
        # output_img = telescope_simulator.generate_image(show=False)
        output_img = output_img.astype(np.int64)
        noisy_img = output_img + np.random.normal(loc=0, scale=noise_radius, size=(height, width))
        noisy_img = np.where(noisy_img > 255, 255, noisy_img)
        noisy_img = np.where(noisy_img < 0, 0, noisy_img)
        noisy_img = noisy_img.astype(np.uint8)
        noisy_img = cv2.resize(noisy_img, (1024, 1024))
        cv2.imwrite(bh_path, noisy_img)
        img.txtGen(txt_path=txt_path)



    if mode == 'csv':
        df.to_csv(f"{data_dir}/train.csv", header=False, index=False)




if __name__ == '__main__':
    # np.arange(5e-5, 2e-4, 1e-5)
    angular_pixel_size_input_images = np.arange(2e-4, 1e-3, 5e-5)
    t1 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        executor.map(generate_image_func, angular_pixel_size_input_images)
    # for i in angular_pixel_size_input_images:
    #     generate_image_func(i)
    # generate_image_func(5e-4)    
    t2 = time.perf_counter()
    with open('tele_datasets/records.txt', 'a') as f:
        for i in angular_pixel_size_input_images:
            f.write(f'{i:2e}\n')
