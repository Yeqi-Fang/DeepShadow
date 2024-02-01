
from stars import BH_stars_img
from telescope_simulator import TelescopeSimulator
import json
import cv2
import os
import time
import pandas as pd
from tqdm import tqdm
import shutil
import concurrent.futures


def generate_image_reg_func(angular_pixel_size_input_image):

    print(f'starting {angular_pixel_size_input_image}')
    num_imgaes = 500
    height = 1024
    width = 1024
    shape = 'rect'
    # angular_pixel_size_input_image = 7e-4


    tele_config = dict(
        # physical parameters
        input_image = r"./stars/BHs.png", telescope_diameter_m = 6.5,
        telescope_focal_length_m = 131.4, angular_pixel_size_input_image = angular_pixel_size_input_image,
        wavelength = 100e-9, CCD_pixel_size = angular_pixel_size_input_image * 131.4 / 206265,
        CCD_pixel_count = 1024, show = False,
    )

    stars_config = dict(
        BHs_path='tele_datasets/224/',num_stars=0, num_BHs=1, stars_lower_size=30, stars_upper_size=50,
        height=height, width=width, bg_color=0, shape=shape, BHS_lower_size=64, BH_upper_size=75
    )


    # /mnt/c/fyq/tele_datasets/
    data_dir = f'tele_datasets/reg_num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_'\
    f'D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_focal_length_m"]}_'\
    f'AS{tele_config["angular_pixel_size_input_image"]:.2e}_BHSize{stars_config["BHS_lower_size"]}-{stars_config["BH_upper_size"]}'
    # data_dir = f'stars{num_stars}_BH{num_BHs}_num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_diameter_m"]}_BHSize{stars_config["BHS_lower_size"]}:{stars_config["BH_upper_size"]}'


    # try:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # except FileExistsError:
    #     shutil.rmtree(data_dir)
    #     os.mkdir(data_dir)


    SIZE = 300
    # IN_SIZE = 8
    loss_fn = 'mse'
    num_epochs = 50
    BATCH_SIZE = 128
    critical_mae = 30
    DROPOUT_RATE = 0.5
    learning_rate = 1e-3
    weight_decay = 1e-4
    image_directory = 'tele_datasets/224/'
    csv_dir = "labels.csv"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    with open(f"{data_dir}/telescope_config.json", "w") as json_file:
        json.dump(tele_config, json_file)

    with open(f"{data_dir}/stars_config.json", "w") as json_file:
        json.dump(stars_config, json_file)


    df_train = pd.DataFrame(columns=['images', 'labels'])
    df_val = pd.DataFrame(columns=['images', 'labels'])


    size_labels = []
    PA_labels = [] # positional angle


    df = pd.read_csv(csv_dir)
    df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
    # df.set_index('PhotoName', inplace=True)


    images = os.listdir(image_directory)
    loop = tqdm(enumerate(images), leave=False)
    for i, image_name in loop:
        if (image_name.split('.')[1] == 'png'):
            # if i == 0:
            if not os.path.exists(os.path.join(data_dir, image_name)):
            # print(os.path.join(data_dir, image_name))
            # print(os.path.exists(os.path.join(data_dir, image_name)))
                stars_config['BHs'] = image_name
                img = BH_stars_img(**stars_config)
                img.stars_gen()
                img.BHs_gen(rotate=True)
                noise_BHs = img.add_noise(img.stars_BHs_img, radius=0)
                tele_config['input_image'] = noise_BHs
                telescope_simulator = TelescopeSimulator(**tele_config)
                output_img = telescope_simulator.generate_image(show=False)
                img_size = tele_config['CCD_pixel_count']
                x, y, r, _ = img.BH_lst[0] * img_size
                x, y, r = int(x), int(y), int(r)
                if (x-120) >= 0 and (x + 120) < img_size and (y-120) >= 0 and (y + 120) < img_size:
                    pass
                else:
                    if x - 120 < 0:
                        x = 120
                    elif x + 120 >= img_size:
                        x = img_size - 120 - 1
                    if y - 120 < 0:
                        y = 120
                    elif y + 120 >= img_size:
                        y = img_size - 120 - 1
                xl, xr, yl, yr = x - 120, x + 120, y - 120, y + 120
                new = output_img[yl: yr, xl: xr]
                cv2.imwrite(os.path.join(data_dir, image_name), new)
                df.loc[i, 'size'] = img.BH_size
                df.loc[i, 'PA'] = img.angle
                df.to_csv(f'{data_dir}/labels.csv')





if __name__ == '__main__':
    # [6e-4, 7e-4, 8e-4, 9e-4, 10e-4, 11e-4, 12e-4, 13e-4]
    # [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 14e-4, 15e-4, 16e-4]
    # [1.5e-4, 2.5e-4, 3.5e-4, 4.5e-4, 5.5e-4, 6.5e-4, 7.5e-4, 8.5e-4]
    # [9.5e-4, 10.5e-4, 11.5e-4, 12.5e-4, 13.5e-4, 14e-4, 14.5e-4, 15e-4]
    angular_pixel_size_input_images = [15.5e-4, 16e-4, 16.5e-4, 17e-4, 17.5e-4, 18e-4, 18.5e-4, 19e-4]
    t1 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(generate_image_reg_func, angular_pixel_size_input_images)
    t2 = time.perf_counter()



