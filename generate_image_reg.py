import os
import cv2
import json
import time
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from stars import BH_stars_img
from telescope_simulator import TelescopeSimulator


num_round = 3
height = 1024
width = 1024
shape = 'rect'


def generate_image_reg_func(angular_pixel_size_input_image):

    print(f'starting {angular_pixel_size_input_image}')

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
    data_dir = Path(f'tele_datasets/reg_num{num_round}_{shape}_wl{tele_config["wavelength"]:.3e}_'\
    f'D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_focal_length_m"]}_'\
    f'AS{tele_config["angular_pixel_size_input_image"]:.2e}_BHSize{stars_config["BHS_lower_size"]}-{stars_config["BH_upper_size"]}'
    )# data_dir = f'stars{num_stars}_BH{num_BHs}_num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_diameter_m"]}_BHSize{stars_config["BHS_lower_size"]}:{stars_config["BH_upper_size"]}'


    # try:
    if not data_dir.exists():
        data_dir.mkdir()
    # except FileExistsError:
    #     shutil.rmtree(data_dir)
    #     os.mkdir(data_dir)


    image_directory = 'tele_datasets/224/'
    csv_dir = "labels.csv"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    with open(f"{data_dir}/telescope_config.json", "w") as json_file:
        json.dump(tele_config, json_file)

    with open(f"{data_dir}/stars_config.json", "w") as json_file:
        json.dump(stars_config, json_file)

    df = pd.read_csv(csv_dir)
    df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
    df.set_index('PhotoName', inplace=True)
    df_all = pd.DataFrame(columns=df.columns)

    for i in range(num_round):
        df = pd.read_csv(csv_dir)
        df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
        df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
        df.set_index('PhotoName', inplace=True)
        # print(df)
        images = os.listdir(image_directory)
        loop = tqdm(enumerate(images), leave=False)
        for _, image_name in loop:
            if (image_name.split('.')[1] == 'png'):
                # if i == 0:
                new_image = f'{i}_' + image_name
                img_path = data_dir / new_image
                # print(img_path)
                if not img_path.exists():
                    stars_config['BHs'] = image_name
                    img = BH_stars_img(**stars_config)
                    img.stars_gen()
                    img.BHs_gen(rotate=True)
                    noise_BHs = img.add_noise(img.stars_BHs_img, radius=0)
                    tele_config['input_image'] = noise_BHs
                    telescope_simulator = TelescopeSimulator(**tele_config)
                    show=False
                    im_array = img.stars_BHs_img
                    intensity_image = telescope_simulator.get_intensity(im_array, show=show)
                    conv_image = telescope_simulator.get_convolved_image(im_array, intensity_image, show=show)
                    output_img = telescope_simulator.generate_image(conv_image, show=show)
                    # output_img = telescope_simulator.generate_image(show=False)
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
                    cv2.imwrite(str(img_path), new)
                    df.loc[image_name, 'size'] = img.BH_size
                    df.loc[image_name, 'PA'] = img.angle
                    df.loc[image_name, 'new_img'] = str(new_image)
                    df.to_csv(f'{data_dir}/labels_{i}.csv')
        df_all = pd.concat([df_all, df], axis=0)
        df_all.to_csv(f'{data_dir}/labels.csv')



if __name__ == '__main__':
    # [6e-4, 7e-4, 8e-4, 9e-4, 10e-4, 11e-4, 12e-4, 13e-4]
    # [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 14e-4, 15e-4, 16e-4]
    # [1.5e-4, 2.5e-4, 3.5e-4, 4.5e-4, 5.5e-4, 6.5e-4, 7.5e-4, 8.5e-4]
    # [9.5e-4, 10.5e-4, 11.5e-4, 12.5e-4, 13.5e-4, 14e-4, 14.5e-4, 15e-4]
    # [15.5e-4, 16e-4, 16.5e-4, 17e-4, 17.5e-4, 18e-4, 18.5e-4, 19e-4]
    # [14e-4, 15e-4, 15.5e-4, 16e-4, 0.5e-4, 0.6e-4, 0.7e-4, 0.8e-4, 0.9e-4]
    # [1.1e-4 ,1.2e-4, 1.3e-4, 1.4e-4, 1.6e-4, 1.7e-4, 1.8e-4, 1.9e-4]
    angular_pixel_size_input_images = [2e-4, 3e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3, 1.6e-3,
                                       1.7e-3, 1.8e-3, 1.9e-3, 2e-3]
    t1 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(generate_image_reg_func, angular_pixel_size_input_images)
    # generate_image_reg_func(angular_pixel_size_input_images[0])
    t2 = time.perf_counter()



