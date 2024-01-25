import subprocess
import uuid
from stars import BH_stars_img
from telescope_simulator import TelescopeSimulator
import json
import cv2
import os
import pandas as pd
from tqdm import tqdm


num_imgaes = 500
height = 1024
width = 1024
shape = 'rect'


tele_config = dict(
    # physical parameters
    input_image = r"./stars/BHs.png", telescope_diameter_m = 6.5,
    telescope_focal_length_m = 131.4, angular_pixel_size_input_image = 2e-4,
    wavelength = 100e-9, CCD_pixel_size = 2e-4 * 131.4 / 206265,
    CCD_pixel_count = 1024, show = False,
)

stars_config = dict(
    BHs_path='./224/',num_stars=0, num_BHs=1, stars_lower_size=30, stars_upper_size=50,
    height=height, width=width, bg_color=0, shape=shape, BHS_lower_size=64, BH_upper_size=75
)


# /mnt/c/fyq/tele_datasets/
data_dir = f'/mnt/c/fyq/tele_datasets/reg_num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_'\
f'D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_focal_length_m"]}_'\
f'AS{tele_config["angular_pixel_size_input_image"]}_BHSize{stars_config["BHS_lower_size"]}:{stars_config["BH_upper_size"]}'
# data_dir = f'stars{num_stars}_BH{num_BHs}_num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_diameter_m"]}_BHSize{stars_config["BHS_lower_size"]}:{stars_config["BH_upper_size"]}'


# !mkdir {data_dir}
subprocess.call(['mkdir', data_dir])

image_directory = '224/'
csv_dir = "labels.csv"


with open(f"{data_dir}/telescope_config.json", "w") as json_file:
    json.dump(tele_config, json_file)

with open(f"{data_dir}/stars_config.json", "w") as json_file:
    json.dump(stars_config, json_file)


df_train = pd.DataFrame(columns=['images', 'labels'])
df_val = pd.DataFrame(columns=['images', 'labels'])


size_labels = []
PA_labels = [] # positional angle


images = os.listdir(image_directory)
for i, image_name in tqdm(enumerate(images)):
    if (image_name.split('.')[1] == 'png'):
        # if i == 0:
        stars_config['BHs'] = image_name
        img = BH_stars_img(**stars_config)
        img.stars_gen()
        img.BHs_gen(rotate=True)
        noise_BHs = img.add_noise(img.stars_BHs_img, radius=0)
        tele_config['input_image'] = noise_BHs
        telescope_simulator = TelescopeSimulator(**tele_config)
        output_img = telescope_simulator.generate_image(show=False)
        cv2.imwrite(os.path.join(data_dir, image_name), output_img)
        size_labels.append(img.BH_size)
        PA_labels.append(img.angle)

df = pd.read_csv(csv_dir)
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
df.PhotoName = df.PhotoName.apply(lambda x: x.split('/')[-1])
df.set_index('PhotoName', inplace=True)
df['size'] = size_labels
df['PA'] = PA_labels

df.to_csv(f'{data_dir}/labels.csv')





