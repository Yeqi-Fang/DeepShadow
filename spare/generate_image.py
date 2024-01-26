import uuid
from stars import BH_stars_img
from telescope_simulator import TelescopeSimulator
import json
import cv2
import pandas as pd
from tqdm import tqdm


num_stars = 10
num_BHs = 1
num_imgaes = 500
height = 3072
width = 3072
shape = 'rect'
mode = 'train_val'
# img_lab
# csv


tele_config = dict(
    # physical parameters
    input_image = r"./stars/BHs.png", telescope_diameter_m = 6.5,
    telescope_focal_length_m = 131.4, angular_pixel_size_input_image = 0.5e-4,
    wavelength = 100e-9, CCD_pixel_size = 0.5e-4 * 131.4 / 206265,
    CCD_pixel_count = 3072, show = False,
)

stars_config = dict(
    BHs_path='./224/',num_stars=num_stars, num_BHs=num_BHs, stars_lower_size=30, stars_upper_size=50,
    height=height, width=width, bg_color=0, shape=shape, BHS_lower_size=64, BH_upper_size=75
)


data_dir = f'/mnt/c/fyq/tele_datasets/stars{num_stars}_BH{num_BHs}_num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_'\
f'D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_focal_length_m"]}_'\
f'AS{tele_config["angular_pixel_size_input_image"]}_BHSize{stars_config["BHS_lower_size"]}:{stars_config["BH_upper_size"]}'
# data_dir = f'stars{num_stars}_BH{num_BHs}_num{num_imgaes}_{shape}_wl{tele_config["wavelength"]:.3e}_D{tele_config["telescope_diameter_m"]:.2f}_F{tele_config["telescope_diameter_m"]}_BHSize{stars_config["BHS_lower_size"]}:{stars_config["BH_upper_size"]}'


data_dir


# if mode == 'train_val':
!mkdir {data_dir}
!mkdir {data_dir}/train
!mkdir {data_dir}/train/images
!mkdir {data_dir}/train/labels
!mkdir {data_dir}/validation
!mkdir {data_dir}/validation/images
!mkdir {data_dir}/validation/labels
# elif mode == 'img_lab':
# !mkdir {data_dir}
# !mkdir {data_dir}/images
# !mkdir {data_dir}/images/train
# !mkdir {data_dir}/images/validation
# !mkdir {data_dir}/labels
# !mkdir {data_dir}/labels/train
# !mkdir {data_dir}/labels/validation
# else:
# !mkdir {data_dir}
# !mkdir {data_dir}/images
# !mkdir {data_dir}/labels


with open(f'{data_dir}/data.yaml', 'w') as f:
    f.write(f'path: {data_dir}\n')
    f.write('train: ./train/images\n')
    f.write('val: ../validation/images\n\n')
    f.write('nc: 2\n')
    f.write("names: ['star', 'BH']\n")



with open(f"{data_dir}/telescope_config.json", "w") as json_file:
    json.dump(tele_config, json_file)

with open(f"{data_dir}/stars_config.json", "w") as json_file:
    json.dump(stars_config, json_file)


df_train = pd.DataFrame(columns=['images', 'labels'])
df_val = pd.DataFrame(columns=['images', 'labels'])


num_train = int(num_imgaes * 0.8)
num_val = min(int(num_imgaes * 0.2), 240)

for i in tqdm(range(num_train)):
    img = BH_stars_img(**stars_config)
    img.stars_gen()
    img.BHs_gen()
    noise_BHs = img.add_noise(img.stars_BHs_img, radius=0)
    tele_config['input_image'] = noise_BHs
    telescope_simulator = TelescopeSimulator(**tele_config)
    code = uuid.uuid4()
    if mode == 'train_val':
        bh_path = f"{data_dir}/train/images/BHs_{code}.png"
        # bh_path_origin = f"{data_dir}/train/images/BHs_{code}_origin.png"
        txt_path = f"{data_dir}/train/labels/BHs_{code}.txt"
    elif mode == 'img_lab':
        bh_path = f"{data_dir}/images/train/BHs_{code}.png"
        txt_path = f"{data_dir}/labels/train/BHs_{code}.txt"
    else:
        df_train = pd.concat([df_train, pd.DataFrame({"images":[f"BHs_{code}.png"], "labels": [f"BHs_{code}.txt"]})])
        bh_path = f"{data_dir}/images/BHs_{code}.png"
        txt_path = f"{data_dir}/labels/BHs_{code}.txt"
    # print(i)
    output_img = telescope_simulator.generate_image(show=False)
    cv2.imwrite(bh_path, output_img)
    img.txtGen(txt_path=txt_path)



for i in tqdm(range(num_val)):
    img = BH_stars_img(**stars_config)
    img.stars_gen()
    img.BHs_gen()
    noise_BHs = img.add_noise(img.stars_BHs_img, radius=5)
    tele_config['input_image'] = noise_BHs
    telescope_simulator = TelescopeSimulator(**tele_config)
    code = uuid.uuid4()
    if mode == 'train_val':
        bh_path = f"{data_dir}/validation/images/BHs_{code}.png"
        txt_path = f"{data_dir}/validation/labels/BHs_{code}.txt"
    elif mode == 'img_lab':
        bh_path = f"{data_dir}/images/validation/BHs_{code}.png"
        txt_path = f"{data_dir}/labels/validation/BHs_{code}.txt"
    else:
        df_val = pd.concat([df_val, pd.DataFrame({"images":[f"BHs_{code}.png"], "labels": [f"BHs_{code}.txt"]})])
        bh_path = f"{data_dir}/images/BHs_{code}.png"
        txt_path = f"{data_dir}/labels/BHs_{code}.txt"
    # cv2.imwrite(bh_path, noise_BHs)
    output_img = telescope_simulator.generate_image(bh_path, show=False)
    # cv2.imwrite(bh_path, output_img)
    img.txtGen(txt_path=txt_path)


if mode == 'csv':
    df_train.to_csv(f"{data_dir}/train.csv", header=False, index=False)
    df_val.to_csv(f"{data_dir}/validation.csv", header=False, index=False)


import glob
img_lst = glob.glob(f'{data_dir}/*/*/*.png')


for img_name in tqdm(img_lst):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (1024, 1024))
    cv2.imwrite(img_name, img_resize)


data_dir


# !mv '/mnt/c/fyq/tele_datasets/stars10_BH1_num500_rect_wl5.000e-08_' '/mnt/c/fyq/tele_datasets/stars10_BH1_num500_rect_wl5.000e-08_D6.50_F131.4_AS0.0001_BHSize64:75'






