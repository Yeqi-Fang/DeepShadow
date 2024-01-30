import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

class BH_stars_img():

    def __init__(self, BHs_path='../224', BHs=None, height=1000, width=1400, bg_color=33, 
                 num_stars=200, num_BHs=10, stars_lower_size=10, stars_upper_size=25,
                 BHS_lower_size=10, BH_upper_size=30, shape='rect'):
        """_summary_

        Args:
            BHs_path (str, optional): _description_. Defaults to '../224'.
            height (int, optional): _description_. Defaults to 1000.
            width (int, optional): _description_. Defaults to 1400.
            bg_color (int, optional): _description_. Defaults to 33.
            num_stars (int, optional): _description_. Defaults to 200.
            num_BHs (int, optional): _description_. Defaults to 10.
            stars_lower_size (int, optional): _description_. Defaults to 10.
            stars_upper_size (int, optional): _description_. Defaults to 25.
            BHS_lower_size (int, optional): _description_. Defaults to 25.
            BH_upper_size (int, optional): _description_. Defaults to 25.
            shape (str, optional): _description_. Defaults to 'rect'.
        """
        
        self.num_BHs = num_BHs
        if BHs is None:
            self.BHs_path = BHs_path
            self.BHs= list(np.random.choice(os.listdir(BHs_path), num_BHs))
        elif type(BHs) == str or type(BHs) == np.ndarray:
            self.BHs_path = BHs_path
            self.BHs= [BHs]
        else:
            raise ValueError
        self.height, self.width = height, width
        self.bg_color = bg_color
        self.num_stars = num_stars
        self.num_BHs = num_BHs
        self.stars_lower_size = stars_lower_size
        self.stars_upper_size = stars_upper_size
        self.BHS_lower_size = BHS_lower_size
        self.BH_upper_size = BH_upper_size
        self.shape = shape
        # sanity check
        assert self.shape in ['rect', 'circle'], "Only support squares and circles !!!"
        if self.shape == 'circle':
            assert self.width == self.height, "Only support square images !!!"

    def generate_distribution(self, size, sigma_x, sigma_y, center_color):
        """_summary_

        Args:
            size (_type_): _description_
            sigma_x (_type_): _description_
            sigma_y (_type_): _description_
            center_color (_type_): _description_

        Returns:
            _type_: _description_
        """        
        x = np.linspace(-size / 2, size / 2, size)
        y = np.linspace(-size / 2, size / 2, size)
        X, Y = np.meshgrid(x, y)
        X /= size / 2
        Y /= size / 2
        rho = np.random.uniform(0, 0.1)
        dist_out = np.sqrt((X / sigma_x)**2 + (Y / sigma_y)**2 - 2*rho*X*Y / sigma_x / sigma_y)
        dist_out = np.sqrt((X / sigma_x)**2 + (Y / sigma_y)**2 - 2*rho*X*Y / sigma_x / sigma_y)
        alpha = np.random.uniform(10, 15)
        u = 0.5
        brightness = center_color / (np.exp(alpha * (dist_out - u)) + 1)
        return brightness
    
    def put_small_images_tolarge_background(self, small_img, bg_height, bg_width, small_img_size):
        """_summary_

        Args:
            small_img (_type_): _description_
            bg_height (_type_): _description_
            bg_width (_type_): _description_
            small_img_size (_type_): _description_

        Returns:
            _type_: _description_
        """        
        pad1 = np.random.randint(0, bg_height - small_img_size)
        pad2 = bg_height - small_img_size - pad1
        pad3 = np.random.randint(0, bg_width - small_img_size)
        pad4 = bg_width - small_img_size - pad3
        # get the coordinate of the center of the square
        xc = (pad3 + small_img_size / 2) / bg_width
        yc = (pad1 + small_img_size / 2) / bg_height
        larger_img = cv2.copyMakeBorder(small_img, pad1, pad2, pad3, pad4, cv2.BORDER_CONSTANT, value=0)
        return larger_img, xc, yc
    
    def add_noise(self, img, radius=5):
        """_summary_

        Args:
            img (_type_): _description_
            radius (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """        
        noisy_img = img + np.random.normal(loc=0, scale=radius, size=(self.height, self.width))
        noisy_img = np.where(noisy_img > 255, 255, noisy_img)
        noisy_img = np.where(noisy_img < 0, 0, noisy_img)
        noisy_img = noisy_img.astype(np.int64)
        return noisy_img

    def bg_gen(self):
        """_summary_

        Returns:
            _type_: _description_
        """        
        bg = np.ones((self.height, self.width), dtype=np.float64) * self.bg_color / 255
        return bg
    
    def stars_gen(self):
        """_summary_
        """        
        bg = self.bg_gen()
        self.stars_BHs_img = bg.copy()
        if self.shape == 'rect': 
            stars_lst = np.ones((self.num_stars, 4), dtype=np.float64)
        elif self.shape == 'circle':
            stars_lst = np.ones((self.num_stars, 3), dtype=np.float64)
        for i in range(self.num_stars):
            # Parameters
            size = np.random.randint(low=self.stars_lower_size, high=self.stars_upper_size)  # Size of the square grid
            sigma_x = np.random.uniform(0.9, 1.1)  # Standard deviation for the Gaussian distribution
            sigma_y = np.random.uniform(0.9, 1.1)  # Standard deviation for the Gaussian distribution
            luminosity = (size - self.stars_lower_size) / (self.stars_upper_size - self.stars_lower_size) * \
                         np.random.uniform(0.5, 0.8)  # Maximum brightness at the center
            brightness = self.generate_distribution(size, sigma_x, sigma_y, luminosity)
            larger_img, xc, yc = self.put_small_images_tolarge_background(brightness, self.height, self.width, size)
            self.stars_BHs_img += larger_img
            stars_lst[i, 0] = xc
            stars_lst[i, 1] = yc
            if self.shape == 'rect':
                stars_lst[i, 2] = size / self.width
                stars_lst[i, 3] = size / self.height
            elif self.shape == 'circle':
                stars_lst[i, 2] = size / self.width
        self.stars_BHs_img *= 255
        self.stars_BHs_img = self.stars_BHs_img.astype(np.int64)
        self.stars_lst = stars_lst
    
    def BHs_gen(self, rotate=False):
        """_summary_
        """
        if self.shape == 'rect':
            BH_lst = np.ones((len(self.BHs), 4), dtype=np.float64)
        elif self.shape == 'circle':
            BH_lst = np.ones((len(self.BHs), 3), dtype=np.float64)
            
        for index, path in enumerate(self.BHs):
            # print(self.BHs_path + path)
            # print(path)
            # print(type(path))
            if type(path) == str or type(path) == np.str_:
                BHimg = cv2.imread(self.BHs_path + path)
                BHimg = cv2.cvtColor(BHimg, cv2.COLOR_BGR2GRAY)
            elif type(path) == np.ndarray:
                BHimg = path
            else:
                raise ValueError
            BH_size = np.random.randint(low=self.BHS_lower_size, high=self.BH_upper_size)
            BHimg_small = cv2.resize(BHimg, (BH_size, BH_size))
            BHimg_small = BHimg_small.astype(np.float64)
            if rotate:
                # rotate the images
                centerX, centerY = (BH_size // 2, BH_size // 2)
                angle = np.random.uniform(0, 360)
                M = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
                BHimg_small = cv2.warpAffine(BHimg_small, M, (BH_size, BH_size))
                self.angle = angle
            bright_factor = np.random.uniform(0.6, 0.9)
            BHimg_small *= bright_factor
            BHimg_small = BHimg_small.astype(np.int64)
            BHimg_small = np.where(BHimg_small < self.bg_color, self.bg_color, BHimg_small)
            # BHimg_small = np.where(BHimg_small < self.bg_color + 10, self.bg_color, BHimg_small)
            BH_larger, xc, yc = self.put_small_images_tolarge_background(BHimg_small - self.bg_color,
                                                                         self.height, self.width, BH_size)
            self.stars_BHs_img += BH_larger
            BH_lst[index, 0] = xc
            BH_lst[index, 1] = yc
            if self.shape == 'rect':
                BH_lst[index, 2] = BH_size / self.width
                BH_lst[index, 3] = BH_size / self.height
            elif self.shape == 'circle':
                BH_lst[index, 2] = BH_size / self.width
        
        self.stars_BHs_img = np.where(self.stars_BHs_img > 255, 255, self.stars_BHs_img)
        self.stars_BHs_img = np.where(self.stars_BHs_img < 0, 0, self.stars_BHs_img)
        self.BH_lst = BH_lst
        self.BH_size = BH_size


        return self.stars_BHs_img
    def save(self, path):
        """_summary_

        Args:
            path (_type_): _description_
        """        
        cv2.imwrite(path, self.stars_BHs_img)


    def txtGen(self, txt_path='labels.txt'):
        """_summary_

        Args:
            txt_path (str, optional): _description_. Defaults to 'labels.txt'.
        """        
        if self.shape == 'rect':
            
            BH_df = pd.DataFrame(self.BH_lst, columns=['xc', 'yc', 'width', 'height'])
            BH_df['type'] = 1
            type_column = BH_df.pop('type')
            BH_df.insert(0, 'type', type_column)
        
            stars_df = pd.DataFrame(self.stars_lst, columns=['xc', 'yc', 'width', 'height'])
            stars_df['type'] = 0
            type_column = stars_df.pop('type')
            stars_df.insert(0, 'type', type_column)
        
        elif self.shape == 'circle':
            BH_df = pd.DataFrame(self.BH_lst, columns=['xc', 'yc', 'radius'])
            BH_df['type'] = 1
            type_column = BH_df.pop('type')
            BH_df.insert(0, 'type', type_column)
        
            stars_df = pd.DataFrame(self.stars_lst, columns=['xc', 'yc', 'radius'])
            stars_df['type'] = 0
            type_column = stars_df.pop('type')
            stars_df.insert(0, 'type', type_column)

        df = pd.concat([BH_df, stars_df])
        # Convert to TXT file with space as delimiter
        df.to_csv(txt_path, sep=' ', index=False, header=False)


if __name__ == '__main__':

    img_arr = cv2.imread('tele_datasets/224/20240108150459_4eae584618014ddca128ea99277295e2.png', 0)
    img = BH_stars_img(BHs_path='tele_datasets/224/', BHs=img_arr, num_stars=0, num_BHs=1, stars_lower_size=25, stars_upper_size=35,
                       BHS_lower_size=256, BH_upper_size=257, height=700, width=700, bg_color=0, shape='rect')
    img.stars_gen()
    img.save('stars/stars.png')
    noise_stars = img.add_noise(img.stars_BHs_img, radius=7)
    cv2.imwrite('stars/noisy_stars.png', noise_stars)
    img.BHs_gen()
    img.save('stars/BHs.png')
    noise_BHs = img.add_noise(img.stars_BHs_img, radius=7)
    cv2.imwrite('stars/noisy_BHs.png', noise_BHs)
    # plt.imshow(img.stars_BHs_img, cmap='gray')
    # plt.show()
    img.txtGen(txt_path='stars/labels.txt')

