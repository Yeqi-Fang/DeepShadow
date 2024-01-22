import utils_tele as utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import imageio
import os
from scipy import interpolate


class TelescopeSimulator():
    def __init__(self, input_image_name, telescope_diameter_m, telescope_focal_length_m,
                 seeing_arcsec_500nm, zenith_angle_deg, atmosphere, wavelength, CCD_pixel_size,
                 CCD_pixel_count, Num_psfs_to_gen, pixels_per_ro):
        """_summary_

        Args:
            input_image_name (_type_): _description_
            telescope_diameter_m (_type_): _description_
            telescope_focal_length_m (_type_): _description_
            seeing_arcsec_500nm (_type_): _description_
            zenith_angle_deg (_type_): _description_
            atmosphere (_type_): _description_
            wavelength (_type_): _description_
            CCD_pixel_size (_type_): _description_
            CCD_pixel_count (_type_): _description_
            Num_psfs_to_gen (_type_): _description_
            pixels_per_ro (_type_): _description_
        """
        
        
        self.input_image_name = input_image_name
        self.telescope_diameter_m = telescope_diameter_m
        self.telescope_focal_length_m = telescope_focal_length_m
        self.seeing_arcsec_500nm = seeing_arcsec_500nm
        self.zenith_angle_deg = zenith_angle_deg
        self.atmosphere = atmosphere
        self.wavelength = wavelength
        self.CCD_pixel_size = CCD_pixel_size
        self.CCD_pixel_count = CCD_pixel_count
        self.Num_psfs_to_gen = Num_psfs_to_gen
        self.pixels_per_ro = pixels_per_ro
        
        self.image_arr = self.get_image()
        self.angular_pixel_size_input_image = 206265 / telescope_focal_length_m * CCD_pixel_size * CCD_pixel_count / self.image_arr.shape[0]
        self.pixel_size_input_image = utils.angular_to_physical_pixels(self.angular_pixel_size_input_image, telescope_focal_length_m)
        
        self.intensity = self.get_intensity(self.image_arr)
        self.convolved_array = self.get_convolved_image(self.image_arr, self.intensity)

        
        
        
    def get_physical_parameters(self):
        theta = 1.22 * wavelength / telescope_diameter_m  # in radians
        T = 2.898e-3 / wavelength
        r = 6e4 * (1e8 / T)**(4/3)
        D = r * 2
        accretion_D = D * 50
        accretion_D / theta / 9.5e15
        print(f'{r/3000:.2e} solar mass')

    def get_image(self, show=True):
        # open image to convolve
        im = Image.open(self.input_image_name)
        im_array = np.asarray(im)
        if np.shape(im_array)[0] != np.shape(im_array)[1]:
            min_dim, max_dim = np.min((np.shape(im_array)[0], np.shape(im_array)[1])), np.max((np.shape(im_array)[0], np.shape(im_array)[1]))
            pixels_to_crop = max_dim - min_dim
            if np.shape(im_array)[0] > np.shape(im_array)[1]:
                left, right = 0, np.shape(im_array)[1]
                top, bottom = np.floor(pixels_to_crop/2), np.shape(im_array)[0] - np.ceil(pixels_to_crop/2)

            if np.shape(im_array)[0] < np.shape(im_array)[1]:
                left, right = np.floor(pixels_to_crop/2), np.shape(im_array)[1] - np.ceil(pixels_to_crop/2)
                top, bottom = 0, np.shape(im_array)[0]

            im = im.crop((left, top, right, bottom))
            im_array = np.asarray(im)
        if show:
            plt.imshow(im, cmap='gray')
            plt.show()

        return im_array


    def get_intensity(self, im_array, show=True):
        
        

        ideal_pixel_size_pupil = (wavelength*telescope_focal_length_m) / (len(im_array)*self.pixel_size_input_image)
        print("pixel_size_input_image", self.pixel_size_input_image)


        r0_cm = utils.fried_parameter_cm(wavelength, arcseconds_of_seeing_500nm=seeing_arcsec_500nm, zenith_angle_deg=zenith_angle_deg)
        telescope_aperture_width_pixels = int(np.ceil((pixels_per_ro/(r0_cm*0.01))*telescope_diameter_m))
        Pixel_size_pupil_plane = telescope_diameter_m/telescope_aperture_width_pixels


        print("telescope_aperture_width_pixels: ", telescope_aperture_width_pixels)


        pixel_size_psf_image_plane = (wavelength*telescope_focal_length_m)/(len(im_array)*Pixel_size_pupil_plane)


        phase_screen = np.zeros((telescope_aperture_width_pixels, telescope_aperture_width_pixels), dtype=np.complex64)
        complex_amplitude = utils.quick_complex_pupil(phase_screen, array_to_propgate_size=len(im_array[0]))
        intensity_image = utils.Focus_beam(complex_amplitude)


        # intensity_image.min(), intensity_image.max()

        # plt.imshow(signal.convolve(im_array, intensity_image))



        # x_psf_samples = np.linspace(-pixel_size_psf_image_plane*len(intensity_image)/2, pixel_size_psf_image_plane*len(intensity_image)/2, len(intensity_image))
        # y_psf_samples = np.linspace(-pixel_size_psf_image_plane*len(intensity_image)/2, pixel_size_psf_image_plane*len(intensity_image)/2, len(intensity_image))

        # f = interpolate.interp2d(x_psf_samples, y_psf_samples, intensity_image, kind='cubic')

        # x_input_image = np.linspace(-pixel_size_input_image*len(im_array)/2, pixel_size_input_image*len(im_array)/2, len(im_array))
        # y_input_image = np.linspace(-pixel_size_input_image*len(im_array)/2, pixel_size_input_image*len(im_array)/2, len(im_array))

        # resampled_psf = f(x_input_image, y_input_image)
        # intensity_image = resampled_psf


        intensity_image = intensity_image - intensity_image.min()
        intensity_image /= np.sum(intensity_image)

        if show:
            plt.imshow(intensity_image)
            plt.show()


        return intensity_image

    def get_convolved_image(self, im_array, intensity_image, show=True):
        
        convolved_array_shape = np.shape(signal.convolve(im_array, intensity_image))


        convolved_array = np.zeros((convolved_array_shape[0], convolved_array_shape[1], 3))
        # for i in range(0, 3):
        convolved_array = signal.convolve(im_array, intensity_image, method='auto')
        convolved_array = np.uint8((convolved_array)*(255/np.max(convolved_array)))


        size_conv = convolved_array.shape[0]


        interval = (size_conv + 1) // 4


        convolved_array = convolved_array[interval:3*interval, interval: 3*interval]


        # convolved_array.sum() / im_array.sum()

        if show:
            plt.imshow(convolved_array)
            plt.show()
        
        return convolved_array
    
    def generate_image(self, out_dir, show=True):
        x_psf_samples = np.linspace(-self.pixel_size_input_image*len(self.convolved_array)/2, self.pixel_size_input_image*len(self.convolved_array)/2, len(self.convolved_array))
        y_psf_samples = np.linspace(-self.pixel_size_input_image*len(self.convolved_array)/2, self.pixel_size_input_image*len(self.convolved_array)/2, len(self.convolved_array))

        x_CCD = np.linspace(-CCD_pixel_size*CCD_pixel_count/2,CCD_pixel_size*CCD_pixel_count/2, CCD_pixel_count)
        y_CCD = np.linspace(-CCD_pixel_size*CCD_pixel_count/2,CCD_pixel_size*CCD_pixel_count/2, CCD_pixel_count)

        output_image = np.zeros((CCD_pixel_count, CCD_pixel_count, 3))
        f = interpolate.interp2d(x_psf_samples, y_psf_samples, self.convolved_array, kind='cubic')
        output_image = f(x_CCD, y_CCD)
        output_image = np.uint8((output_image)*(255/np.max(output_image)))


        # current_directory = os.getcwd()
        # final_directory = os.path.join(current_directory, r'output_images_2')
        # if not os.path.exists(final_directory):
            # os.makedirs(final_directory)
        # os.chdir(final_directory)
        imageio.imwrite(out_dir, output_image)
        # os.chdir(current_directory)
        if show:
            plt.imshow(output_image)
            plt.show()
        
        return output_image




if __name__ == '__main__':
    # physical parameters
    input_image_name = r"./stars/BHs.png"
    telescope_diameter_m = 6.5  # in meters
    telescope_focal_length_m = 131.4  # in meters
    seeing_arcsec_500nm = .015  # in arcseconds
    zenith_angle_deg = 0  # in deg, zero being at the zenith
    atmosphere = False  # True or False, if True simulates atmospheric perturbations. If False simulates purely diffraction effects
    wavelength = 50e-9  # in meters
    CCD_pixel_size = 6e-5  # in meters
    CCD_pixel_count = 1000  # The pixel width of your simulated CCD
    
    # simulation parameters
    Num_psfs_to_gen = 10# number of psfs (and in turn output images) the run will generate
    pixels_per_ro = 30  # how well you wish to sample your phase screen
    
    
    telescope_simulator = TelescopeSimulator(input_image_name, telescope_diameter_m,telescope_focal_length_m,
                 seeing_arcsec_500nm, zenith_angle_deg, atmosphere, wavelength, CCD_pixel_size,
                 CCD_pixel_count, Num_psfs_to_gen, pixels_per_ro)
    
    telescope_simulator.generate_image(r'Telescope_Simulator-master/output_images_2/psf.png', show=True)
