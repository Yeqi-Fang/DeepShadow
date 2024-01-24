import utils_tele as utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import imageio
import os
from scipy import interpolate


class TelescopeSimulator():
    def __init__(self, input_image, telescope_diameter_m, telescope_focal_length_m,
                 seeing_arcsec_500nm, zenith_angle_deg, atmosphere, wavelength, CCD_pixel_size,
                 CCD_pixel_count, Num_psfs_to_gen, pixels_per_ro, show=True):
        """ 
        
        Modeling optical telescope imaging, including optical diffraction and atmospheric scattering effects

        Args:
            input_image (str or numpy.ndarray): _description_
            telescope_diameter_m (float):  in meters
            telescope_focal_length_m (float): in arcseconds
            seeing_arcsec_500nm (float): 
            zenith_angle_deg (float): in deg, zero being at the zenith
            atmosphere (bool): True or False, if True simulates atmospheric perturbations. If False simulates purely diffraction effects
            wavelength (float): 
            CCD_pixel_size (float): The pixel width of your simulated CCD
            CCD_pixel_count (int): _description_
            Num_psfs_to_gen (int): number of psfs (and in turn output images) the run will generate
            pixels_per_ro (int): how well you wish to sample your phase screen
            show (bool): True or False, if True shows the psf and phase screen
        """

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

        self.image_arr = self.get_image(input_image, show=show)
        
        self.angular_pixel_size_input_image = 206265 / telescope_focal_length_m * CCD_pixel_size * CCD_pixel_count / self.image_arr.shape[0]
        self.pixel_size_input_image = utils.angular_to_physical_pixels(self.angular_pixel_size_input_image, telescope_focal_length_m)
        
        self.intensity = self.get_intensity(self.image_arr, show=show)
        self.convolved_array = self.get_convolved_image(self.image_arr, self.intensity, show=show)
        
        
        
    def get_physical_parameters(self):
        theta = 1.22 * wavelength / telescope_diameter_m  # in radians
        T = 2.898e-3 / wavelength
        r = 6e4 * (1e8 / T)**(4/3)
        D = r * 2
        accretion_D = D * 50
        accretion_D / theta / 9.5e15
        print(f'{r/3000:.2e} solar mass')


    def get_image(self, input_image, show=True):
        
        if type(input_image) == str:
            # open image to convolve
            im = Image.open(input_image)
            im_array = np.asarray(im)
        elif type(input_image) == np.ndarray:
            im_array = input_image

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
            plt.imshow(im_array, cmap='gray')
            plt.show()

        return im_array


    def get_intensity(self, im_array, show=True):
        """_summary_

        Args:
            im_array (_type_): _description_
            show (bool, optional): _description_. Defaults to True.

        Returns:
            intensity image: _description_
        """        
        pixel_size_input_image = self.pixel_size_input_image

        ideal_pixel_size_pupil = (self.wavelength*self.telescope_focal_length_m) / (len(im_array)*self.pixel_size_input_image)
        Pixel_size_pupil_plane = ideal_pixel_size_pupil
        telescope_aperture_width_pixels = int(telescope_diameter_m / Pixel_size_pupil_plane)
        pixel_size_psf_image_plane = (wavelength*telescope_focal_length_m)/(len(im_array)*Pixel_size_pupil_plane) #term in denominator is the gridwidth in the pupil plane, image plane psf MUST NOT be cropped Verified this is correct!


        phase_screen = np.zeros((telescope_aperture_width_pixels, telescope_aperture_width_pixels), dtype=np.complex64)
        complex_amplitude = utils.quick_complex_pupil(phase_screen, array_to_propgate_size=len(im_array[0]))
        intensity_image = utils.Focus_beam(complex_amplitude)

        
        x_psf_samples = np.linspace(-pixel_size_psf_image_plane*len(intensity_image)/2, pixel_size_psf_image_plane*len(intensity_image)/2, len(intensity_image))
        y_psf_samples = np.linspace(-pixel_size_psf_image_plane*len(intensity_image)/2, pixel_size_psf_image_plane*len(intensity_image)/2, len(intensity_image))

        f = interpolate.interp2d(x_psf_samples, y_psf_samples, intensity_image, kind='cubic')
        
        x_input_image = np.linspace(-pixel_size_input_image*len(im_array)/2, pixel_size_input_image*len(im_array)/2, len(im_array))
        y_input_image = np.linspace(-pixel_size_input_image*len(im_array)/2, pixel_size_input_image*len(im_array)/2, len(im_array))


        resampled_psf = f(x_input_image, y_input_image)
        intensity_image = resampled_psf
        
        if show:
            plt.imshow(intensity_image)
            plt.show()

        return intensity_image

    def get_convolved_image(self, im_array, intensity_image, show=True):
        """_summary_

        Args:
            im_array (np.array): _description_
            intensity_image (np.array): _description_
            show (bool, optional): _description_. Defaults to True.

        Returns:
            np.array: image after convolving
        """        
        convolved_array_shape = np.shape(signal.convolve(im_array, intensity_image*(1/np.max(intensity_image)))) #this line carries out a test convolution to get the shape of the convolved arrays for the variable convolved_array

        convolved_array = np.zeros((convolved_array_shape[0],convolved_array_shape[1])) 
        convolved_array = signal.convolve(im_array, intensity_image*(1/np.max(intensity_image)))
        convolved_array = np.uint8((convolved_array)*(254/np.max(convolved_array)))
        
        # convolved_array.sum() / im_array.sum()

        if show:
            plt.imshow(convolved_array)
            plt.show()
        
        return convolved_array
    
    def generate_image(self, convolved_array, out_dir, show=True):
        """_summary_

        Args:
            convolved_array (np.array): images after convolving.
            out_dir (str): _description_
            show (bool, optional): _description_. Defaults to True.

        Returns:
            np.array: final output image
        """        
        CCD_pixel_size = self.CCD_pixel_size
        CCD_pixel_count = self.CCD_pixel_count
        pixel_size_input_image = self.pixel_size_input_image
        
        x_psf_samples = np.linspace(-pixel_size_input_image*len(convolved_array)/2, pixel_size_input_image*len(convolved_array)/2, len(convolved_array))
        y_psf_samples = np.linspace(-pixel_size_input_image*len(convolved_array)/2, pixel_size_input_image*len(convolved_array)/2, len(convolved_array))

        x_CCD = np.linspace(-CCD_pixel_size*CCD_pixel_count/2, CCD_pixel_size*CCD_pixel_count/2, CCD_pixel_count)
        y_CCD = np.linspace(-CCD_pixel_size*CCD_pixel_count/2, CCD_pixel_size*CCD_pixel_count/2, CCD_pixel_count)

        output_image = np.zeros((CCD_pixel_count,CCD_pixel_count))

        f = interpolate.interp2d(x_psf_samples, y_psf_samples, convolved_array, kind='cubic')
        output_image = f(x_CCD, y_CCD)

        output_image = np.uint8((output_image)*(255/np.max(output_image)))

        imageio.imwrite(out_dir, output_image)
        if show:
            plt.imshow(output_image)
            plt.show()
        
        return output_image




if __name__ == '__main__':
    # physical parameters
    input_image = r"./stars/BHs.png"
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
    show = True
    
    
    img = Image.open(input_image)
    im_array = np.asarray(img)
    
    telescope_simulator = TelescopeSimulator(im_array, telescope_diameter_m,telescope_focal_length_m,
        seeing_arcsec_500nm, zenith_angle_deg, atmosphere, wavelength, CCD_pixel_size,
        CCD_pixel_count, Num_psfs_to_gen, pixels_per_ro, show
    )
    
    # image_arr = telescope_simulator.get_image(input_image=input_image)

    telescope_simulator.generate_image(r'Telescope_Simulator-master/output_images/psf.png', show=True)
