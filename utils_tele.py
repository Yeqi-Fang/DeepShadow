import numpy as np
import MegaScreen as M
from scipy.fftpack import fft
import warnings


warnings.filterwarnings("ignore")

def Sum2d(a):
    """"Sum over the rightmost two dimensions of an array of dimension >= 2"""
    return (np.sum(np.sum(a, -1), -1))



def Circ_Aperture_Mask(propagator_size):
    """
    Returns a square 2D array with dimensions propagator_size, and a circular mask with diameter the
    length of the input array
    """

    x = np.arange(0, propagator_size)
    y = np.arange(0, propagator_size)
    arr = np.zeros((y.size, x.size))
    diam = propagator_size
    r = diam/2

    # The two lines below could be merged, but I stored the mask
    # for code clarity.
    mask = (x[np.newaxis, :]-propagator_size/2)**2 + \
        (y[:, np.newaxis]-propagator_size/2)**2 < r**2
    arr[mask] = 1.

    return arr


def quick_complex_pupil(phaseScreen, array_to_propgate_size):
    """
    Takes in the 2D phase screen, converts it to a complex amplitude and returns it
    """

    # gets the dimension of the square phase screen grid
    propagator_size = len(phaseScreen)
    telescope_wavefront = np.exp(1j*phaseScreen)
    # complex amplitude with A = 1 inside UT aperture, 0 outside. Note, if "telescope_diameter_m" in "temporalPhaseScreens" is the size of the telescope aperture "mask_size_fraction" here should be = 1
    initial_pupil = Circ_Aperture_Mask(propagator_size)*telescope_wavefront

    # DM padding
    pad_left, pad_right = int(np.floor(array_to_propgate_size/2 - initial_pupil.shape[1]/2)), int(
        np.ceil(array_to_propgate_size/2 - initial_pupil.shape[1]/2))
    pad_up, pad_down = int(np.floor(array_to_propgate_size/2 - initial_pupil.shape[0]/2)), int(
        np.ceil(array_to_propgate_size/2 - initial_pupil.shape[0]/2))

    thisPupil = np.pad(initial_pupil, pad_width=((pad_left, pad_right), (pad_up, pad_down)), mode='constant')

    return thisPupil


def Focus_beam(Collimated_Pupil, pad_width=0):
    """
    Takes the collimated pupil and transforms it to a psf via a Fourier transform 
    """

    Collimated_Pupil_padded = np.pad(Collimated_Pupil, pad_width=int(pad_width), mode='constant')

    # must be complex amplitude going in here
    fshift = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Collimated_Pupil_padded)))
    intensity_image = (np.abs(fshift))**2

    return intensity_image


def angular_to_physical_pixels(angular_pixel_size, focal_length):
    """
    Takes an angular pixel size in arcseconds/pixel ("/pixel) and converts it to meters/pixel
    or whatever the units of focal length is. 
    """
    plate_scale = 206265/focal_length
    pixel_size_input_image = angular_pixel_size*(1/plate_scale)

    return pixel_size_input_image
