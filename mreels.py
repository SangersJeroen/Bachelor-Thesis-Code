from ncempy import io
import numpy as np
import scipy.fftpack as sfft
from scipy.signal import convolve2d as cv2
from tqdm import tqdm

#Some physical constants:
electron_voltage = 200e3 #Later setup object defined
planck_constant = 6.626e-34
electron_mass = 9.109e-31
elementary_charge = 1.602e-19
speed_limit = 299792458

incident_beam_energy = electron_voltage
electron_wave_lambda = planck_constant * speed_limit / np.sqrt( (elementary_charge
                       * electron_voltage)**2+2*elementary_charge
                       * electron_voltage*electron_mass*speed_limit**2 )
electron_wave_k = 2*np.pi / electron_wave_lambda


def develop(frame):
    """Develops or processes an image so that it can be better processed by ...,
    Applies a hamming filter to the frame and gets convolved by two 3x3 Sobel matrices, the resulting frames get normalised by a sigmoid function centered around the mean and adds the result in a square root.

    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a given energy

    Returns
    -------
    developed_frame : ndarray
        The processed frame
    """
    edge_x = np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]
    ])

    edge_y = np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1]
    ])

    new_image = np.copy(frame)[:,:] #Copies the given frame into new_image to conserve the original

    a = np.average(new_image)
    b = np.std(new_image)

    developed_frame = np.sqrt(cv2(1/(1+np.exp(-(new_image-a)/b))*np.hamming( new_image.shape[0] ),
                        edge_x)**2 + cv2(1/(1+np.exp(-(new_image-a)/b))
                        *np.hamming( new_image.shape[0] ), edge_y)**2)

    return developed_frame


def get_drift(frame0, frame1):
    """Calculates the drift between frame0 and frame1 using fourier phase correlation.
    https://en.wikipedia.org/wiki/Phase_correlation

    Parameters
    ----------
    frame0 : ndarray
        A slice of the stack for a given energy
    frame1 : ndarray
        A slice of the stack for a given energy

    Returns
    -------
    list
        [ydrift, xdrift] pixel drift in floats calculated from frame0 to frame1
    """

    fr0_win = frame0*np.hanning( frame0.shape )
    fr1_win = frame1*np.hanning( frame1.shape )

    fr0_pws = sfft.fft2( fr0_win )
    fr1_pws = sfft.fft2( fr1_win )

    cross_pws = ( fr0_pws * fr1_pws.conj() ) / np.abs(fr0_pws * fr1_pws.conj())
    cross_cor = np.abs( sfft.ifft2( cross_pws ) )
    cross_cor = sfft.fftshift( cross_cor )

    [py, px] = np.argwhere( cross_cor == cross_cor.max())[0]
    [ydrift, xdrift] = [py - frame0.shape[1]/2, px-frame0.shape[0]/2 ]
    return [ydrift, xdrift]


def shift(frame, yshift, xshift):
    """Shift the given frame by yshift in the y-direction axis 0 assumed, shifts by xshift in the axis 1 direction.
    Function pads the original image with zeros as not to add or misplace information.

    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a given energy
    yshift : int
        Number of pixels the frame is to be shifted will be casted to int
    xshift : int
        Number of pixels the frame is to be shifted will be casted to int

    Returns
    -------
    ndarray
        The original frame shifted
    """
    new_frame_shape = ( frame.shape[0].int(abs(yshift)), frame.shape[1]+int(abs(xshift)))
    new_frame = np.zeros( new_frame_shape )
    new_frame[0:frame.shape[0], 0:frame.shape[1]] = frame

    y_corrected_frame = np.roll( new_frame, int(yshift), axis=0)
    corrected_frame = np.roll( y_corrected_frame, int(xshift), axis=1)

    return corrected_frame[0:frame.shape[0], 0:frame.shape[1]]


def get_true_centres(frame, false_centres, leeway=50):
    """Takes a list of estimated centres of bright spots in a frame and with some leeway around this false centre looks for the brightest pixel which will be designated the true centre

    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a given energy
    false_centres : list
        list of false centres in list format: [ [fy0, fx0], [fy.., fx..], [fyN, fxN] ]
    leeway : int, optional
        Amount of pixels around the false centre in which the true centre lies, by default 50

    Returns
    -------
    list
        list of true centres in zipped list [ ty0, tx0, ty.., tx.., tyN, txN ]
    """
    true_centres = np.array([], dtype=object)
    for i in false_centres:
        part_frame = frame[i[0]-leeway:i[0]+leeway, i[1]-leeway:i[1]+leeway]
        [part_true_y, part_true_x] = np.argwhere(part_frame==part_frame.max())[0]
        [true_y, true_x] = [part_true_y+(i[0]-leeway), part_true_x+(i[1]-leeway)]
        true_centres = np.append(true_centres, (true_y, true_x))
    return true_centres


def get_beam_centre(peak, antipeak):
    """Calculates the centre of the ZLP using the centres of a peak and a corresponding anti-peak

    Parameters
    ----------
    peak : iterable
        Coordinates of peak in an iterable i.e.: (y, x) or [y, x]
    antipeak : iterable
        Coordinates of peak in an iterable i.e.: (y, x) or [y, x]

    Returns
    -------
    tuple
        tuple of coordinates of centre of the zero-loss peak (y, x)
    """
    if peak[0] > antipeak[0]:
        zlp_x = antipeak[0]+(peak[0]-antipeak[0])/2
    else:
        zlp_x = peak[0]+(antipeak[0]-peak[0])/2
    if peak[1] > antipeak[1]:
        zlp_y = antipeak[1]+(peak[1]-antipeak[1])/2
    else:
        zlp_y = peak[1]+(antipeak[1]-peak[1])/2
    return (zlp_y, zlp_x)


def generate_angle_map(frame_dimensions, beam_centre, ccd_pixel_size, camera_distance):
    """Generates a map of angles between centre of zero-loss peak and pixel on ccd

    Parameters
    ----------
    frame_dimensions : tuple
        Dimensions of frame: (ysize, xsize)
    beam_centre : tuple
        Coordinates of the centre of the beam (y, x)
    ccd_pixel_size : float
        Size of the physical pixel on the ccd, only square pixels supported
    camera_distance : float
        Distance between beam_centre pixel and sample in the same units as ccd_pixel_size

    Returns
    -------
    ndarray
        Array of angles that correspond to each pixel in the frame
    """
    map_size_y = (np.arange(frame_dimensions[1]) -int(frame_dimensions[1]/2)
                  +beam_centre[1])*ccd_pixel_size
    map_size_x = (np.arange(frame_dimensions[1]) -int(frame_dimensions[0]/2)
                  +beam_centre[0])*ccd_pixel_size

    map_y, map_x = np.meshgrid(map_size_y, map_size_x)
    radius_map = np.sqrt(map_y**2 + map_x**2)
    angle_map = radius_map /2 /camera_distance

    return angle_map


def momentum_trans_map(angle_map, dE, E0):
    """Generates a map momenta transfers per pixel in the ccd

    Parameters
    ----------
    angle_map : ndarray
        Output of mreels.generate_angle_map() or an array of angles between centre of zero-loss peak and pixel on ccd
    dE : float
        Energy difference of adjacent energy slices
    E0 : float
        Energy of unscattered electron

    Returns
    -------
    ndarray
        Map of size angle_map that contains the momentum transfer per pixel

    See Also
    -------
    mreels.generate_angle_map :
        Generates a map of angles between centre of zero-loss peak and pixel on ccd
    """
    char_elec_angle = dE /2 /E0
    mom_trans_par = char_elec_angle*electron_wave_k
    mom_trans_per = angle_map*electron_wave_k
    return np.sqrt(mom_trans_par**2 + mom_trans_per**2)


def radial_integration(frame, slice_centre, r1, r0=0, ringsize=5):
    """Performs radial integration of the slice from slice_centre outwards.
    sums all values where the distance of those values is greater than r0 and inbetween r1 and r1-ringsize.

    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a specific energy
    slice_centre : tuple
        Coordinates of the centre of the slice
    r1 : int
        Outermost diameter of integration
    ringsize : int
        Thickness of integration ring
    r0 : int, optional
        Inner limit of integration ring, disregarded if r1-ringsize>r0, by default 0

    Returns
    -------
    value
        value of the sum over the integration area
    """
    offset_y = slice_centre[0]-int(frame.shape[0]/2)
    offset_x = slice_centre[1]-int(frame.shape[1]/2)
    y = np.linspace( -int(frame.shape[0]/2) -offset_y,
                     int(frame.shape[0]/2)+offset_y, frame.shape[0])
    x = np.linspace( -int(frame.shape[1]/2) -offset_y,
                     int(frame.shape[1]/2)+offset_y, frame.shape[1])
    Y, X = np.meshgrid(y, x)
    radii = np.sqrt( (X-offset_x)**2 + (Y-offset_y)**2 )

    integration_area_ring = np.where( radii < r1, np.where( (r1 - ringsize) < radii, frame, 0), 0)
    integration_area = np.where( radii > r0, integration_area_ring, 0)
    integral = np.sum(integration_area)

    return integral


def radial_integration_stack(stack, stack_centre, r1, r0=0, ringsize=5):
    """Performs radial integration on a stack from stack_centre outwards in only spatial directions.
    Sums all values where the distance of those values is greater than r0 and inbetween r1 and r1-ringsize.
    Centre is shared for the whole stack so a shift-corrected stack is advised.


    Parameters
    ----------
    stack : ndarray
        Stack containing MREELS values with axis=0 the energy axis, (E, Y, X)
    stack_centre : tuple
        Centre of the stack (y, x)
    r1 : int
        Outermost diameter of integration
    r0 : int, optional
        Inner limit of integration ring, disregarded if r1-ringsize>r0, by default 0
    ringsize : int, optional
        Thickness of integration ring, by default 5
    """
    offset_x = stack_centre[1]-stack.shape[1]/2
    offset_y = stack_centre[0]-stack.shape[0]/2
    x = np.linspace(-int(stack.shape[2]/2)-offset_x, int(stack.shape[2]/2)+offset_x, stack.shape[2])
    y = np.linspace(-int(stack.shape[1]/2)-offset_y, int(stack.shape[1]/2)+offset_y, stack.shape[1])
    Y, X = np.meshgrid(y, x)
    radii = np.sqrt( (X-offset_x)**2 + (Y-offset_y)**2 )
    radii3d = np.broadcast_to(radii, stack.shape)

    integration_area = np.where( (radii3d>(r1-ringsize)) & (radii3d<r1) & radii3d>r0, stack, 0)
    integral = np.sum( np.sum(integration_area, axis=2), axis=1)
    return(integral)


def get_qeels_data(stack, r1, peak, anti_peak, ccd_pixel_size, camera_distance,
                    dE, E0, preferred_frame=0, ringsize=5, r0=0):
    """Generates a data set containing a map of intensities per pixel and corresponding momentum and energy axis.

    Parameters
    ----------
    stack : ndarray
        Stack containing MREELS values with axis=0 the energy axis, (E, Y, X)
    r1 : int
        Outermost diameter of integration
    peak : tuple
        Coordinates of a peak (y, x)
    antipeak : tuple
        Coordinates of the antipeak corresponding to peak (y, x)
    ccd_pixel_size : float
        Size of the physical pixels in the camera, only square pixels supported
    camera_distance : float
        Distance between sample and centre pixel in the same units as ccd_pixel_size
    dE : float
        Energy step between adjacent slices.
    E0 : float
        Energy of the shot electron in keV
    r0 : int, optional
        Innermost bound of integration, by default 0
    preferred_frame : int, optional
        The frame or slice on which the peaks are to be found, by default 0
    ringsize : int, optional
        Thickness of the integration ring, by default 5

    Returns
    -------
    intensity_map : ndarray
        Array of size energy x number of integrations, each entry is the sum of all intensities within the integration area
    momentum_axis : ndarray
        Array containing the momentum corresponding to the area of integration

    See Also
    --------
    mreels.radial_integration :
       Performs radial integration of the slice from slice_centre outwards.
    mreels.radial_integration_stack :
       Performs radial integration on a stack from stack_centre outwards in only spatial directions.
    """
    false_peaks = [peak, anti_peak]
    esize = stack.shape[1]
    iterate = range( r0, r1)
    momentum_axis = np.array([])
    intensity_map = np.zeros((len(iterate), esize))

    true_peak_centres = get_true_centres(stack[preferred_frame], false_peaks)
    true_centre_peak = [true_peak_centres[0], true_peak_centres[1]]
    true_centre_anti_peak = [true_peak_centres[2], true_peak_centres[3]]
    beam_centre = get_beam_centre(true_centre_peak, true_centre_anti_peak)

    angle_map = generate_angle_map(stack[0].shape, beam_centre, ccd_pixel_size, camera_distance)
    momentum_map = momentum_trans_map( angle_map, dE, E0 )
    index = 0
    for i in tqdm(iterate):
        momentum_frame_total = radial_integration(momentum_map[preferred_frame], beam_centre,
                                                  r0, i, ringsize)
        momentum_axis = np.append(momentum_axis, momentum_frame_total)
    for j in tqdm(iterate):
        intensity = radial_integration_stack(stack, beam_centre, r0, j, ringsize)
        intensity_map[index,:] = intensity
        index += 1

    return intensity_map, momentum_axis


class MomentumResolvedDataStack:

    def __init__(self, filename: str) -> None:

        dm4_file = io.dm.fileDM(filename)

        KEY = '.ImageList.2.ImageData.Calibrations.Dimension.'
        KEY2 = '.ImageList.2.ImageTags.EFTEMSI.Acquisition.'

        self.axis_0_origin = dm4_file.allTags[KEY2+'Start Energy (eV)']
        self.axis_1_origin = dm4_file.allTags[KEY+'2.Origin']
        self.axis_2_origin = dm4_file.allTags[KEY+'1.Origin']

        self.axis_0_scale = dm4_file.allTags[KEY2+'Step Size (eV)']
        self.axis_1_scale = dm4_file.allTags[KEY+'2.Scale']
        self.axis_2_scale = dm4_file.allTags[KEY+'1.Scale']

        self.axis_0_units = dm4_file.allTags[KEY+'3.Units']
        self.axis_1_units = dm4_file.allTags[KEY+'2.Units']
        self.axis_2_units = dm4_file.allTags[KEY+'1.Units']

        self.axis_0_steps = dm4_file.allTags['.ImageList.2.ImageData.Dimensions.3']
        self.axis_1_steps = dm4_file.allTags['.ImageList.2.ImageData.Dimensions.2']
        self.axis_2_steps = dm4_file.allTags['.ImageList.2.ImageData.Dimensions.1']

        self.axis_0_end = dm4_file.allTags[KEY2+'End Energy (eV)']

        self.stack = dm4_file.getDataset(0)['data']

    def build_axes(self):
        self.axis0 = np.linspace(self.axis_0_origin,
                                 self.axis_0_end,
                                 self.axis_0_steps)
        self.axis1 = np.linspace(self.axis_1_origin,
                                 self.axis_1_scale*self.axis_1_steps,
                                 self.axis_1_steps)
        self.axis2 = np.linspace(self.axis_2_origin,
                                 self.axis_2_scale*self.axis_2_steps,
                                 self.axis_2_steps)

class ImagingSetup:
    def __init__(self, dm4_file) -> None:
        KEY0 = '.ImageList.2.ImageTags.Acquisition.Device.'
        KEY1 = '.ImageList.2.ImageTags.Microscope Info.'

        self.resolution = dm4_file.allTags[KEY0+'Active Size (pixels)']
        self.pixelsize = dm4_file.allTags[KEY0+'CCD.Pixel Size (um)']*1e-6
        self.voltage = dm4_file.allTags[KEY1+'Formatted Voltage']
        self.distance = dm4_file.allTags[KEY1+'STEM Camera Length']