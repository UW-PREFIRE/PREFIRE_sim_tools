import matplotlib as mpl
import numpy as np


def get_color_list():
    """ create list of colors for TIRS channels. each filter block
    has a different color, with darker colors in the short wavelength
    end and brighter at the long wavelength end.
    Block 1 = cyan (short wavelength MIR)
    Block 2 = blue (long wavelength MIR)
    Block 3 = green (short wavelength FIR)
    Block 4 = red   (long wavelength FIR).

    masked channels are black.

    assumes the standard channel numbering scheme, where the TIRS
    detectors are labeled 0 - 63, with channel 0 as the undispersed beam,
    and channel 1 is the first with an actual SRF (though 1, 2, 3 are invalid).
    """
    color_list = []
    for c in range(0, 4):
        color_list.append([0.0, 0.0, 0.0])

    for c in range(4, 8):
        d = c - 4
        color_list.append([0.0, 0.6+0.133*d, 0.8+0.0666*d])

    color_list.append([0.0, 0.0, 0.0])
    color_list.append([0.0, 0.0, 0.0])
    
    for c in range(10, 17):
        d = c - 10
        color_list.append([0.0, 0.0, 0.6+0.0666*d])

    color_list.append([0.0, 0.0, 0.0])
    color_list.append([0.0, 0.0, 0.0])
    
    for c in range(19, 35):
        d = c - 19
        color_list.append([0.0, 0.4+0.04*d, 0.0])

    color_list.append([0.0, 0.0, 0.0])
    color_list.append([0.0, 0.0, 0.0])
    
    for c in range(37, 64):
        d = c-37
        color_list.append([0.4+0.023*d, 0.0, 0.0])

    return color_list

def get_listed_cmap():
    """ create a colormap object for TIRS channels,
    using the ListedColormap approach."""
    cm = mpl.colors.ListedColormap(get_color_list())
    return cm


def get_cmap_segment_data():
    """
    create segment data for a LinearSegmentedColormap that will
    color groups of TIRS channels.
    This is intended to be mapped to a even division of 64 channels.

    This method has some issues because it is hard to get the edges
    and one-off errors under control, and I think should be disfavored
    for the ListedColorMap.
    """
    x = np.zeros((65,3))
    x[:,0] = np.linspace(0,1.0,65)
    x[:,0] -= 1/128
    x[0,0] = 0.0
    x[-1,0] = 1.0

    idx_mir1 = [3,  4,  5,  6,]
    idx_mir2 = [9, 10, 11, 12, 13, 14, 15,]
    idx_fir1 = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,]
    idx_fir2 = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

    idx_mir1 = np.array(idx_mir1) + 1
    idx_mir2 = np.array(idx_mir2) + 1
    idx_fir1 = np.array(idx_fir1) + 1
    idx_fir2 = np.array(idx_fir2) + 1

    r = x.copy()
    g = x.copy()
    b = x.copy()

    b[idx_mir1+1, 1] = np.linspace(0.80, 1.00, len(idx_mir1))
    b[idx_mir1,   2] = np.linspace(0.80, 1.00, len(idx_mir1))
    b[idx_mir2+1, 1] = np.linspace(0.60, 1.0, len(idx_mir2))
    b[idx_mir2,   2] = np.linspace(0.60, 1.0, len(idx_mir2))

    g[idx_mir1+1, 1] = np.linspace(0.60, 1.00, len(idx_mir1))
    g[idx_mir1,   2] = np.linspace(0.60, 1.00, len(idx_mir1))

    g[idx_fir1+1, 1] = np.linspace(0.40, 1.0, len(idx_fir1))
    g[idx_fir1,   2] = np.linspace(0.40, 1.0, len(idx_fir1))

    r[idx_fir2+1, 1] = np.linspace(0.40, 1.0, len(idx_fir2))
    r[idx_fir2,   2] = np.linspace(0.40, 1.0, len(idx_fir2))
    
    segment_data = {'red':r, 'green':g, 'blue':b}
    
    return segment_data


def get_linear_cmap():
    """ create a colormap object for TIRS channels,
    using the LinearSegmented approach.
    the listed colormap should typically be used instead of this."""
    segmentdata = get_cmap_segment_data()
    cm = mpl.colors.LinearSegmentedColormap('TIRSch', segmentdata)
    return cm
