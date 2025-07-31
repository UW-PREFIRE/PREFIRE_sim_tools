"""
utility functions to facilitate use of profile training databases
"""

import warnings
import os.path

import numpy as np
from scipy.io import matlab

def read_SeeBor5_clear(filename, return_raw=False,
                       tracegas_file=None):
    """
    Reads the SeeBor 5.X binary database into a python dictionary, containing
    numpy ndarray for each field. This works for the clear profiles.

    Note
    ----
    The directory containing the filename is searched for the trace gas
    file (which is assumed to be named "raqms_tracegases_clear_15704.mat").
    If this is found, then the additional trace gas fields will be loaded:
    ch4, co, n2o, o3, so2.

    All returned data listed below are keys in a single returned dictionary
    (unless return_raw was True). Each dictionary value is a numpy ndarray.
    The shapes are: nlvl = 101; nemis_wavenum = 10, nprofile = the number
    of profiles in the datafile.


    Parameters
    ----------
    
    filename : str
        path to SeeBor data file

    return_raw : bool
        optional keyword argument. If set to True, then instead of a python 
        dictionary, the raw numpy structured dtype is returned.
        This is mainly useful for debugging.

    Returns
    -------

    dat : dict
        data dictionary containing the following fields:
    temp
        temperature profiles [K], shaped (nprofile, nlvl)
    h2o_mr
        water vapor profiles in mixing ratio (kg/kg)
    o3_ppm
        ozone profiles in [ppm]
    ch4, co, n2o, o3, so2
        if the trace gas file was located, then additional, the following
        fields should be present (note there is an additional o3 profile;
        this is derived from RAQMS, I believe the other - o3_ppm - is 
        derived from ozone sondes). All profiles are in units of [ppm].
    lat
        degrees latitude, shaped (nprofile,)
    lon
        degrees longitude
    p_surf
       surface pressure in hPa
    t_skin
        surface skin temperature, [K]
    wspeed
        wind speed (used in seawater emissivity), [m/s]
    tpw
        total precipitable water vapor [cm]
    igbp
        the IGBP surface classification code (1-18)
    elevation
        surface elevation [m]
    land_flag
        1 = land, 0 = ocean
    year, month, day, hour
        time of observation
    profile_type
        code denoting the profile data source:
        1 = NOAA-88b
        2 = TIGR-3
        3 = Radiosondes
        4 = Ozonesondes
        5 = ECMWF
    emis_wn
        wavenumbers for the emissivity values [1/cm], shaped
        (nprofiles, nemis_wavenum)
    emis
        emissivity values

    """

    nlvl = 101
    nemis_wavenum = 10

    _rec_dtype = np.dtype([
        ('temp', np.float32, nlvl), 
        ('h2o_mr', np.float32, nlvl), 
        ('o3_ppm', np.float32, nlvl), 
        ('lat', np.float32), 
        ('lon', np.float32), 
        ('p_surf', np.float32), 
        ('t_skin', np.float32), 
        ('wspeed', np.float32), 
        ('tpw', np.float32), 
        ('igbp', np.float32), 
        ('elevation', np.float32), 
        ('land_flag', np.float32), 
        ('year', np.float32), 
        ('month', np.float32), 
        ('day', np.float32), 
        ('hour', np.float32), 
        ('profile_type', np.float32), 
        ('emis_wn', np.float32, nemis_wavenum),
        ('emis', np.float32, nemis_wavenum),
        ('_dummyvalue', np.float32), 
    ])

    x = np.fromfile(filename, dtype=_rec_dtype)

    if return_raw:
        return x

    d = {}
    copy_vars = ('temp', 'h2o_mr', 'o3_ppm', 'lat', 'lon',
                 'p_surf', 't_skin', 'wspeed', 'tpw',
                 'emis_wn', 'emis')

    int_vars = ('igbp', 'land_flag', 'year', 'month', 'day', 'hour',
                'profile_type')

    # use copy here, so that there is no view into the original
    # loaded data array (and then it should get garbage collected when
    # we are done)
    for v in copy_vars:
        d[v] = x[v].copy()
    for v in int_vars:
        d[v] = x[v].astype(np.int16)

    # look for trace gas file.
    tracegas_file = os.path.join(
        os.path.split(filename)[0], 'raqms_tracegases_clear_15704.mat')
    try:
        m = matlab.loadmat(tracegas_file)
        if d['temp'].shape[0] != m['ch4'].shape[0]:
            warnings.warn('trace gas file was located, but has a different '+
                          'number of profiles than the profile database')
        else:
            gas_list = ('ch4', 'co', 'n2o', 'o3', 'so2')
            for v in gas_list:
                d[v] = m[v]

    except FileNotFoundError:
        warnings.warn('could not locate the trace gas file')
        print(tracegas_file)

    return d


def read_SeeBor5_cloudy(filename, return_raw=False):
    """
    Reads the SeeBor 5.X binary database into a python dictionary, containing
    numpy ndarray for each field. This is intended for the cloudy profiles.

    Note
    ----

    See the clear sky reader (read_SeeBor5_clear) for addition descriptions.
    The returned data, and trace gas data are handled as in the clear data
    reader.

    Parameters
    ----------
        
    filename : str
        path to SeeBor data file

    return_raw : bool
        optional keyword argument. If set to True, then instead of a python 
        dictionary, the raw numpy structured dtype is returned.
        This is mainly useful for debugging.

    Returns
    -------

    dat : dict
        data dictionary containing the following fields (in addition to the
        fields returned in the clear sky data reader):
    cloud_top_pres
        cloud top pressure [hPa]
    cloud_phase
        1 = water, 2 = ice
    cloud_tau
        cloud optical thickness
    cloud_De
        cloud effective diameter (De) [micron]
    
    """

    nlvl = 101
    nemis_wavenum = 10

    _rec_dtype = np.dtype([
        ('temp', np.float32, nlvl), 
        ('h2o_mr', np.float32, nlvl), 
        ('o3_ppm', np.float32, nlvl), 
        ('lat', np.float32), 
        ('lon', np.float32), 
        ('p_surf', np.float32), 
        ('t_skin', np.float32), 
        ('wspeed', np.float32), 
        ('tpw', np.float32), 
        ('igbp', np.float32), 
        ('elevation', np.float32), 
        ('land_flag', np.float32), 
        ('year', np.int32), 
        ('month', np.int32), 
        ('day', np.float32), 
        ('hour', np.float32), 
        ('profile_type', np.float32), 
        ('emis_wn', np.float32, nemis_wavenum),
        ('emis', np.float32, nemis_wavenum),
        ('surf_air_temp', np.float32),
        ('surface_emis_coef', np.float32, 15),
        ('cloud_top_pres', np.float32),
        ('cloud_phase', np.float32),
        ('cloud_tau', np.float32),
        ('cloud_De', np.float32),
    ])

    x = np.fromfile(filename, dtype=_rec_dtype)

    if return_raw:
        return x

    d = {}
    copy_vars = ('temp', 'h2o_mr', 'o3_ppm', 'lat', 'lon',
                 'p_surf', 't_skin', 'wspeed', 'tpw',
                 'emis_wn', 'emis',
                 'cloud_top_pres', 'cloud_tau', 'cloud_De')

    int_vars = ('igbp', 'land_flag', 'year', 'month', 'day', 'hour',
                'profile_type', 'cloud_phase')

    # use copy here, so that there is no view into the original
    # loaded data array (and then it should get garbage collected when
    # we are done)
    for v in copy_vars:
        d[v] = x[v].copy()
    for v in int_vars:
        d[v] = x[v].astype(np.int16)

    # look for trace gas file.
    tracegas_file = os.path.join(
        os.path.split(filename)[1], 'raqms_tracegases_clear_21069.mat')
    try:
        m = matlab.loadmat(tracegas_file)
        if d['temp'].shape[0] != m['ch4'].shape[0]:
            warnings.warn('trace gas file was located, but has a different '+
                          'number of profiles than the profile database')
        else:
            gas_list = ('ch4', 'co', 'n2o', 'o3', 'so2')
            for v in gas_list:
                d[v] = m[v]

    except FileNotFoundError:
        warnings.warn('could not locate the trace gas file')
        pass

    return d
