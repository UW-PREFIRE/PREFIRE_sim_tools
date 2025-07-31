import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import os.path
import pyPCRTM
import copy


def get_interp_emis_pcrtm(in_emis,in_wn,pcrtm_emis_wn=None):
    """
    Interpolate input emissivity on given wavenumber grid to PCRTM wavenumbers.

    Parameters
    ----------
    in_emis   : ndarray
          array of high sprectral resolution emissivity at wavenumbers defind by in_wns
    in_wns    : ndarray
          array of high spectral resolution wavenumbers (1/cm)

    Returns
    -------
    pcrtm_emis : ndarray
          array of emissivities at the PCRTM emissivity wavenumbers
    pcrtm_emis_wn : ndarray
          array of PCRTM emissivity wavenumbers
    """

    if pcrtm_emis_wn == None:
        F = pyPCRTM.PCRTM()
        F.init(2, output_ch_flag=True)

        pcrtm_emis_wn = copy.deepcopy(F.monofreq)

    pcrtm_emis = np.interp(pcrtm_emis_wn,in_wn,in_emis)

    return pcrtm_emis,pcrtm_emis_wn


def mix_emis_pcrtm(stype1,stype2,ptype1,mix_pcrtm_emis_wn=None):
    """
    Find an array of emissivity values by combining two emissivity types.
    The emissivity is from Xianlei Huang's emissivity data.
    The output is using the correct wavenumbers to use as input for PCRTM emissivity.

    The surface types are as follows:
    1 = pure water
    2 = fine snow
    3 = medium snow
    4 = coarse snow 
    5 = ice

    Parameters
    ----------
    stype1   : int
          surface type #1
    stype2   : int
          surface type #2
    ptype1   : float
          percentage of stype1 to use (0-100[%])

    Returns
    -------
    mix_emis_pcrtm : ndarray
          array of the resultant emissivities at the PCRTM emissivity wavenumbers
    mix_wn_pcrtm : ndarray
          array of PCRTM emissivity wavenumbers
    """

    ptype2 = 100.0 - ptype1

    #read in Xianlei's emissivity data
    Xfile = '/data/users/mmm/FIR_emissivity_data/surface_emissivity_for_11types_53deg_2.nc'

    nc = netCDF4.Dataset(Xfile,'r')

    x_wn = nc['wn'][:].astype(np.float32)
    #pure water
    x_emis_pw = nc['emis5'][:].astype(np.float32)
    #fine snow 
    x_emis_fs = nc['emis6'][:].astype(np.float32)
    #medium snow
    x_emis_ms = nc['emis7'][:].astype(np.float32)
    #coarse snow
    x_emis_cs = nc['emis8'][:].astype(np.float32)
    #ice
    x_emis_ice = nc['emis9'][:].astype(np.float32)

    nc.close()

    x_wl = 10000.0/x_wn

    if stype1 == 1:
        emis1 = copy.deepcopy(x_emis_pw)
    if stype1 == 2:
        emis1 = copy.deepcopy(x_emis_fs)
    if stype1 == 3:
        emis1 = copy.deepcopy(x_emis_ms)
    if stype1 == 4:
        emis1 = copy.deepcopy(x_emis_cs)
    if stype1 == 5:
        emis1 = copy.deepcopy(x_emis_ice)

    if stype2 == 1:
        emis2 = copy.deepcopy(x_emis_pw)
    if stype2 == 2:
        emis2 = copy.deepcopy(x_emis_fs)
    if stype2 == 3:
        emis2 = copy.deepcopy(x_emis_ms)
    if stype2 == 4:
        emis2 = copy.deepcopy(x_emis_cs)
    if stype2 == 5:
        emis2 = copy.deepcopy(x_emis_ice)

    mix_emis = emis1*(ptype1/100.0) + emis2*(ptype2/100.0)

    mix_emis_pcrtm,mix_wn_pcrtm = get_interp_emis_pcrtm(mix_emis,x_wn,pcrtm_emis_wn = mix_pcrtm_emis_wn)

    return mix_emis_pcrtm,mix_wn_pcrtm

#to test out the functions

#stype1 = 3
#stype2 = 4
#ptype1 = 50.0

#import pcrtm_surface_emis

#out1,out2 = pcrtm_surface_emis.mix_emis_pcrtm(stype1,stype2,ptype1)








