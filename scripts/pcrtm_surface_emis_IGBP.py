import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import os.path
import pyPCRTM
import copy
import pdb


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

    if pcrtm_emis_wn is None:
        F = pyPCRTM.PCRTM()
        F.init(2, output_ch_flag=True)

        pcrtm_emis_wn = copy.deepcopy(F.monofreq)


    pcrtm_emis = np.interp(pcrtm_emis_wn,in_wn,in_emis)

    return pcrtm_emis,pcrtm_emis_wn


def mix_emis_pcrtm_all(stype1,stype2,ptype1,mix_pcrtm_emis_wn=None):
    """
    Find an array of emissivity values by combining two emissivity types.
    The emissivity is from Xianlei Huang's emissivity data.
    The output is using the correct wavenumbers to use as input for PCRTM emissivity.

    The surface types are as follows:
    1	        Grass
    2	        Dry grass
    3	        Decidous
    4	        Confier
    5	        Pure water
    6	        Fine snow
    7	        Medium snow
    8	        Coarse snow
    9	        Ice
    10 (10.6)   50%  silt(re=35mm)  + 50%  planar (Desert)
    11          Combination (45% desert + 55% grass)(base)  FIR_11types_53deg_2.ncncdump -h surface_emissivity_for 

 

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
          array of PCRTM emissivity wavenumbers [cm^-1]
    """

    ptype2 = 100.0 - ptype1

    #read in Xianlei's emissivity data
    Xfile = '/data/users/mmm/FIR_emissivity_data/surface_emissivity_for_11types_53deg_2.nc'

    nc = netCDF4.Dataset(Xfile,'r')

    #wavenumber (cm^-1)
    x_wn = nc['wn'][:].astype(np.float32)
    #grass
    x_emis_grass = nc['emis1'][:].astype(np.float32)
    #dry grass
    x_emis_dgrass = nc['emis2'][:].astype(np.float32)
    #Decidous
    x_emis_deci = nc['emis3'][:].astype(np.float32)
    #Conifer
    x_emis_con = nc['emis4'][:].astype(np.float32)
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
    #desert
    x_emis_des = nc['emis10'][5,:].astype(np.float32)
    #combo desert and grass
    x_emis_combo = nc['emis11'][:].astype(np.float32)
    

    nc.close()

    x_wl = 10000.0/x_wn

    if stype1 == 1:
        emis1 = copy.deepcopy(x_emis_grass)
    if stype1 == 2:
        emis1 = copy.deepcopy(x_emis_dgrass)
    if stype1 == 3:
        emis1 = copy.deepcopy(x_emis_deci)
    if stype1 == 4:
        emis1 = copy.deepcopy(x_emis_con)
    if stype1 == 5:
        emis1 = copy.deepcopy(x_emis_pw)
    if stype1 == 6:
        emis1 = copy.deepcopy(x_emis_fs)
    if stype1 == 7:
        emis1 = copy.deepcopy(x_emis_ms)
    if stype1 == 8:
        emis1 = copy.deepcopy(x_emis_cs)
    if stype1 == 9:
        emis1 = copy.deepcopy(x_emis_ice)
    if stype1 == 10:
        emis1 = copy.deepcopy(x_emis_des)
    if stype1 == 11:
        emis1 = copy.deepcopy(x_emis_combo)

    if stype2 == 1:
        emis2 = copy.deepcopy(x_emis_grass)
    if stype2 == 2:
        emis2 = copy.deepcopy(x_emis_dgrass)
    if stype2 == 3:
        emis2 = copy.deepcopy(x_emis_deci)
    if stype2 == 4:
        emis2 = copy.deepcopy(x_emis_con)
    if stype2 == 5:
        emis2 = copy.deepcopy(x_emis_pw)
    if stype2 == 6:
        emis2 = copy.deepcopy(x_emis_fs)
    if stype2 == 7:
        emis2 = copy.deepcopy(x_emis_ms)
    if stype2 == 8:
        emis2 = copy.deepcopy(x_emis_cs)
    if stype2 == 9:
        emis2 = copy.deepcopy(x_emis_ice)
    if stype2 == 10:
        emis2 = copy.deepcopy(x_emis_des)
    if stype2 == 11:
        emis2 = copy.deepcopy(x_emis_combo)

    mix_emis = emis1*(ptype1/100.0) + emis2*(ptype2/100.0)

    mix_emis_pcrtm,mix_wn_pcrtm = get_interp_emis_pcrtm(mix_emis,x_wn,pcrtm_emis_wn = mix_pcrtm_emis_wn)

    return mix_emis_pcrtm,mix_wn_pcrtm

#to test out the functions

#stype1 = 3
#stype2 = 4
#ptype1 = 50.0

#import pcrtm_surface_emis_IGBP

#out1,out2 = pcrtm_surface_emis_IGBP.mix_emis_pcrtm_all(stype1,stype2,ptype1)

def convert_IGBP_to_pcrtmemis(stype,pcrtm_emis_wn=None):
    """
    This converts a value from International Geosphere-Biosphere Programme 
    (IGBP), see table 12 from MCD12_User_Guide_V6.pdf 
    (https://modis.gsfc.nasa.gov/data/dataprod/mod12.php), 
    to a surface emissivity spectrum from reference = 
    "Huang et al.,An observationally based global 
    band-by-band surface emissivity dataset for climate and weather simulations,
    accepted by JAS, 2016." 
    The values below are using the best judgement of N. (3/1/2022)
    
    Matching IGBP [first] to XH database (second)
    Water bodies = [0] =(5)
    Evergreen Needle Forests = [1] = (4)
    Evergreen Broadleaf Forests = [2] = (4)
    Deciduous Needleleaf Forests  = [3] = (3)
    Deciduous Broadleaf Forests = [4] = (3)
    Mixed Forests = [5] = (50% 3 and 50% 4)
    Closed Shrublands = [6] = (3)
    Open Shrublands = [7] = (50% 1 and 50% 3)
    Woody Savannas = [8] = (50% 2 and 50% 3)
    Savannas = [9] = (2)
    Grasslands = [10] = (1)
    Permanent Wetlands = [11] = (50% 5 and 50% 3)
    Croplands = [12] = (50% 1 and 50% 2)
    Urban and Built-up Lands = [13] = (11)
    Cropland/Natural Vegetation Mosaics = [14] = (50% 1 and 50% 3)
    Permanent Snow and Ice = [15] = (8)
    Barren = [16] = (10.6)
    Water bodies = [17] =(5)


    Input
    ----------
    stype   : int
          surface type of IGBP classification (1-16) see above for
             surface type descriptions
 

    Returns
    -------
    out1 : ndarray
          array of the resultant emissivities at the PCRTM emissivity wavenumbers
    out2 : ndarray
          array of PCRTM emissivity wavenumbers [cm^-1]
    """
    #pdb.set_trace()
    if pcrtm_emis_wn is None:
        F = pyPCRTM.PCRTM()
        F.init(2, output_ch_flag=True)

        pcrtm_emis_wn = copy.deepcopy(F.monofreq)


    #note the second number doesn't matter if the last number is 100%
    if stype == 0:
        out1,out2 = mix_emis_pcrtm_all(5,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 1:
        out1,out2 = mix_emis_pcrtm_all(4,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 2:
        out1,out2 = mix_emis_pcrtm_all(4,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 3:
        out1,out2 = mix_emis_pcrtm_all(3,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 4:
        out1,out2 = mix_emis_pcrtm_all(3,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 5:
        out1,out2 = mix_emis_pcrtm_all(3,4,50.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 6:
        out1,out2 = mix_emis_pcrtm_all(3,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 7:
        out1,out2 = mix_emis_pcrtm_all(1,3,50.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 8:
        out1,out2 = mix_emis_pcrtm_all(2,3,50.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 9:
        out1,out2 = mix_emis_pcrtm_all(2,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 10:
        out1,out2 = mix_emis_pcrtm_all(1,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 11:
        out1,out2 = mix_emis_pcrtm_all(5,3,50.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 12:
        out1,out2 = mix_emis_pcrtm_all(1,2,50.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 13:
        out1,out2 = mix_emis_pcrtm_all(11,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 14:
        out1,out2 = mix_emis_pcrtm_all(1,3,50.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 15:
        out1,out2 = mix_emis_pcrtm_all(8,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 16:
        out1,out2 = mix_emis_pcrtm_all(10,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)
    if stype == 17:
        out1,out2 = mix_emis_pcrtm_all(5,1,100.0,mix_pcrtm_emis_wn=pcrtm_emis_wn)




    return out1,out2


def combine_emis_pcrtm(emis1,emis2,ptype1):
    """
    Find an array of emissivity values by combining two emissivity types.
    
    Note: the 2 emissivity values must be on the same wavenumber grid

    Parameters
    ----------
    emis1   : int
          surface type #1
    emis2   : int
          surface type #2
    ptype1   : float
          percentage of emis1 to use (0-100[%])

    Returns
    -------
    mix_emis : ndarray
          array of the resultant emissivities at the PCRTM emissivity wavenumbers
   
    """

    ptype2 = 100.0 - ptype1

    if emis1.shape == emis2.shape:
        mix_emis = emis1*(ptype1/100.0) + emis2*(ptype2/100.0)
    else:
        raise ValueError()
 

    return mix_emis



