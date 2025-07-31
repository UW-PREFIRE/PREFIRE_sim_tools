# Convert data from X. Huang's package into netCDF.
from scipy.io.matlab import loadmat
import netCDF4
import numpy as np

import datetime
import os

def write_ncdata(nc, dat):

    nc.createDimension('levels', dat['T'].shape[0])
    nc.createDimension('profiles', dat['T'].shape[1])

    dim2D = ('levels', 'profiles')
    nc.createVariable('z', np.float, dim2D)
    nc.createVariable('pres', np.float, dim2D)
    nc.createVariable('temp', np.float, dim2D)
    nc.createVariable('h2o', np.float, dim2D)
    nc.createVariable('q', np.float, dim2D)
    nc.createVariable('co2', np.float, dim2D)
    nc.createVariable('ch4', np.float, dim2D)
    nc.createVariable('o3', np.float, dim2D)
    nc.createVariable('n2o', np.float, dim2D)
    nc.createVariable('co', np.float, dim2D)

    nc['z'][:] = dat['z'][:]
    nc['z'].setncattr('long_name', 'level altitude')
    nc['z'].setncattr('units', 'km')
    
    nc['pres'][:] = dat['Pres'][:]
    nc['pres'].setncattr('long_name', 'level pressure')
    nc['pres'].setncattr('units', 'hPa')
    
    nc['temp'][:] = dat['T'][:]
    nc['temp'].setncattr('long_name', 'level temperature')
    nc['temp'].setncattr('units', 'K')
    
    # I think these are total air mole fractions
    # (ratio of number of molecules to total number of air molecules;
    #  and not the dry air only.)
    for k in ('h2o', 'co2', 'ch4', 'o3', 'n2o', 'co'):
        nc[k][:] = dat[k][:]
        nc[k].setncattr('long_name', k+' mole fraction')
        nc[k].setncattr('units', 'ppm')

    # convert mole fraction (molecule number ratio of the species
    # relative to the number of the total air mixture) to specific
    # humidity (ratio of h2o mass to total air mass.)
    Mv = 18.0
    Md = 28.96
    nv = dat['h2o'][:]*1e-6
    q = (nv * Mv) / (Md + nv * (Mv-Md))
    nc['q'][:] = q
    nc['q'].setncattr('long_name', 'specific humidity')
    nc['q'].setncattr('units', 'g/kg')

def PCRTM_interp_data(idat):

    P101 = np.array([
        0.0050,    0.0161,    0.0384,    0.0769,    0.1370,
        0.2244,    0.3454,    0.5064,    0.7140,    0.9753,
        1.2972,    1.6872,    2.1526,    2.7009,    3.3398,
        4.0770,    4.9204,    5.8776,    6.9567,    8.1655,
        9.5119,    11.0038,   12.6492,   14.4559,   16.4318,
        18.5847,   20.9224,   23.4526,   26.1829,   29.1210,
        32.2744,   35.6505,   39.2566,   43.1001,   47.1882,
        51.5278,   56.1260,   60.9895,   66.1253,   71.5398,
        77.2396,   83.2310,   89.5204,   96.1138,   103.0172,
        110.2366,  117.7775,  125.6456,  133.8462,  142.3848,
        151.2664,  160.4959,  170.0784,  180.0183,  190.3203,
        200.9887,  212.0277,  223.4415,  235.2338,  247.4085,
        259.9691,  272.9191,  286.2617,  300.0000,  314.1369,
        328.6753,  343.6176,  358.9665,  374.7241,  390.8926,
        407.4738,  424.4698,  441.8819,  459.7118,  477.9607,
        496.6298,  515.7200,  535.2322,  555.1669,  575.5248,
        596.3062,  617.5112,  639.1398,  661.1920,  683.6673,
        706.5654,  729.8857,  753.6275,  777.7897,  802.3714,
        827.3713,  852.7880,  878.6201,  904.8659,  931.5236,
        958.5911,  986.0666, 1013.9476, 1042.2319, 1070.9170,
        1100.0000 ])

    dat = {}
    # interp must be done TOA -> surf (monotonic increasing in P)
    # MODTRAN data (idat) is surf -> TOA, so flip when reading it
    xp = np.log(idat['Pres'])[::-1]
    x = np.log(P101)
    for k in ('Pres', 'h2o', 'co2', 'ch4', 'o3', 'n2o', 'co', 'T', 'z'):
        v = idat[k]
        dat[k] = np.zeros((x.shape[0], v.shape[1]))
        for i in range(v.shape[1]):
            # now flip surf->TOA back to TOA->surf
            dat[k][:,i] = np.interp(x, xp[:,i], v[::-1,i])
    
    # note that Pres (from the above interp) winds up being slightly 
    # different than the P101 - I am not sure why, and I don't have time
    # to carefully debug it. I think it is subtleties with doing
    # the log-P interpolation back to itself.
    # for now, just overwrite the Pres with P101
    dat['Pres'][:] = P101[:, np.newaxis]
    
    return dat


def convert_modtran_file():

    mfile = '/data/users/mmm/ECMWF2PCRTM_V3.4/Modtran_standard_profiles.mat'
    dat = loadmat(mfile)
    dat['T'] = dat['T_z'][:,1:]
    dat['z'] = np.zeros_like(dat['T'])
    dat['z'][:] = dat['T_z'][:,[0]]

    outfile = '../data/Modtran_standard_profiles.nc'

    with netCDF4.Dataset(outfile, 'w') as nc:

        write_ncdata(nc, dat)
        
        nc.setncattr('source_file', mfile)
        nc.setncattr('contents', 'MODTRAN standard atmosphere profiles')
        nc.setncattr('origin', 'Direct copy of MATLAB datafile in the '+
                     'ECWMF-PCRTM package from X. Huang et al, U. Michigan')


def convert_modtran_file_toPCRTM():

    mfile = '/data/users/mmm/ECMWF2PCRTM_V3.4/Modtran_standard_profiles.mat'
    dat = loadmat(mfile)
    dat['T'] = dat['T_z'][:,1:]
    dat['z'] = np.zeros_like(dat['T'])
    dat['z'][:] = dat['T_z'][:,[0]]

    dat_intp = PCRTM_interp_data(dat)

    outfile = '../data/Modtran_standard_profiles_PCRTM_levels.nc'

    with netCDF4.Dataset(outfile, 'w') as nc:

        write_ncdata(nc, dat_intp)
        
        nc.setncattr('source_file', mfile)
        nc.setncattr('contents', 'MODTRAN standard atmosphere profiles')
        nc.setncattr('contents', 'Data is interpolated to standard '+
                     '101-pressure level used by PCRTM, '+
                     'using linear interpolation in log-P')
        nc.setncattr('origin', 'Direct copy of MATLAB datafile in the '+
                     'ECWMF-PCRTM package from X. Huang et al, U. Michigan')


if __name__ == "__main__":
    convert_modtran_file()
    convert_modtran_file_toPCRTM()
