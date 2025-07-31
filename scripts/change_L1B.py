import netCDF4
import numpy as np
import os
import copy

#This script will add random noise to the Radiances
# as specified by the NEDR file
#
#Update the filename below before running the program. Both filenames_orig and filenames
#
#The script copy the original files first before
# running this program which adds random noise to the orginal values
# i.e. 
# cp PREFIRE_TEST_1B-RAD_S00_R00_2016000000000_00001.nc PREFIRE_TEST_Radnoise_1B-RAD_S00_R00_2016000000000_00001.nc

filenames_orig= ['/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod0p8/PREFIRE_TEST_1B-RAD_S00_R00_2016000000000_00001.nc',
            '/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod0p9/PREFIRE_TEST_1B-RAD_S00_R00_2016000000000_00001.nc',
            '/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod1p0/PREFIRE_TEST_1B-RAD_S00_R00_2016000000000_00001.nc',
            '/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod0p7/PREFIRE_TEST_1B-RAD_S00_R00_2016000000000_00001.nc']

filenames= ['/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod0p8/PREFIRE_TEST_1B-RAD_Radnoise_S00_R00_2016000000000_00001.nc',
            '/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod0p9/PREFIRE_TEST_1B-RAD_Radnoise_S00_R00_2016000000000_00001.nc',
            '/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod1p0/PREFIRE_TEST_1B-RAD_Radnoise_S00_R00_2016000000000_00001.nc',
            '/data/users/nnn/datasim_examples/cloudtests/1B_Rad_variable_surfemis_ice_cod0p7/PREFIRE_TEST_1B-RAD_Radnoise_S00_R00_2016000000000_00001.nc']

for a in range(len(filenames)):
    os.system('cp '+filenames_orig[a]+' '+filenames[a])

#set the SRF file to use
#srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc'
srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'

#read in the spectral rad uncertainty from srf_file.
srfdata = netCDF4.Dataset(srf_file,'r')
NEDR = srfdata['NEDR'][:].astype(np.float32)
if NEDR.ndim == 1:
    NEDR_use = copy.deepcopy(NEDR)
if NEDR.ndim == 2:
    #for now use the 0th index as they are all the same
    NEDR_use = NEDR[:,0]
srfdata.close()

for w in range(len(filenames)):
    nc = netCDF4.Dataset(filenames_orig[w],'r+')
    print(filenames_orig[w])
    print(filenames[w])

    rad_orig = nc['Radiance']['spectral_radiance'][:]
    nc.close()

    rad_noise = np.zeros_like(rad_orig)
    at,xt,chan = rad_noise.shape
    np.random.seed(1112)
    for i in range(at):
        for j in range(xt):
            for c in range(chan):
                # can skip this if the channel is masked.
                if NEDR_use[c] is np.ma.masked:
                    continue
                rad_noise[i,j,c] = np.random.normal(0,NEDR_use[c])

    rad_with_noise = rad_orig + rad_noise

    nc2 = netCDF4.Dataset(filenames[w],'r+')
    test2 = nc2['Radiance']['spectral_radiance'][:] 
    nc2['Radiance']['spectral_radiance'][:] = rad_with_noise
    nc2.close

