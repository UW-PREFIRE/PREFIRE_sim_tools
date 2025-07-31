import netCDF4
import numpy as np
import pyPCRTM
import matplotlib.pyplot as plt 
import PREFIRE_sim_tools
import OE_prototypes
import os.path
from PREFIRE_sim_tools.datasim import file_creation
import copy

#set to 1 if you'd like to write data to file
writedat = 1

#set the SRF file to use
srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc'

#number of channels in SRF output
nchan = 63

#Choose which standard profile to use as to fill in profile data
# use subactric winter (4) 
atm_num = 4

#choose the directory to write to
top_outdir = '/data/users/nnn/datasim_examples/'

#choose which Aux-met data set to use 1=ocean, 2=greenland, 3=antarctia, 4=tropics
case = 1

#begin program

if case == 1:
    nc = netCDF4.Dataset('/data/users/nnn/datasim_examples/'+
                     'PREFIRE_TEST_AUX-MET_S00_R00_2016000000000_00001.nc','r')
if case == 2:
    nc = netCDF4.Dataset('/data/users/nnn/datasim_examples/'+
                     'PREFIRE_TEST_AUX-MET_S00_R00_2016000000000_00002.nc','r')
if case == 3:
    nc = netCDF4.Dataset('/data/users/nnn/datasim_examples/'+
                     'PREFIRE_TEST_AUX-MET_S00_R00_2016000000000_00003.nc','r')
if case == 4:
    nc = netCDF4.Dataset('/data/users/nnn/datasim_examples/'+
                     'PREFIRE_TEST_AUX-MET_S00_R00_2016000000000_00004.nc','r')


temp = nc['Aux-Met']['temp_profile'][:].astype(np.float32)
surf_temp = nc['Aux-Met']['surface_temp'][:].astype(np.float32)
surf_pres = nc['Aux-Met']['surface_pressure'][:].astype(np.float32)
pressure = nc['Aux-Met']['pressure_profile'][:].astype(np.float32)
q = nc['Aux-Met']['wv_profile'][:].astype(np.float32)
o3 = nc['Aux-Met']['o3_profile'][:].astype(np.float32)
co2 = nc['Aux-Met']['xco2'][:].astype(np.float32)
ch4 = nc['Aux-Met']['xch4'][:].astype(np.float32)

sensor_zen = nc['Geometry']['sensor_zenith'][:].astype(np.float32)

adat = {}
for varname in nc['Aux-Met'].variables:
    adat[varname] = nc['Aux-Met'][varname][:]
tdat = {}
for varname in nc['Geometry'].variables:
    tdat[varname] = nc['Geometry'][varname][:]


nc.close()

# load std as the met profile (effectively)
nc = netCDF4.Dataset('../../PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')


n2o = nc['n2o'][:,atm_num].astype(np.float32)
co = nc['co'][:,atm_num].astype(np.float32)

nc.close()



pobs = 0.005
emis = 1.0

F = pyPCRTM.PCRTM()
F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True)

atrack,xtrack,levs = temp.shape
rad_out = np.zeros([atrack,xtrack,5421]) - 9999.0

for i in range(atrack):
    print(i)
    for j in range(xtrack):
        F.psfc = surf_pres[i,j]
        F.pobs = pobs
        F.sensor_zen = sensor_zen[i,j]
        F.emis = emis + np.zeros(F.num_monofreq, np.float32)
        F.tskin = surf_temp[i,j]

        F.tlev = temp[i,j,:]
        F.h2o = q[i,j,:]
        F.co2 = np.zeros(co.shape) + co2[i,j]
        F.o3 = o3[i,j,:]
        F.n2o = n2o
        F.co = co
        F.ch4 = np.zeros(co.shape) + ch4[i,j]

        
        dat = F.forward_rt()
        rad_out[i,j,:] = dat['rad']
        

rad_trans=rad_out.transpose()
rad_srf_orig = np.zeros([nchan,xtrack,atrack]) - 9999.0

for k in range(xtrack):
    w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(rad_trans[:,k,:], SRFfile=srf_file,spec_grid='wl')
    rad_srf_orig[:,k,:] = yc
    
rad_srf = rad_srf_orig.transpose()

#flag the nan values (set to 1 for now)
#assuming rad_srf[0,0,:] is representative of all the missing channels. 
det_flag = np.zeros([xtrack,nchan])
flag = ((rad_srf[0,0,:] > 0.0)==0)
det_flag[:,flag] = 1

#read in the spectral rad uncertainly from srf_file.
srfdata = netCDF4.Dataset(srf_file,'r')
NEDR_srf = srfdata['NEDR'][:].astype(np.float32)
srfdata.close()

spec_rad_unc = np.tile(NEDR_srf,(atrack,xtrack,1))


Rad_fstr = os.path.join(
        top_outdir, 
        'PREFIRE_TEST_1B-RAD_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
ymd = '2016000' 
hms = '000000'

if case == 1:
    gran = 1
if case == 2:
    gran = 2
if case == 3:
    gran = 3
if case == 4:
    gran = 4

Rad_filename = Rad_fstr.format(ymd=ymd, hms=hms, granule=gran)



#needs to be [xtrack, spectral]
w_xt = np.tile(w,(xtrack,1))

rdat = {}
rdat['spectral_radiance'] = rad_srf
rdat['spectral_radiance_unc'] = spec_rad_unc 
rdat['wavelength'] = w_xt
rdat['detector_flag'] = det_flag


dat = {'Geometry':tdat, 'Aux-Met':adat, 'Radiance':rdat}


if writedat ==1:
    file_creation.write_data_fromspec(dat, '/home/nnn/projects/PREFIRE_sim_tools/PREFIRE_sim_tools/datasim/file_specs.json', Rad_filename)
    
