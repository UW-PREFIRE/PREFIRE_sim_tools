import netCDF4
import numpy as np
import pyPCRTM
import matplotlib.pyplot as plt 
import PREFIRE_sim_tools
import OE_prototypes
import os.path
from PREFIRE_sim_tools.datasim import file_creation
import copy
import pcrtm_surface_emis
from PREFIRE_sim_tools.utils import map_emis_TIRStoPCRTM


#set to 1 if you want to write data to file
writedat = 0

#convert to TIRS resolution and use only selected channels (mimicing surface retrieval)  
#choose 0(no) or 1(yes)?
select_chans = 1


#choose the directory to read from and write to
top_outdir = '/data/users/nnn/datasim_examples/srfv0p10p4_1B_Rad_variable_surfemis_clr/'

#choose which ANC-SimTruth data set to use 1=ocean, 2=greenland, 3=antarctia, 4=tropics
case = 1

#choose which srf file to use for wn ranges
case_srf = 1
#choose srf file to use for wn ranges
if case_srf == 0:
    srffile = '/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc'
else:
    srffile = '/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'


#choose srf file to use for wn ranges
#srffile = '/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc'
#srffile = '/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'

if case == 1:
     file_use = os.path.join(
        top_outdir,
        'PREFIRE_TEST_ANC-SimTruth_S00_R00_2016000000000_00001.nc')
     nc = netCDF4.Dataset(file_use)
if case == 2:
     file_use = os.path.join(
        top_outdir,
        'PREFIRE_TEST_ANC-SimTruth_S00_R00_2016000000000_00002.nc')
     nc = netCDF4.Dataset(file_use)
if case == 3:
     file_use = os.path.join(
        top_outdir,
        'PREFIRE_TEST_ANC-SimTruth_S00_R00_2016000000000_00003.nc')
     nc = netCDF4.Dataset(file_use)
if case == 4:
     file_use = os.path.join(
        top_outdir,
        'PREFIRE_TEST_ANC-SimTruth_S00_R00_2016000000000_00004.nc')
     nc = netCDF4.Dataset(file_use)
   

temp = nc['Anc-SimTruth']['temp_profile'][:].astype(np.float32)
surf_temp = nc['Anc-SimTruth']['surface_temp'][:].astype(np.float32)
surf_pres = nc['Anc-SimTruth']['surface_pressure'][:].astype(np.float32)
pressure = nc['Anc-SimTruth']['pressure_profile'][:].astype(np.float32)
q = nc['Anc-SimTruth']['wv_profile'][:].astype(np.float32)

surf_emis = nc['Anc-SimTruth']['surf_emis'][:].astype(np.float32)

nc.close()


insfc = {}


if select_chans == 0:
    insfc['emis'] = surf_emis

    Met_fstr = os.path.join(
        top_outdir,
        'PREFIRE_TEST_origIN-SFC_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')

if select_chans == 1:
    
    surf_emis_mapped = surf_emis*0.0 - 99.0
    at,xt,nfreq = surf_emis_mapped.shape 

    Met_fstr = os.path.join(
        top_outdir,
        'PREFIRE_TEST_IN-SFC_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')

    nc = netCDF4.Dataset(srffile,'r')

    if case_srf == 0:
         wn_min = nc['channel_wavenum1'][:].astype(np.float32)
         wn_max = nc['channel_wavenum2'][:].astype(np.float32)
    else:
         wn_min = nc['channel_wavenum1'][:,0].astype(np.float32)
         wn_max = nc['channel_wavenum2'][:,0].astype(np.float32)


    nc.close()

    #input1 (Array with shape (63, 2): wavenumber ranges for all channels)
    wn_tot = np.swapaxes(np.vstack((wn_min,wn_max)),0,1)
    ave_tirs_wn = (wn_tot[:,0]+wn_tot[:,1]) / 2.0

    #input2 (Array with shape (n,): the integer channel numbers that have retrieved emissivity) here n = 14
    #PREFIRE Channels to use starting the counting from 1
    valid_emis_in = np.array([10,12,13,14,15,16,20,21,22,23,24,25,26,27])
    valid_emis = valid_emis_in -1 #change indexing from Matlab to python convention

    #input3 (Array with shape (n,):  the retrieved emissivity for each channel)
    #from surf_emis array

    #input4 (Array with shape (m=740,): the monochromatic wavenumbers)
    #get the wavenumbers from PCRTM
    F = pyPCRTM.PCRTM()
    F.init(2, output_ch_flag=True)
    pcrtm_emis_wn = copy.deepcopy(F.monofreq)

    for i in range(at):
        for j in range(xt):
            emis_in = surf_emis[i,j,:]
            tirs_out = map_emis_TIRStoPCRTM.emis_PCRTMtoTIRS(wn_tot,emis_in,pcrtm_emis_wn)
            #use the valid channels from above in python indices
            tirs_select = tirs_out[valid_emis]
            tirs_select_wn = ave_tirs_wn[valid_emis]
    
            emis_mapped = map_emis_TIRStoPCRTM.emis_TIRStoPCRTM(wn_tot,valid_emis,tirs_select,pcrtm_emis_wn)
            surf_emis_mapped[i,j,:] = emis_mapped
            #raise ValueError()

    insfc['emis'] = surf_emis_mapped

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


Met_filename = Met_fstr.format(ymd=ymd, hms=hms, granule=gran)


if writedat ==1:
    
    dat = {'In-Sfc':insfc}
    file_creation.write_data_fromspec(dat, '/home/nnn/projects/PREFIRE_sim_tools/PREFIRE_sim_tools/datasim/file_specs.json', Met_filename)

raise ValueError()
