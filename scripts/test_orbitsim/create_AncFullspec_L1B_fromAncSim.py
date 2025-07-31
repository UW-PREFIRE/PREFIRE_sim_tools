import netCDF4
import numpy as np
import pyPCRTM 
import PREFIRE_sim_tools
import os.path
#from PREFIRE_sim_tools.datasim import file_creation
import copy
import sys
import pdb
import time
from datetime import datetime
from PREFIRE_product_creator import file_creation

def create_ancfullspec_L1B(dir_name,anc_simtruth_infile,ancfullspec_outfile,l1b_outfile,writeboth=1,allsky=1,polewardlat=0.0,northonly=0.0):
    """
    Function to read in an Anc-SimTruth file and then do some calculations
    to produce a new Anc-SimTruth file with spectral surface emissivity
    and cloud properties (cloud_OD, cloud_Deff, cloud_type and cloud_pres)
    included in the new Anc-SimTruth file. 
    
    Input:
    dir_name: str
        Path from which to read from and write the file to
    anc_simtruth_infile: str
        File from from which to read data from as input into PCRTM
    ancfullspec_outfile: str
        File to write FullSpectrum data to 
        
    #set this filename if you write out the L1B-like files
    l1b_outfile: str
         File to write L1B type data to
         NOTE: writeboth keyword must be set=1
         in order to write out to this filename
    writeboth: int (default = 1)
         if 0 then only write to ancsullspec_outfile
         if 1 then write to ancfullspec_outfile and l1b_outfile
    allsky: int (default = 1)
         if 0 then run PCRTM without clouds (clear-sky)
         if 1 then run PCRTM with cloud fields
         
    polewardlat: float (default = 0)
         the latitude from which to calculate the radiances
         for example 60 would be all scenes from -60 to -90 and from
         60 to 90 degrees. 
    northonly: float (default = 0)
         if set to 1 then will only do latitudes north of the polewardlat
         value, For example poleward lat = 60 would be all scenes 
         from 60 to 90 degrees. 

    Returns:
       None
    
    """

    to = datetime.now()

    #set the SRF file to use
    srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'
    
    #number of channels in SRF output
    nchan = 63

    #Choose which standard profile to use as to fill in profile data
    # use subactric winter (4) 
    atm_num = 4

    
    file_use = os.path.join(
        dir_name, 
        anc_simtruth_infile)

    rad_filename = os.path.join(
        dir_name, 
        l1b_outfile)
    fullspec_filename = os.path.join(
        dir_name, 
        ancfullspec_outfile)

    nc = netCDF4.Dataset(file_use)

    temp = nc['Anc-SimTruth']['temp_profile'][:].astype(np.float32)
    #surf_temp = nc['Anc-SimTruth']['surface_temp'][:].astype(np.float32)
    surf_pres = nc['Anc-SimTruth']['surface_pressure'][:].astype(np.float32)
    skin_temp = nc['Anc-SimTruth']['skin_temp'][:].astype(np.float32)
    pressure = nc['Anc-SimTruth']['pressure_profile'][:].astype(np.float32)
    q = nc['Anc-SimTruth']['wv_profile'][:].astype(np.float32)
    o3 = nc['Anc-SimTruth']['o3_profile'][:].astype(np.float32)
    co2 = nc['Anc-SimTruth']['xco2'][:].astype(np.float32)
    ch4 = nc['Anc-SimTruth']['xch4'][:].astype(np.float32)
    
    cld_prof = nc['Anc-SimTruth']['cloud_flag'][:].astype(np.float32)
    cld_od = nc['Anc-SimTruth']['cloud_od'][:].astype(np.float32)
    cld_dp = nc['Anc-SimTruth']['cloud_dp'][:].astype(np.float32)
    cld_de = nc['Anc-SimTruth']['cloud_de'][:].astype(np.float32)
    
    sensor_zen = nc['Geometry']['sensor_zenith'][:].astype(np.float32)
    #lon = nc['Geometry']['longitude'][:].astype(np.float32)
    lat = nc['Geometry']['latitude'][:].astype(np.float32)
    
    surf_emis = nc['Anc-SimTruth']['surf_emis'][:].astype(np.float32)

    #pdb.set_trace()

    tdat = {}
    for varname in nc['Geometry'].variables:
        tdat[varname] = nc['Geometry'][varname][:]

    nc.close()

    # load std as the met profile (effectively)
    nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')
    
    n2o = nc['n2o'][:,atm_num].astype(np.float32)
    co = nc['co'][:,atm_num].astype(np.float32)
    o3_stdatm = nc['o3'][:,atm_num].astype(np.float32)

    nc.close()

    pobs = 0.005

    F = pyPCRTM.PCRTM()
    #F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True)
    F.init(2, output_jacob_flag=True, output_ch_flag=True)

    atrack,xtrack,levs = temp.shape

    rad_out = np.zeros([atrack,xtrack,5421]) - 9999.0
    #keep track of TIRS scenes that aren't poleward of given value
    srfrad_msk = np.zeros([atrack,xtrack,nchan]) 
    i_good = 0
    j_good = 0

    for i in range(atrack):
        for j in range(xtrack):
            
            if northonly == 0:
                testlat = np.abs(lat[i,j])
            else:
                #this will only do north of this latitude
                testlat = lat[i,j]
            #if np.abs(lat[i,j]) >= polewardlat:
            if testlat >= polewardlat:
            
                F.psfc = surf_pres[i,j]
                F.pobs = pobs
                F.sensor_zen = sensor_zen[i,j]
        
                F.emis = surf_emis[i,j,:] # input spectral emissivity
                F.tskin = skin_temp[i,j]

                F.tlev = temp[i,j,:]
                F.h2o = q[i,j,:]
            
                co2_test = np.ma.is_masked(co2[i,j])
                if co2_test == False:
                    F.co2 = np.zeros(co.shape) + co2[i,j]
                else:
                    #use if co2 is missing
                    F.co2 = np.zeros(co.shape) + 400.0
                F.o3 = o3[i,j,:]
                F.n2o = n2o
                F.co = co
                ch4_test = np.ma.is_masked(ch4[i,j])
                if ch4_test == False:
                    F.ch4 = np.zeros(co.shape) + ch4[i,j]
                else:
                    #use if ch4 is missing
                    F.ch4 = np.zeros(co.shape) + 1.8

                #if sum(cld_prof[i,j,:]) == 0:
                if allsky == 0:
                    nlayer = 100
                    F.cld = np.zeros(nlayer, np.int32)
                    F.cldDe = np.zeros(nlayer, np.float32)
                    F.cldOD = np.zeros(nlayer, np.float32)
                    F.cldP = cld_dp[i,j,:]
                else:
                    F.cld = cld_prof[i,j,:]
                    F.cldDe = cld_de[i,j,:]
                    F.cldOD = cld_od[i,j,:]
                    F.cldP = cld_dp[i,j,:]
       
        
                dat = F.forward_rt()
                rad_out[i,j,:] = dat['rad']
                wn_pcrtm = dat['wn']
                i_good = copy.deepcopy(i)
                j_good = copy.deepcopy(j)
            else:
                srfrad_msk[i,j,:] = 1 
    if writeboth == 1:
        
        rad_trans=rad_out.transpose()
        rad_srf_orig = np.zeros([nchan,xtrack,atrack]) - 9999.0
    
        for k in range(xtrack):
            w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(rad_trans[:,k,:], SRFfile=srf_file,spec_grid='wl')
            rad_srf_orig[:,k,:] = yc
    
        rad_srf = rad_srf_orig.transpose()

        #set latitudes below the cutoff value to -9999 for rad_srf
        msk_srf = (srfrad_msk == 1)
        if np.sum(msk_srf) > 0:
            rad_srf[msk_srf] = -9999 
        
    
        #flag the nan values (set to 1)
        #assuming rad_srf[i_good,j_good,:] is representative of all the missing channels. 
        det_flag = np.zeros([xtrack,nchan])
        flag = ((rad_srf[i_good,j_good,:] > 0.0)==0)
        #flag = ((rad_srf[0,0,:] > 0.0)==0)
        det_flag[:,flag] = 1
        
        #read in the spectral rad uncertainly from srf_file.
        srfdata = netCDF4.Dataset(srf_file,'r')
        NEDR_srf = srfdata['NEDR'][:].astype(np.float32)
        if NEDR_srf.ndim == 1:
            NEDR_use = copy.deepcopy(NEDR_srf)
        if NEDR_srf.ndim == 2:
            #for now use the 0th index as they are all the same
            NEDR_use = NEDR_srf[:,0]
        srfdata.close()
    
        spec_rad_unc = np.tile(NEDR_use,(atrack,xtrack,1))

        #needs to be [xtrack, spectral]
        w_xt = np.tile(w,(xtrack,1))
    
        rdat = {}
        rdat['spectral_radiance'] = rad_srf
        rdat['spectral_radiance_unc'] = spec_rad_unc 
        rdat['wavelength'] = w_xt
        rdat['detector_flag'] = det_flag

    
        #for Radiance data
        dat1 = {'Geometry':tdat, 'Radiance':rdat}

    #for Anc_fullspectra
    sdat = {}
    sdat['radiance'] = rad_out
    sdat['wavenum'] = wn_pcrtm
    dat2 = {'Geometry':tdat, 'Anc-Fullspectra':sdat}

    #raise ValueError()
    file_creation.write_data_fromspec(dat2, fullspec_filename)
    #file_creation.write_data_fromspec(dat2, '/home/nnn/projects/PREFIRE_sim_tools/PREFIRE_sim_tools/datasim/file_specs.json', fullspec_filename)
    
    if writeboth == 1:
        file_creation.write_data_fromspec(dat1, rad_filename)
        #file_creation.write_data_fromspec(dat1, '/home/nnn/projects/PREFIRE_sim_tools/PREFIRE_sim_tools/datasim/file_specs.json', rad_filename)

    tf = datetime.now()
    time_elasp = tf -to
    print(str(time_elasp))

if __name__ == "__main__":
        
    dir_name = sys.argv[1]
    anc_simtruth_infile = sys.argv[2]
    ancfullspec_outfile = sys.argv[3]
    l1b_outfile = sys.argv[4]
    #writebothkey = sys.argv[5]

    #not sure exactly how to pass in keywords so this will write both files
    create_ancfullspec_L1B(dir_name,anc_simtruth_infile,ancfullspec_outfile,l1b_outfile,writeboth=1,allsky=1)
