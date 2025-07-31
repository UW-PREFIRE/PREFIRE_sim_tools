import netCDF4
import numpy as np
import pyPCRTM 
import PREFIRE_sim_tools
import os
import copy
import sys
import pdb
import datetime
#import h5py
from PREFIRE_PRD_GEN.file_creation import write_data_fromspec
from PREFIRE_PRD_GEN.file_read import load_all_vars_of_nc4group, load_all_atts_of_nc4group
from PREFIRE_sim_tools.filepaths import package_ancillary_data_dir


def create_anctirsres_L1B(ancillary_data_dir, auxmet_in_fpath,
                           anc_st_sfcemis_in_fpath, output_dir,
                           product_fullver, srf_file, provenance,
                           allsky=False,
                           polewardlat=None, northonly=False,
                           write_rad_TIRSres=True):
    """
    Function to read in an ANC-SimTruth file  write out the modeled radiances at TIRS resolution (based on the SRFs).
    
    Input:
    ancillary_data_dir: str
        Path from which to read ancillary data files from (e.g., .json files
    auxmet_in_fpath: str
        Filepath from which to read met data that will be input for PCRTM
    anc_st_sfcemis_in_fpath: str
        Filepath from which to read estimated surface emissivity data that will
         be input for PCRTM
    output_dir: str
        Directory to write output file(s) to
    product_fullver: str
        Full product versions; SimRad-FullRes version first, a '+', then the
         SimRad_TIRSRes version last (e.g., "B02_R00+S06_R00")
    provenance: str
        "Provenance" of the code being used
    srf_file: str
        Full path of file to be used
        ex. srf_file = '/data/rttools/TIRS_ancillary/PREFIRE_TIRS2_SRF_v12_2023-08-09.nc'
        
    write_rad_TIRSres: bool (default = True)
         if False then only write to full-res radiance output file
         if True then write both full-res and TIRS-res radiance outfile files
    allsky: bool (default = True)
         if False then run PCRTM, ignoring any cloud fields ("clear-sky")
         if True then run PCRTM, using all cloud fields ("all-sky"
         
    polewardlat: float or None (default = None)
         a latitude constraint for the radiance calculations -- e.g., a value of
         would calculate all scenes from -60 to -90 and from 60 to 90 degrees
         latitude.  A value of None means radiance calculations will not be
         constrained by latitude.
    northonly: bool (default = False)
         if set to True, then will only calc radiances for latitudes north of
         the polewardlat value. 

    Returns:
       None
    """

    to = datetime.datetime.now()
    
    #number of channels in SRF output
    nchan = 63

    #Choose which standard profile to use as to fill in profile data
    # use subactric winter (4) 
    atm_num = 4

    file_use = auxmet_in_fpath
    fpath_ancsfcemis = anc_st_sfcemis_in_fpath

    #figure out which srffile to use according to sensor ID
    #assuming version 12 srf for now. Can add options later for other versions?
    with netCDF4.Dataset(file_use) as dataset:
        senID = dataset.sensor_ID

    if write_rad_TIRSres:
        #if senID == 'TIRS01':
        #    srf_file = '/data/rttools/TIRS_ancillary/PREFIRE_TIRS1_SRF_v12_2023-08-09.nc'
        #elif senID == 'TIRS02':
        #    srf_file = '/data/rttools/TIRS_ancillary/PREFIRE_TIRS2_SRF_v12_2023-08-09.nc'

        #load SRF data
        SRFdata_in = PREFIRE_sim_tools.TIRS_srf.load_SRFdata(SRFfile=srf_file,
                                                             spec_grid='wl')

        with netCDF4.Dataset(srf_file) as dataset:
            inst_mod_ver = dataset.instrument_model_version
            srf_src_file = dataset.SRF_source_file
            det_bflags = dataset['detector_bitflags'][:].astype(np.uint16)

    nc = netCDF4.Dataset(file_use)

    temp = nc['Aux-Met']['temp_profile'][:].astype(np.float32)
    #surf_temp = nc['Aux-Met']['surface_temp'][:].astype(np.float32)
    surf_pres = nc['Aux-Met']['surface_pressure'][:].astype(np.float32)
    skin_temp = nc['Aux-Met']['skin_temp'][:].astype(np.float32)
    pressure = nc['Aux-Met']['pressure_profile'][:].astype(np.float32)
    q = nc['Aux-Met']['wv_profile'][:].astype(np.float32)
    o3 = nc['Aux-Met']['o3_profile'][:].astype(np.float32)
    co2 = nc['Aux-Met']['xco2'][:].astype(np.float32)
    ch4 = nc['Aux-Met']['xch4'][:].astype(np.float32)
    
    #cld_prof = nc['SimTruth']['cloud_flag'][:].astype(np.float32)
    #cld_od = nc['SimTruth']['cloud_od'][:].astype(np.float32)
    #cld_dp = nc['SimTruth']['cloud_dp'][:].astype(np.float32)
    #cld_de = nc['SimTruth']['cloud_de'][:].astype(np.float32)
    
    sensor_zen = nc['Geometry']['viewing_zenith_angle'][:].astype(np.float32)

    #lon = nc['Geometry']['longitude'][:].astype(np.float32)
    lat = nc['Geometry']['latitude'][:].astype(np.float32)

    #pdb.set_trace()

    nc.close()

    with netCDF4.Dataset(fpath_ancsfcemis, 'r') as nc:
        surf_emis = nc['SimTruth-SfcEmis']['surf_emis'][:].astype(np.float32)

    # load std as the met profile (effectively)
    tmp_fpath = os.path.join(package_ancillary_data_dir,
                             "Modtran_standard_profiles_PCRTM_levels.nc")
    with netCDF4.Dataset(tmp_fpath, 'r') as nc:
        n2o = nc['n2o'][:,atm_num].astype(np.float32)
        co = nc['co'][:,atm_num].astype(np.float32)
        o3_stdatm = nc['o3'][:,atm_num].astype(np.float32)

    pobs = 0.005

    F = pyPCRTM.PCRTM()
    #F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True)
    #F.init(2, output_jacob_flag=True, output_ch_flag=True)
    F.init(2, output_ch_flag=True)

    plev = F.plevels
    
    atrack,xtrack,levs = temp.shape

    missing_val = -9.999e3

    cld_dp =  np.zeros((atrack,xtrack,levs-1))+missing_val

    for i in range(100):
            cld_dp[:,:,i] = (plev[i] + plev[i+1])/2.0

    rad_out = np.zeros([atrack,xtrack,5421])+missing_val
    #keep track of TIRS scenes that aren't poleward of given value
    #can this be used for missing values as well?
    srfrad_msk = np.full((atrack,xtrack,nchan), False)
    i_good = 0
    j_good = 0

    if polewardlat is None:
        rad_for_this_scene = np.full((atrack, xtrack), True)
    elif not northonly:
        # Apply latitude constraint:
        rad_for_this_scene = (np.abs(lat[i,j]) >= polewardlat)
    else:
        # Apply latitude and north-only constraints:
        rad_for_this_scene = (lat[i,j] >= polewardlat)
          
    for i in range(atrack):
    #test to do fewer along-track values
        #if i == 10:
        #   break
            
        for j in range(xtrack):
            
            if rad_for_this_scene[i,j]:
            
                F.psfc = surf_pres[i,j]
                F.pobs = pobs
                F.sensor_zen = sensor_zen[i,j]
        
                F.emis = surf_emis[i,j,:] # input spectral emissivity
                F.tskin = skin_temp[i,j]

                F.tlev = temp[i,j,:]
                F.h2o = q[i,j,:]
            
                if np.ma.is_masked(co2[i,j]):
                    F.co2 = np.zeros(co.shape) + 400.0
                else:
                    F.co2 = np.zeros(co.shape) + co2[i,j]

                F.o3 = o3[i,j,:]
                F.n2o = n2o
                F.co = co

                if np.ma.is_masked(ch4[i,j]):
                    F.ch4 = np.zeros(co.shape) + 1.8
                else:
                    F.ch4 = np.zeros(co.shape) + ch4[i,j]
                
                if allsky:
                    F.cld = cld_prof[i,j,:]
                    F.cldDe = cld_de[i,j,:]
                    F.cldOD = cld_od[i,j,:]
                    F.cldP = cld_dp[i,j,:]
                else:
                    nlayer = 100
                    F.cld = np.zeros(nlayer, np.int32)
                    F.cldDe = np.zeros(nlayer, np.float32)
                    F.cldOD = np.zeros(nlayer, np.float32)
                    F.cldP = cld_dp[i,j,:]

                dat = F.forward_rt()
                rad_out[i,j,:] = copy.deepcopy(dat['rad'])
                wn_pcrtm = copy.deepcopy(dat['wn'])
                i_good = copy.deepcopy(i)
                j_good = copy.deepcopy(j)
            else:
                srfrad_msk[i,j,:] = True
            #pdb.set_trace()
       
    #pdb.set_trace()

    #writing out data
    dat = {}

    with netCDF4.Dataset(file_use) as nc:
        dat["Geometry"] = load_all_vars_of_nc4group("Geometry", nc)
        dat["Geometry_Group_Attributes"] = (
                                     load_all_atts_of_nc4group("Geometry", nc))

    global_atts = {}
    with netCDF4.Dataset(file_use) as dataset:
        global_atts["granule_ID"] = dataset.granule_ID
        global_atts["spacecraft_ID"] = dataset.spacecraft_ID
        global_atts["sensor_ID"] = dataset.sensor_ID
        global_atts["ctime_coverage_start_s"] = dataset.ctime_coverage_start_s
        global_atts["ctime_coverage_end_s"] = dataset.ctime_coverage_end_s
        global_atts["UTC_coverage_start"] = dataset.UTC_coverage_start
        global_atts["UTC_coverage_end"] = dataset.UTC_coverage_end
        global_atts["orbit_sim_version"] = dataset.orbit_sim_version
        global_atts["input_product_files"] = ', '.join(
                                   [os.path.basename(file_use),
                                    os.path.basename(anc_st_sfcemis_in_fpath)])
        global_atts["netCDF_lib_version"] = netCDF4.getlibversion().split()[0]
        global_atts["provenance"] = provenance

    global_atts_fullres = copy.deepcopy(global_atts)
    global_atts_TIRSres = copy.deepcopy(global_atts)

        #pdb.set_trace()

    #fullver_l = product_fullver.split('+')
    fullver_l = copy.deepcopy(product_fullver)
    
  #--- For full spectra:
  #not saving this
  #  sdat = {}
  #  sdat['radiance'] = rad_out
  #  sdat['wavenum'] = wn_pcrtm

#    dat['SimRad-FullRes'] = sdat

#    global_atts_fullres["summary"] = ("The PREFIRE ANC-SimRad-FullRes "
#             "product provides a spectrum from PCRTM at 0.5 cm^-1 resolution.")
#    global_atts_fullres["full_versionID"] = fullver_l[0]
#    global_atts_fullres["archival_versionID"] = (
#                                   fullver_l[0].split('_')[1].replace('R', ''))

#    fn_tmp = os.path.basename(file_use)
#    tokens = fn_tmp.split('_')
#    tmp_fname = "raw-"+fn_tmp.replace("ANC-SimTruth", "ANC-SimRad-FullRes")
#    ancradfull_fname = tmp_fname.replace(f"{tokens[3]}_{tokens[4]}",
#                                         fullver_l[0])
#    global_atts_fullres["file_name"] = ancradfull_fname

#    global_atts_fullres["SRF_NEdR_version"] = ' '  # Not applicable
#    now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
#    global_atts_fullres["UTC_of_file_creation"] = now_UTC_DT.strftime(
#                                                        "%Y-%m-%dT%H:%M:%S.%f")
#    dat['Global_Attributes'] = global_atts_fullres

    #pdb.set_trace()
#    product_specs_fpath = os.path.join(ancillary_data_dir,
#                                       "Anc-Rad_product_filespecs.json")
#    fullspec_filename = os.path.join(output_dir, ancradfull_fname)
#    write_data_fromspec(dat, fullspec_filename, product_specs_fpath,
#                        verbose=False)

    #pdb.set_trace()

    product_specs_fpath = os.path.join(ancillary_data_dir,
                                       "Anc-Rad_product_filespecs.json")

    if write_rad_TIRSres:
      #--- For TIRS-resolution spectra:
        #del dat['SimRad-FullRes']

        rad_trans=rad_out.transpose()
        rad_srf_orig = np.zeros([nchan,xtrack,atrack])+missing_val
        weighted_w_xt = np.zeros([xtrack,nchan])+missing_val
        idealized_w_xt = np.zeros([xtrack,nchan])+missing_val
    
        #use preloaded SRFdata
        for k in range(xtrack):
            w, wr, yc, SRFdata = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(
                           rad_trans[:,k,:], footprint=k+1, SRFdata=SRFdata_in,
                           spec_grid='wl',)
            rad_srf_orig[:,k,:] = yc
            #pdb.set_trace()
            if "SRFweighted_w" in SRFdata:
                weighted_w_xt[k,:] = SRFdata["SRFweighted_w"][:,k]
                idealized_w_xt[k,:] = SRFdata["idealized_w"][:,k]
            else:
                weighted_w_xt[k,:] = w
                idealized_w_xt[k,:] = w
    
        rad_srf = rad_srf_orig.transpose()

        #set latitudes below the cutoff value to -9999 for rad_srf
        msk_srf = srfrad_msk
        if np.sum(msk_srf) > 0:
            rad_srf[msk_srf] = missing_val

        rdat = {}
        rdat['spectral_radiance'] = rad_srf
        #rdat['spectral_radiance_unc'] = spec_rad_unc 
        rdat['wavelength'] = weighted_w_xt
        rdat['idealized_wavelength'] = idealized_w_xt
        rdat['detector_bitflags'] = det_bflags.transpose()

        dat['SimRad-TIRSRes'] = rdat

        global_atts_TIRSres["summary"] = ("The PREFIRE ANC-SimRad-TIRSRes "
                    "product provides a spectrum originating from the FullRes "
                    "data, convolved with the specified TIRS SRF.")
        #global_atts_TIRSres["full_versionID"] = fullver_l[1]
        global_atts_TIRSres["full_versionID"] = fullver_l
        #global_atts_TIRSres["archival_versionID"] = (
        #                           fullver_l[1].split('_')[1].replace('R', ''))
        global_atts_TIRSres["archival_versionID"] = (
                                   fullver_l.split('_')[1].replace('R', ''))

        fn_tmp = os.path.basename(file_use)
        tokens = fn_tmp.split('_')
        #tmp_fname = "raw-"+fn_tmp.replace("ANC-SimTruth", "ANC-SimRad-TIRSRes")
        #ancradtirs_fname = tmp_fname.replace(f"{tokens[3]}_{tokens[4]}",
        #                                    fullver_l[1])
        tmp_fname = "raw-"+fn_tmp.replace("AUX-MET", "ANC-SimRad-TIRSRes")
        ancradtirs_fname = tmp_fname.replace(f"{tokens[3]}_{tokens[4]}",
                                             fullver_l)
        global_atts_TIRSres["file_name"] = ancradtirs_fname

        global_atts_TIRSres["SRF_NEdR_version"] = inst_mod_ver
        now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
        global_atts_TIRSres["UTC_of_file_creation"] = now_UTC_DT.strftime(
                                                        "%Y-%m-%dT%H:%M:%S.%f")
        dat['Global_Attributes'] = global_atts_TIRSres

        rad_filename = os.path.join(output_dir, ancradtirs_fname)
        write_data_fromspec(dat, rad_filename, product_specs_fpath,
                            verbose=False)
        
        
    tf = datetime.datetime.now()
    time_elasp = tf -to
    print(str(time_elasp))


if __name__ == "__main__":

    if len(sys.argv)-1 == 0:  # No command-line arguments given
        ancillary_data_dir = os.environ["ANCILLARY_DATA_DIR"]
        anc_simtruth_in_fpath = os.environ["ANC_ST_FILE"]
        anc_st_sfcemis_in_fpath = os.environ["ANC_STSE_FILE"]
        output_dir = os.environ["OUTPUT_DIR"]
        product_fullver = os.environ["PRODUCT_FULLVER"]
        provenance = os.environ["PROVENANCE"]
    elif len(sys.argv)-1 == 6:  # Command-line argument(s) provided
        ancillary_data_dir = sys.argv[1]
        auxmet_in_fpath = sys.argv[2]
        anc_st_sfcemis_in_fpath = sys.argv[3]
        output_dir = sys.argv[4]
        product_fullver = sys.argv[5]
        srf_file = sys.argv[6]
        provenance = "unknown"
    else:
        print("ERROR: an improper number of command-line options was given.\n\n"
              "Usage: python orbitsim_create_AncFullspec_L1B.py  "
              "ANCILLARY_DATA_DIR  ANC_SIMTRUTH_INFILE  "
              "ANC_SIMTRUTH_SFCEMIS_INFILE  OUTPUT_DIR  PRODUCT_FULLVER")

    # not sure exactly how to pass in keywords so this will write both files and
    # all-sky unless changed (doublecheck below), so you need to change it here
    # if running as main
    create_anctirsres_L1B(ancillary_data_dir, auxmet_in_fpath,
                           anc_st_sfcemis_in_fpath, output_dir,
                           product_fullver, srf_file, provenance, allsky=False,
                           write_rad_TIRSres=True)


#python orbitsim_create_tirsRes_L1B.py /home/nnn/projects/PREFIRE_sim_tools/scripts/test_orbitsim/ /data/ops/PREFIRE-SAT2/AUX-MET-prelim/PREFIRE_SAT2_AUX-MET_P00_R00_20240703234635_00000.nc /data/users/nnn/IOC_data/SAT2_Anc-SfcEmis/raw-PREFIRE_SAT2_ANC-SfcEmis_P00_R00_20240703234635_00000.nc /data/users/nnn/IOC_data/SAT2_clear-sky/ 'P00_R00'

#python orbitsim_create_tirsRes_L1B.py /home/nnn/projects/PREFIRE_sim_tools/scripts/test_orbitsim/ /data/ops/PREFIRE-SAT2/testlocal/AUX-MET/PREFIRE_SAT2_AUX-MET_P99_R00_20240706054614_00632.nc /data/users/nnn/IOC_data/SAT2_Anc-SfcEmis_20240926/raw-PREFIRE_SAT2_ANC-SfcEmis_P99_R00_20240706054614_00632.nc /data/users/nnn/IOC_data/SAT2_clear-sky_20240926/ 'P99_R00'

#after srf_file input was added. Oct 29, 2024 Example call. 
#python orbitsim_create_tirsRes_L1B.py /home/nnn/projects/PREFIRE_sim_tools/scripts/test_orbitsim/ /data/ops/PREFIRE-SAT2/testlocal/AUX-MET/PREFIRE_SAT2_AUX-MET_P99_R00_20240706054614_00632.nc /data/users/nnn/IOC_data/SAT2_Anc-SfcEmis_20240926/raw-PREFIRE_SAT2_ANC-SfcEmis_P99_R00_20240706054614_00632.nc /data/users/nnn/IOC_data/SAT2_clear-sky_srftests/srfv12/ 'P99_R00' /data/rttools/TIRS_ancillary/PREFIRE_TIRS2_SRF_v12_2023-08-09.nc
