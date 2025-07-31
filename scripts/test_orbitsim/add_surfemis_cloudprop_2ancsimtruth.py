import netCDF4
import numpy as np
import pyPCRTM
import datetime
import os
import copy
import pcrtm_surface_emis_IGBP
import sys
import pdb
#import h5py
from PREFIRE_PRD_GEN.file_creation import write_data_fromspec
from PREFIRE_PRD_GEN.file_read import load_all_atts_of_nc4group, load_all_vars_of_nc4group


def add_surfemis_cloudprop(ancillary_data_dir, anc_simtruth_in_fpath,
                           output_dir, product_fullver, provenance,
                           just_emis=True):
    """
    Function to read in an ANC-SimTruth file and then do some calculations
    to produce a new ANC-SimTruth file with spectral surface emissivity.
    and possibly also the cloud properties (cloud_OD, cloud_Deff, cloud_type and cloud_pres)
    which are now included in the new ANC-SimTruth file. 
    
    Input:
    ancillary_data_dir: str
        Path from which to read ancillary data files from (e.g., .json files)
    anc_simtruth_in_fpath: str
        Filepath of the input ANC-SimTruth product file
    output_dir: str
        Directory to write output file to
    product_fullver: str
        Full product version (e.g., "S06_R00")
    provenance: str
        "Provenance" of the code being used

    Keyword:
    just_emis: True or False
        If False: then set/calculate both cloud and emissivity values
        If True: then just calculate emissivity values. 
        ANC-SimTruth file should have the cloud values, so default to
         just_emis=True
    
    Returns:
       None
    
    """
    to = datetime.datetime.now()
    
    file_use = anc_simtruth_in_fpath
    
    out_fname = "raw-"+os.path.basename(anc_simtruth_in_fpath).replace(
                                        'ANC-SimTruth', 'ANC-SimTruth-SfcEmis')
    fpath_outuse = os.path.join(output_dir, out_fname)

    missing_val = -9.999e3
    
    nc = netCDF4.Dataset(file_use,'r')

    
    cldmsk = nc['SimTruth']['cloud_mask_profile_correlated'][:].astype(np.uint8)
    qi = nc['SimTruth']['qi_profile'][:].astype(np.float32)
    ql = nc['SimTruth']['ql_profile'][:].astype(np.float32)
    

    ## for the Aux-Met group
    temp = nc['Aux-Met']['temp_profile'][:].astype(np.float32)
    at,xt,lev = temp.shape
    
    #surf_temp = nc['Aux-Met']['surface_temp'][:].astype(np.float32)
    surf_temp = nc['Aux-Met']['skin_temp'][:].astype(np.float32)
    surf_pres = nc['Aux-Met']['surface_pressure'][:].astype(np.float32)
    pressure = nc['Aux-Met']['pressure_profile'][:].astype(np.float32)
    q = nc['Aux-Met']['wv_profile'][:].astype(np.float32)
    o3 = nc['Aux-Met']['o3_profile'][:].astype(np.float32)
    co2 = nc['Aux-Met']['xco2'][:].astype(np.float32)
    ch4 = nc['Aux-Met']['xch4'][:].astype(np.float32)
    
    surf_type = nc['Aux-Met']['VIIRS_surface_type'][:].astype(np.int64)
    if surf_type.ndim == 3:
        #17 is the value for water bodies, use that as the default
        #if there are any other values then find the class with the max occurances
        surftype_2D = np.zeros((at,xt))+17
        for i in range(at):
            for j in range(xt):
                if np.sum(surf_type[i,j,0:16]) > 0:
                    #add one since python indices start at zero
                    surftype_2D[i,j] = np.argmax(surf_type[i,j,0:16])+1
        #pdb.set_trace()            
        surf_type = copy.deepcopy(surftype_2D)
    
    snow_cover = nc['Aux-Met']['snow_cover'][:].astype(np.float32)
    sea_ice = nc['Aux-Met']['seaice_concentration'][:].astype(np.float32)
    merged_surf_type = nc['Aux-Met']['merged_surface_type_prelim'][:].astype(np.float32)
    
    sensor_zen = nc['Geometry']['viewing_zenith_angle'][:].astype(np.float32)
    
    adat = {}
    for varname in nc['SimTruth'].variables:
        adat[varname] = nc['SimTruth'][varname][:]
    #auxdat = {}
    #for varname in nc['Aux-Met'].variables:
    #    auxdat[varname] = nc['Aux-Met'][varname][:]
    #tdat = {}
    #for varname in nc['Geometry'].variables:
    #    tdat[varname] = nc['Geometry'][varname][:]
        
    #copy adat so we can add values later
    ancdat = {}
    if not just_emis:
        ancdat['SimTruth'] = adat
    ancdat['SimTruth-SfcEmis'] = {}

    ancdat['Geometry'] = load_all_vars_of_nc4group("Geometry", nc)
    ancdat["Geometry_Group_Attributes"] = load_all_atts_of_nc4group("Geometry",
                                                                    nc)

    nc.close()

    

    
    #create arrays to save the values used to create the resultant surface emissivity
    surface_emis = np.zeros((at,xt,740))+missing_val
    
    if not just_emis:
        #create arrays for cloud variables
        cld_temp =  np.zeros((at,xt,lev-1))+missing_val
        cld_q =  np.zeros((at,xt,lev-1))+missing_val
        cld_od =  np.zeros((at,xt,lev-1)) 
        cld_flag =  np.zeros((at,xt,lev-1))
        cld_de =  np.zeros((at,xt,lev-1)) 
        cld_dp =  np.zeros((at,xt,lev-1))+missing_val
    
    
    #get the emis wavenumbers for PCRTM
    F = pyPCRTM.PCRTM()
    F.init(2, output_ch_flag=True)
    pcrtm_emis_wnin = copy.deepcopy(F.monofreq)
    
    plev = F.plevels

    #find medium snow emissivity to use
    emissnow_med,wn_snow_med = pcrtm_surface_emis_IGBP.mix_emis_pcrtm_all(7,7,100)

    #find coarse snow emissivity to use
    emissnow_cor,wn_snow_cor = pcrtm_surface_emis_IGBP.mix_emis_pcrtm_all(8,8,100)

    #find water to use
    emis_water,wn_water = pcrtm_surface_emis_IGBP.mix_emis_pcrtm_all(5,5,100)

    if not just_emis:

        for i in range(100):
            cld_temp[:,:,i] = ((temp[:,:,i] + temp[:,:,i+1])/2.0) - 273.15
            cld_dp[:,:,i] = (plev[i] + plev[i+1])/2.0
            cld_q[:,:,i] = ((q[:,:,i] + q[:,:,i+1])/2.0)
    
        #find LWC in g/m^3
        Mdry= 28.9644 #g/mol
        Mh2o = 18.01528 #g/mol
        R = 8.3144598 #J/(mol*K)
        eps = Mh2o/Mdry
        Rdry = R/Mdry #J/(g*K)
        Rmoist = Rdry*(1+((1-eps)*q/(1000.0*eps))) #J/(g*K)
    
        Tv =  (cld_temp+273.15)*(1+((1-eps)*cld_q/(1000.0*eps))) # K
    
        LWC = cldmsk*ql*100.0*pressure/(Rmoist*temp*1000.0) #g/m^3
        IWC = cldmsk*qi*100.0*pressure/(Rmoist*temp*1000.0) #g/m^3
    
        rho_ice = 917 #kg/m^3
        rho_liq = 997 #kg/m^3
    
        
        #parameterized based on cloud temperature (S-C Ou, K-N. Liou, 1995);
        de_ice_all = 326.3 + 12.42*cld_temp + 0.197*cld_temp**2 + 0.0012*cld_temp**3
        #PCRTM max is 180 for ice effective diameter
        de_large_msk = de_ice_all > 180.0
        de_small_msk = de_ice_all < 10.0
        de_ice = copy.deepcopy(de_ice_all)
        de_ice[de_large_msk] = 180.0
        de_ice[de_small_msk] = 10.0
    
        #set effective diameter of liquid to 20 um
        de_liq = 20.0
        tot_clr = 0
        tot = 0
    #loop over the at and xt dimensions
    for i in range(at):
        for j in range(xt):
            if not just_emis:
                tot = tot+1
                #this is for the cloud properties
                if np.sum(cldmsk[i,j,:]) > 0:
                    for w in range(lev-1):
                        #find cloud properties if cldmsk =1
                        #for 101 shaped variables use the w+1 index
                        #this will shift the cloud away from the surface by a level
                        if cldmsk[i,j,w+1] == 1:
                            #pdb.set_trace()
                            #use hypsometric equation to get layer thickness
                            dz = 1000.0*Rdry*Tv[i,j,w]*np.log(pressure[w+1]/pressure[w])/9.81
                            tau_liq_all = 1000.0*3.0*LWC[i,j,w+1]*dz/(2.0*rho_liq*(de_liq/2.0))
                            tau_ice_all = 1000.0*3.0*IWC[i,j,w+1]*dz/(2.0*rho_ice*(de_ice[i,j,w]/2.0))
                        
                            #limit OD of liquid to 100 and ice to 20
                            # defined by PCRTM limits
                            if tau_liq_all > 100.0:
                                tau_liq = 100.0
                            else:
                                tau_liq = copy.deepcopy(tau_liq_all)
                            if tau_ice_all > 20.0:
                                tau_ice = 20.0
                            else:
                                tau_ice = copy.deepcopy(tau_ice_all)
                        
                        
                            #can't model mixed clouds, so. . .
                            #if liquid optical depth dominates then set liq values
                            #use the original tau (before limits) to set the phase
                            if tau_liq_all > tau_ice_all:
                                cld_od[i,j,w] =  tau_liq
                                cld_flag[i,j,w] =  2
                                cld_de[i,j,w] = de_liq

                            #if ice optical depth dominates then set ice values
                            else:
                                cld_od[i,j,w] =  tau_ice
                                cld_flag[i,j,w] =  1
                                cld_de[i,j,w] = de_ice[i,j,w] 
                        
                else:
                    tot_clr=tot_clr+1
                

            #the following is for the surface emissivity calculation for the new Aux-Met files
            #surf_type needs to come from VIIRS_surface_type(atrack, xtrack)
            #snow_cover comes from snow_cover(atrack, xtrack)
            #sea_ice comes from seaice_concentration(atrack, xtrack)
            #merged_surf_type needs to come from merged_surface_type_prelim(atrack, xtrack)
            # where: (1)open water, (2)sea ice, (3)partial sea ice, (4)permanent land ice, (5)Antarctic ice shelf,
            #        (6)snow covered land, (7)partial snow covered land, (8)snow free land" 

            #check if over land (not water) 6-8, then don't factor in sea ice
            #(6)snow covered land, (7)partial snow covered land, (8)snow free land"
            if merged_surf_type[i,j] >=6 and merged_surf_type[i,j] <=8:
                #if there is not snow will use this value
                emisland,wnland =  pcrtm_surface_emis_IGBP.convert_IGBP_to_pcrtmemis(surf_type[i,j],pcrtm_emis_wn=pcrtm_emis_wnin)
                #pdb.set_trace()
                
                #factor in snow cover if more than 5%
                if snow_cover[i,j] > 0.05 and snow_cover[i,j] < 1.05:
                    #mix land emis with snow emis, according to snow cover
                    emis_wsnow = pcrtm_surface_emis_IGBP.combine_emis_pcrtm(emissnow_med,emisland,snow_cover[i,j]*100.0)
                    surface_emis[i,j,:] = copy.deepcopy(emis_wsnow)
                    
                else:
                    surface_emis[i,j,:] = copy.deepcopy(emisland)

            #if mergred surface type 1-3 then it is water and could be over ocean
            #(1)open water, (2)sea ice, (3)partial sea ice
            if merged_surf_type[i,j] >= 1 and merged_surf_type[i,j] <=3:
                emisocean = copy.deepcopy(emis_water)

                #factor in sea ice if more than 5%
                if sea_ice[i,j] > 0.05 and sea_ice[i,j] < 1.05:
                    #mix ocean emis with snow emis, according to snow cover
                    emis_wseaice = pcrtm_surface_emis_IGBP.combine_emis_pcrtm(emissnow_med,emisocean,sea_ice[i,j]*100.0)
                    surface_emis[i,j,:] = copy.deepcopy(emis_wseaice)
               
                    
                else:
                    surface_emis[i,j,:] = copy.deepcopy(emisocean)
            
            #for permenent ice surfaces, (4)permanent land ice, (5)Antarctic ice shelf
            if merged_surf_type[i,j] >= 4 and merged_surf_type[i,j] <=5:
                surface_emis[i,j,:] = copy.deepcopy(emissnow_cor)
                
            #pdb.set_trace()
              

    ancdat['SimTruth-SfcEmis']['surf_emis'] = surface_emis
    if not just_emis:
        ancdat['SimTruth']['cloud_flag'] = cld_flag
        ancdat['SimTruth']['cloud_od'] = cld_od
        ancdat['SimTruth']['cloud_de'] = cld_de
        ancdat['SimTruth']['cloud_dp'] = cld_dp

    ancdat['SimTruth-SfcEmis']['wn'] = wn_water

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
        global_atts["input_product_files"] = os.path.basename(file_use)
        global_atts["file_name"] = out_fname
        global_atts["full_versionID"] = product_fullver
        global_atts["archival_versionID"] = (
                                product_fullver.split('_')[1].replace('R', ''))
        global_atts["netCDF_lib_version"] = netCDF4.getlibversion().split()[0]
        global_atts["provenance"] = provenance

    now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
    global_atts["UTC_of_file_creation"] = now_UTC_DT.strftime(
                                                        "%Y-%m-%dT%H:%M:%S.%f")

    ancdat['Global_Attributes'] = global_atts

    #pdb.set_trace()

    product_specs_fpath = os.path.join(ancillary_data_dir,
                                 "Anc-SimTruth-SfcEmis_product_filespecs.json")
    write_data_fromspec(ancdat, fpath_outuse, product_specs_fpath,
                        verbose=False)
    
    #pdb.set_trace()


    #write out surface emissivity.
    #K. has incorporated the code for cloud estimates into the Aux_met code
    #the cloud values here should match his values
    #later create a proper netCDF with the surf_emis data only
    ##h = h5py.File(file_outuse,'w')
    ##h['surf_emis'] = surface_emis
    ##if justemis == 0:
    ##    h['cloud_flag'] = cld_flag
    ##    h['cloud_od'] = cld_od
    ##    h['cloud_de'] = cld_de
    ##    h['cloud_dp'] = cld_dp
    ##h.close()

    tf = datetime.datetime.now()
    time_elasp = tf -to
    print(str(time_elasp))

if __name__ == "__main__":

    if len(sys.argv)-1 == 0:  # No command-line arguments given
        ancillary_data_dir = os.environ["ANCILLARY_DATA_DIR"]
        anc_simtruth_in_fpath = os.environ["ANC_ST_FILE"]
        output_dir = os.environ["OUTPUT_DIR"]
        product_fullver = os.environ["PRODUCT_FULLVER"]
        provenance = os.environ["PROVENANCE"]
    elif len(sys.argv)-1 == 4:  # Command-line argument(s) provided
        ancillary_data_dir = sys.argv[1]
        anc_simtruth_in_fpath = sys.argv[2]
        output_dir = sys.argv[3]
        product_fullver = sys.argv[4]
        provenance = "unknown"
    else:
        print("ERROR: an improper number of command-line options was given.\n\n"
              "Usage: python add_surfemis_cloudprop_2ancsimtruth.py  "
              "ANCILLARY_DATA_DIR  "
              "ANC_SIMTRUTH_INFILE  OUTPUT_DIR  PRODUCT_FULLVER")

    add_surfemis_cloudprop(ancillary_data_dir, anc_simtruth_in_fpath,
                           output_dir, product_fullver, provenance,
                           just_emis=True)
