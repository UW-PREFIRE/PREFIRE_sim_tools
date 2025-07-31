import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import copy
import pyPCRTM
import PREFIRE_sim_tools
import OE_prototypes

def calc_dfs_temp(atm_num,temp_in,q_in,surf_temp,surf_pres,F,srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc',use_chan=np.array(range(63))):
    """
    Calculates the DFS and temperature A matrix for the temp and q profiles input, using the choosen standard profile to fillin gaps. 

    atm_num : int
        Is the index of the standard profile to be used
        0 = tropic
        1 = midlat_summer
        2 = midlat_winter
        3 = subarc_summer
        4 = subarc_winter
        5 = us_standard

    temp_in   : float
       profile of temperature. Expecting 101 levels interpolated to PCRTM levels.
    q_in      : float
       profile of specific humidity. Expecting 101 levels interpolated to PCRTM levels.
    surf_temp  : float
        surface temperature
.    surf_pres  : float
        surface pressure
    F          : 
    Should be the initialized PCRTM as described below to output the Jacobians
    F = pyPCRTM.PCRTM()
    F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True,output_bt_flag=True,output_jacob_bt_flag=True)
    
    srf_file : str or None
        optional keyword to specify the stored SRF data.
        If None, then a hardcoded file is used (v0.10.4 at the moment.)

    use_chan : int (index starting at channel 1)
        Optional keyword to specify which channels to use for the DFS analysis
        If None,then use all 63 channels and mask out the sacrificed channels in the function. 
        If keyword is set, do account for the masked channels 
        (1,2,3,8,9,17,18,35,36)

    Returns


    A : ndarray
        the A matrix, shaped (n,n) where n is the number of non-zero SRF channels
    dfs : float
        degrees of freedom

    dfs_diag : ndarray
        the diagonal of the A martix, [101] where the missing values outside of n are -9999.0

    """

    nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')
    

    F.z = nc['z'][:,atm_num].astype(np.float32)
    F.tlev = nc['temp'][:,atm_num].astype(np.float32)
    # very approx conversion from ppm to q [g/kg]
    q = nc['h2o'][:,atm_num] * 0.622 * 1e-6 * 1e3
    F.h2o = q.astype(np.float32)
    F.co2 = nc['co2'][:,atm_num].astype(np.float32)
    F.o3 = nc['o3'][:,atm_num].astype(np.float32)
    F.n2o = nc['n2o'][:,atm_num].astype(np.float32)
    F.co = nc['co'][:,atm_num].astype(np.float32)
    F.ch4 = nc['ch4'][:,atm_num].astype(np.float32)
    nc.close()

    
    F.pobs = 0.005
    F.sensor_zen = 0.0
    F.emis = 1.0 + np.zeros(F.num_monofreq, np.float32)

    # the PCRTM fixed pressure levels.
    P_levels_orig = np.loadtxt('/home/nnn/projects/PREFIRE_sim_tools/data/plevs101.txt')
 

    F.tlev = copy.deepcopy(temp_in)

    F.tskin = copy.deepcopy(surf_temp)
    F.psfc = copy.deepcopy(surf_pres)

 
    F.h2o = copy.deepcopy(q_in)

    dat = F.forward_rt()
    K = copy.deepcopy(dat['krad_t'])
    #K_zero_test = np.all(K==0,axis=0)==0
    max_valid_level = P_levels_orig.searchsorted(surf_pres) + 1
    K_zero_test = P_levels_orig < P_levels_orig[max_valid_level]
    Kt=K[:,K_zero_test]


    w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(Kt, SRFfile=srf_file,spec_grid='wl')
    #mask the array to plot with all channels
    ym = np.ma.masked_invalid(yc)
    #create an array to remove the blank channels
    #if all 63 channels are selected then use this mask for invalid channels
    use_chan_shape = use_chan.shape
    if use_chan_shape[0] == 63:
        yrm_test = np.all(np.isfinite(yc),axis=1)

    else:
    #use the selected channels
        yrm_test = use_chan - 1

    yc_trim = yc[yrm_test,:]
    w_trim = w[yrm_test]

    n_channels, n_levels_both = yc_trim.shape
    n_channels_freq, n_levels = Kt.shape

    ####
    # Get Se, Sa, from assumed parameters.
    srfdata = netCDF4.Dataset(srf_file,'r')
    NEDR_srf = srfdata['NEDR'][:].astype(np.float32)
    # version 10.4 and above has 2 dimensions. 
    # make this compatible with v0.09.2 (no footprint
    # dependence,1 dim) or v0.10.4 or later (fp-depedent axis in SRF. 2 dim)
    if NEDR_srf.ndim == 1:
        Se = np.diag(NEDR_srf[yrm_test]**2)
    if NEDR_srf.ndim == 2:
        #for now use the 0th index as they are all the same
        Se = np.diag(NEDR_srf[yrm_test,0]**2)
    srfdata.close()

    # uncertainty for each of the 101 levels
    T_uncertainty = np.zeros(n_levels) + 5.0

    # T_correlation, this is the exponential scale length in [hPa], since
    # we are using pressure levels.
    T_correlation = np.linspace(10.0, 200.0, n_levels)

    P_levels = P_levels_orig[K_zero_test]
    

    Sa_t, Ra_t = OE_prototypes.level_exp_cov_matrix(P_levels, T_uncertainty, T_correlation)

    dfs_diag = np.zeros([101]) - 9999.0
    A = OE_prototypes.compute_A(yc_trim, Se, Sa_t)
    DFS = np.trace(A)
    dfs_diag_sm = np.diag(A)
    dfs_diag[K_zero_test] = dfs_diag_sm
    alt = copy.deepcopy(F.z[K_zero_test])

    return A,DFS,dfs_diag

def calc_dfs_h2o(atm_num,temp_in,q_in,surf_temp,surf_pres,F,srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc',use_chan=np.array(range(63))):
    """
    Calculates the DFS and moisture  A matrix for the temp and q profiles input, using the choosen standard profile to fillin gaps. 

    atm_num : int
        Is the index of the standard profile to be used
        0 = tropic
        1 = midlat_summer
        2 = midlat_winter
        3 = subarc_summer
        4 = subarc_winter
        5 = us_standard

    temp_in   : float
       profile of temperature. Expecting 101 levels interpolated to PCRTM levels.
    q_in      : float
       profile of specific humidity. Expecting 101 levels interpolated to PCRTM levels.
    surf_temp  : float
        surface temperature
.    surf_pres  : float
        surface pressure
    F          : 
    Should be the initialized PCRTM as described below to output the Jacobians
    F = pyPCRTM.PCRTM()
    F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True,output_bt_flag=True,output_jacob_bt_flag=True)
    
    srf_file : str or None
        optional keyword to specify the stored SRF data.
        If None, then a hardcoded file is used (v0.10.4 at the moment.)

    use_chan : int (index starting at channel 1)
        Optional keyword to specify which channels to use for the DFS analysis
        If None,then use all 63 channels and mask out the sacrificed channels in the function. 
        If keyword is set, do account for the masked channels 
        (1,2,3,8,9,17,18,35,36

    Returns


    A : ndarray
        the A matrix, shaped (n,n) where n is the number of non-zero SRF channels
    dfs : float
        degrees of freedom
    dfs_diag : ndarray
        the diagonal of the A martix, [101] where the missing values outside of n are -9999.0

    """


    nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')
    

    F.z = nc['z'][:,atm_num].astype(np.float32)
    F.tlev = nc['temp'][:,atm_num].astype(np.float32)
    # very approx conversion from ppm to q [g/kg]
    q = nc['h2o'][:,atm_num] * 0.622 * 1e-6 * 1e3
    F.h2o = q.astype(np.float32)
    F.co2 = nc['co2'][:,atm_num].astype(np.float32)
    F.o3 = nc['o3'][:,atm_num].astype(np.float32)
    F.n2o = nc['n2o'][:,atm_num].astype(np.float32)
    F.co = nc['co'][:,atm_num].astype(np.float32)
    F.ch4 = nc['ch4'][:,atm_num].astype(np.float32)
    nc.close()

    
    F.pobs = 0.005
    F.sensor_zen = 0.0
    F.emis = 1.0 + np.zeros(F.num_monofreq, np.float32)

    # the PCRTM fixed pressure levels.
    P_levels_orig = np.loadtxt('/home/nnn/projects/PREFIRE_sim_tools/data/plevs101.txt')
    


    F.tlev = copy.deepcopy(temp_in)

    F.tskin = copy.deepcopy(surf_temp)
    F.psfc = copy.deepcopy(surf_pres)

  
    F.h2o = copy.deepcopy(q_in)

    dat = F.forward_rt()
    K = copy.deepcopy(dat['krad_t'])
    #K_zero_test = np.all(K==0,axis=0)==0
    max_valid_level = P_levels_orig.searchsorted(surf_pres) + 1
    K_zero_test = P_levels_orig < P_levels_orig[max_valid_level]
    Kt=K[:,K_zero_test]

    Kt_pcrm_all6 = copy.deepcopy(dat['krad_mol'])
    Kt_h2o = Kt_pcrm_all6[:,K_zero_test,0]


    w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(Kt_h2o, SRFfile=srf_file,spec_grid='wl')
    #mask the array to plot with all channels
    ym = np.ma.masked_invalid(yc)
    #create an array to remove the blank channels
    #if all 63 channels are selected then use this mask for invalid channels
    use_chan_shape = use_chan.shape
    if use_chan_shape[0] == 63:
        yrm_test = np.all(np.isfinite(yc),axis=1)

    else:
    #use the selected channels
        yrm_test = use_chan - 1

    yc_trim = yc[yrm_test,:]
    w_trim = w[yrm_test]

    n_channels, n_levels_both = yc_trim.shape
    n_channels_freq, n_levels = Kt.shape

    ####
    # Get Se, Sa, from assumed parameters.
    srfdata = netCDF4.Dataset(srf_file,'r')
    NEDR_srf = srfdata['NEDR'][:].astype(np.float32)
    # version 10.4 and above has 2 dimensions. 
    # make this compatible with v0.09.2 (no footprint
    # dependence,1 dim) or v0.10.4 or later (fp-depedent axis in SRF. 2 dim)
    if NEDR_srf.ndim == 1:
        Se = np.diag(NEDR_srf[yrm_test]**2)
    if NEDR_srf.ndim == 2:
        #for now use the 0th index as they are all the same
        Se = np.diag(NEDR_srf[yrm_test,0]**2)
    srfdata.close()


    # uncertainty for each of the 101 levels
    Q_uncertainty = np.zeros(n_levels) + q_in[K_zero_test]*0.3

    # Q_correlation, this is the exponential scale length in [hPa], since
    # we are using pressure levels.
    Q_correlation = np.linspace(10.0, 200.0, n_levels)

    P_levels = P_levels_orig[K_zero_test]

    Sa_q, Ra_q = OE_prototypes.level_exp_cov_matrix(P_levels, Q_uncertainty, Q_correlation)

    dfs_diag = np.zeros([101]) - 9999.0
    A = OE_prototypes.compute_A(yc_trim, Se, Sa_q)
    DFS = np.trace(A)
    dfs_diag_sm = np.diag(A)
    dfs_diag[K_zero_test] = dfs_diag_sm
    alt = copy.deepcopy(F.z[K_zero_test])
    
    return A,DFS,dfs_diag


def calc_dfs_temp_h2o(atm_num,temp_in,q_in,surf_temp,surf_pres,F,srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc',use_chan=np.array(range(63))):
    """
    Calculates the DFS and (temp + moisture A matrix) for a input profile

    atm_num : int
        Is the index of the standard profile to be used
        0 = tropic
        1 = midlat_summer
        2 = midlat_winter
        3 = subarc_summer
        4 = subarc_winter
        5 = us_standard

    temp_in   : float
       profile of temperature. Expecting 101 levels interpolated to PCRTM levels.
    q_in      : float
       profile of specific humidity. Expecting 101 levels interpolated to PCRTM levels.
    surf_temp  : float
        surface temperature
.    surf_pres  : float
        surface pressure    
     F          : 
    Should be the initialized PCRTM as described below to output the Jacobians
    F = pyPCRTM.PCRTM()
    F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True,output_bt_flag=True,output_jacob_bt_flag=True)

    srf_file : str or None
        optional keyword to specify the stored SRF data.
        If None, then a hardcoded file is used (v0.10.4 at the moment.)

    use_chan : int (index starting at channel 1)
        Optional keyword to specify which channels to use for the DFS analysis
        If None,then use all 63 channels and mask out the sacrificed channels in the function. 
        If keyword is set, do account for the masked channels 
        (1,2,3,8,9,17,18,35,36)

    Returns


    A : ndarray
        the A matrix, shaped (n,n) where n is the number of non-zero SRF channels
    dfs : float
        degrees of freedom

    dfs_diag : ndarray
        the diagonal of the A martix, [202] where the missing values outside of n are -9999.0

    """

    nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')
    

    F.z = nc['z'][:,atm_num].astype(np.float32)
    F.tlev = nc['temp'][:,atm_num].astype(np.float32)
    # very approx conversion from ppm to q [g/kg]
    q = nc['h2o'][:,atm_num] * 0.622 * 1e-6 * 1e3
    F.h2o = q.astype(np.float32)
    F.co2 = nc['co2'][:,atm_num].astype(np.float32)
    F.o3 = nc['o3'][:,atm_num].astype(np.float32)
    F.n2o = nc['n2o'][:,atm_num].astype(np.float32)
    F.co = nc['co'][:,atm_num].astype(np.float32)
    F.ch4 = nc['ch4'][:,atm_num].astype(np.float32)
    nc.close()

    
    F.pobs = 0.005
    F.sensor_zen = 0.0
    F.emis = 1.0 + np.zeros(F.num_monofreq, np.float32)

    # the PCRTM fixed pressure levels.
    P_levels_orig = np.loadtxt('/home/nnn/projects/PREFIRE_sim_tools/data/plevs101.txt')


    F.tlev = copy.deepcopy(temp_in)

    F.tskin = copy.deepcopy(surf_temp)
    F.psfc = copy.deepcopy(surf_pres)

 
    F.h2o = copy.deepcopy(q_in)
 

    dat = F.forward_rt()
    K = copy.deepcopy(dat['krad_t'])
    #K_zero_test = np.all(K==0,axis=0)==0
    max_valid_level = P_levels_orig.searchsorted(surf_pres) + 1
    K_zero_test = P_levels_orig < P_levels_orig[max_valid_level]
    Kt=K[:,K_zero_test]

    Kt_pcrm_all6 = copy.deepcopy(dat['krad_mol'])
    Kt_h2o = Kt_pcrm_all6[:,K_zero_test,0]
    Kt_both = np.hstack([Kt_h2o,Kt])


    w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(Kt_both, SRFfile=srf_file,spec_grid='wl')
    #mask the array to plot with all channels
    ym = np.ma.masked_invalid(yc)
    #create an array to remove the blank channels
    #if all 63 channels are selected then use this mask for invalid channels
    use_chan_shape = use_chan.shape
    if use_chan_shape[0] == 63:
        yrm_test = np.all(np.isfinite(yc),axis=1)

    else:
    #use the selected channels
        yrm_test = use_chan - 1

    yc_trim = yc[yrm_test,:]
    w_trim = w[yrm_test]

    n_channels, n_levels_both = yc_trim.shape
    n_channels_freq, n_levels = Kt_h2o.shape

    ####
    # Get Se, Sa, from assumed parameters.
    srfdata = netCDF4.Dataset(srf_file,'r')
    NEDR_srf = srfdata['NEDR'][:].astype(np.float32)
    # version 10.4 and above has 2 dimensions. 
    # make this compatible with v0.09.2 (no footprint
    # dependence,1 dim) or v0.10.4 or later (fp-depedent axis in SRF. 2 dim)
    if NEDR_srf.ndim == 1:
        Se = np.diag(NEDR_srf[yrm_test]**2)
    if NEDR_srf.ndim == 2:
        #for now use the 0th index as they are all the same
        Se = np.diag(NEDR_srf[yrm_test,0]**2)
    srfdata.close()

    # uncertainty for each of the 101 levels
    T_uncertainty = np.zeros(n_levels) + 5.0
    Q_uncertainty = np.zeros(n_levels) + q_in[K_zero_test]*0.3

    # T_correlation, this is the exponential scale length in [hPa], since
    # we are using pressure levels.
    T_correlation = np.linspace(10.0, 200.0, n_levels)
    Q_correlation = np.linspace(10.0, 200.0, n_levels)

    P_levels = P_levels_orig[K_zero_test]

    Sa_t, Ra_t = OE_prototypes.level_exp_cov_matrix(P_levels, T_uncertainty, T_correlation)

    Sa_q, Ra_q = OE_prototypes.level_exp_cov_matrix(P_levels, Q_uncertainty, Q_correlation)


    fillin = Sa_t*0.0
    temp1 = np.hstack([fillin,Sa_t])
    temp2 = np.hstack([Sa_q,fillin])
    Sa_both = np.vstack([temp2,temp1])


    ####
    # do the DFS calc.
    dfs_diag1 = np.zeros([101]) - 9999.0
    dfs_diag2 = np.zeros([101]) - 9999.0
    A = OE_prototypes.compute_A(yc_trim, Se, Sa_both)
    DFS = np.trace(A)
    dfs_diag_sm1 = np.diag(A[0:n_levels,0:n_levels])
    dfs_diag_sm2 = np.diag(A[n_levels:n_levels*2,n_levels:n_levels*2])
    dfs_diag1[K_zero_test] = dfs_diag_sm1
    dfs_diag2[K_zero_test] = dfs_diag_sm2
    dfs_diag = np.hstack([dfs_diag1,dfs_diag2])
    return A,DFS,dfs_diag
