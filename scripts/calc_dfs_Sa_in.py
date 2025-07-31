import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import copy
import pyPCRTM
import PREFIRE_sim_tools
import OE_prototypes
import pdb
import h5py

def calc_dfs_3opt(ret_type,atm_num,temp_in,q_in,surf_temp,surf_pres,F,srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc',use_chan=np.array(range(63)),Sa_in = np.array(1),NEDR_in = np.array(1),Sa_logq=0,K_in=None,use_lev=None):
    """
    Calculates the DFS and (temp and/or moisture) A matrix for a input profile

    ret_type: int
        Is the type of retrieval T only, Q only, or Joint TQ
        0 = Temperature retreival
        1 = Moisture retrieval
        2 = Joint temperature and moisture retrieval

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

    Sa_in: array [203,203]
        optional input if you wish to set the Sa covariance matrix as input
        If none the Sa matrix will be calculated from 5 k uncertainty in Temp
        profile. and 30% uncertainly in the Q profile
        The Sa_in should have 101 values for the temp profile, 101 values for
        the Q profile in that order.
        It doesn't use the 203rd indices for the Tsurf. 

    NEDR_in: array [63]
         NEDR vales to test different noise levels on different channels

    Sa_logq: 0 or 1
         Default is 0 for a linear Sa matrix
         Set to 1 if the input Sa matrix is log space

    K_in: Set the Jacobian if you'd rather run that outside the function
         The Dimensions must match that of 'use_chan'
         Also if you set this you must also set 'use_lev'
         If None then this function will calculate the Jacobian

    use_lev: a 101 array that has the level to use set as 'True'
         Only set this if you set 'K_in'

    Returns


    A : ndarray
        the A matrix, shaped (n,n) where n is the number of non-zero SRF channels
    dfs : float
        degrees of freedom

    dfs_diag : ndarray
        the diagonal of the A martix, 
        [101 or 202,depending of if if a joint retrieval]
        NOTE: The Q values are reported first [0:101], 
              T reported second [101:202] 
        where the missing values outside of n are -9999.0

    """

    if K_in is None:

        nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')
    

        F.z = nc['z'][:,atm_num].astype(np.float32)
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
    
        max_valid_level = P_levels_orig.searchsorted(surf_pres) + 1
        K_zero_test = P_levels_orig < P_levels_orig[max_valid_level]
        Kt=K[:,K_zero_test]

        K_pcrm_all6 = copy.deepcopy(dat['krad_mol'])
        #K_h2o = K_pcrm_all6[:,K_zero_test,0]
        if Sa_logq == 1:
            K_h2o = K_pcrm_all6[:,K_zero_test,0] * q_in[K_zero_test]
        else:
            K_h2o = K_pcrm_all6[:,K_zero_test,0]

        K_both = np.hstack([K_h2o,Kt])

        #set the K matrix to use according to retrieval type
        if ret_type == 0:
            K_use = copy.deepcopy(Kt)
        if ret_type == 1:
            K_use = copy.deepcopy(K_h2o)
        if ret_type == 2:
            K_use = copy.deepcopy(K_both)
    else:
        K_use = copy.deepcopy(K_in)
        K_zero_test = copy.deepcopy(use_lev)



    w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(K_use, SRFfile=srf_file,spec_grid='wl')
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
    #n_channels_freq, n_levels = K_h2o.shape
    n_levels = np.sum(K_zero_test)

    #pdb.set_trace()

    if NEDR_in.ndim == 0:
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
    else:
        Se = np.diag(NEDR_in[yrm_test]**2)

    # if Sa_in isn't set, then do the following:
    if Sa_in.ndim == 0:
        # uncertainty for each of the 101 levels
        T_uncertainty = np.zeros(n_levels) + 5.0
        Q_uncertainty = np.zeros(n_levels) + q_in[K_zero_test]*0.3
        T_surf_uncertainty = 5.0

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

      

    #using the Sa_in provided, assuming [203,203] and no corr between T and Q
    else:
        
        nlev = 101
        
        Sa_t_all = Sa_in[0:nlev, 0:nlev]
        Sa_t = Sa_t_all[:,K_zero_test]
        Sa_t = Sa_t[K_zero_test,:]
        Sa_q_all = Sa_in[nlev:2*nlev, nlev:2*nlev]
        Sa_q =  Sa_q_all[:,K_zero_test]
        Sa_q = Sa_q[K_zero_test,:]

        fillin = Sa_t*0.0 #assuming correlation between T and Q is 0
        temp1 = np.hstack([fillin,Sa_t])
        temp2 = np.hstack([Sa_q,fillin])
        Sa_both = np.vstack([temp2,temp1])

    if ret_type == 0:
        Sa_use = copy.deepcopy(Sa_t)
    if ret_type == 1:
        Sa_use = copy.deepcopy(Sa_q)
    if ret_type == 2:
        Sa_use = copy.deepcopy(Sa_both)


    ####
    # do the DFS calc.
    A = OE_prototypes.compute_A(yc_trim, Se, Sa_use)
    xnum, ynum = A.shape
    DFS = np.trace(A)
    if ret_type <= 1:
        dfs_diag = np.zeros([101]) - 9999.0
        A_flev = np.zeros((101,101)) - 9999.0
        dfs_diag_sm = np.diag(A)
        dfs_diag[K_zero_test] = dfs_diag_sm
        A_flev[0:xnum,0:ynum] = A
    
    if ret_type == 2:
        dfs_diag1 = np.zeros([101]) - 9999.0
        dfs_diag2 = np.zeros([101]) - 9999.0
        A_flev = np.zeros((202,202)) - 9999.0
        dfs_diag_sm1 = np.diag(A[0:n_levels,0:n_levels])
        dfs_diag_sm2 = np.diag(A[n_levels:n_levels*2,n_levels:n_levels*2])
        dfs_diag1[K_zero_test] = dfs_diag_sm1
        dfs_diag2[K_zero_test] = dfs_diag_sm2
        dfs_diag = np.hstack([dfs_diag1,dfs_diag2])
        
        A_flev[0:xnum,0:ynum] = A

    return A_flev,DFS,dfs_diag
