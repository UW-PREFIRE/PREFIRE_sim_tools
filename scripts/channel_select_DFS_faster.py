import netCDF4
import numpy as np
import PREFIRE_sim_tools
from calc_dfs_auxtypein import calc_dfs_temp,calc_dfs_h2o,calc_dfs_temp_h2o
import q2pwv
import h5py
import pyPCRTM
import PREFIRE_sim_tools
from calc_dfs_Sa_in import calc_dfs_3opt
import copy
import array

#set to 1 if Sa_in is in log space
Salogq = 1

#retrieval type 0=temp_only 1=q_only 2=both
#if you aren't setting the jacobian then you must set the input to the DFS function
ret_type = 2

nfilename = ('/home/mmm/projects/PREFIRE_ATM_retrieval/scripts/OE_static_data_Sa_model_01.h5')
new = h5py.File(nfilename,'r')
Sa = new['Sa']

#or set Sa with ndim = 0, this will compute an analytic Sa for each profile
#Sa = np.array(1)


#if you are ready to save the data set savedat = 1 and define the outfilename
savedat = 1
outfilename = '/data/users/nnn/DFS_analysis/channel_select/ERA5_DFSselect_pwv_SaMod01_Tropics_at1000xt8_faster.h5'

#all 54 channels
all_chan = np.array([4,5,6,7,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63])



#this code will change NEDR to perform tests
#set the SRF file to use
##srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'
# the ch7test
srf_file='/home/mmm/projects/PREFIRE_sim_tools/data/PREFIRE_SRF_v0.10.4_ch7test_360_2021-03-28.nc'


#use the default NEDR
NEDR = np.array(1)

#Choose which standard profile to use as to fill in profile data
# use subactric winter (4) ?
atm_num = 4

#choose which Aux-met data set to use 1= arctic ocean, 2=greenland, 3=antarctia, 4=tropic ocean
case = 4

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

nc.close()



#atrack,xtrack,levs = temp.shape
#test 1 profile
atrack = 1000
xtrack = 8
#dfs_temp = np.zeros([atrack,xtrack]) - 9999.0
#dfs_q = np.zeros([atrack,xtrack]) - 9999.0
#dfs_tq = np.zeros([atrack,xtrack]) - 9999.0
pwv = np.zeros([atrack,xtrack]) - 9999.0
#diag_t = np.zeros([atrack,xtrack,101]) - 9999.0
#diag_q = np.zeros([atrack,xtrack,101]) - 9999.0
#diag_tq = np.zeros([atrack,xtrack,202]) - 9999.0
#A_tq = np.zeros([atrack,xtrack,202,202]) - 9999.0
#A_t = np.zeros([atrack,xtrack,101,101]) - 9999.0
#A_q = np.zeros([atrack,xtrack,101,101]) - 9999.0


F = pyPCRTM.PCRTM()
F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True,output_bt_flag=True,output_jacob_bt_flag=True)

chan_order_all = np.zeros([atrack,xtrack,len(all_chan)])  - 9999.0
chan_total_dfs_all = np.zeros([atrack,xtrack,len(all_chan)])  - 9999.0

for i in range(atrack):
    for j in range(xtrack):
        print(i)
        temp_in = temp[i,j,:]
        q_in = q[i,j,:]
        st_in = surf_temp[i,j]
        sp_in = surf_pres[i,j]

        #find the Jacobian (K_in) to use for this profile
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

        F.tskin = copy.deepcopy(st_in)
        F.psfc = copy.deepcopy(sp_in)

 
        F.h2o = copy.deepcopy(q_in)
 

        dat = F.forward_rt()
        K = copy.deepcopy(dat['krad_t'])
    
        max_valid_level = P_levels_orig.searchsorted(sp_in) + 1
        K_zero_test = P_levels_orig < P_levels_orig[max_valid_level]
        Kt=K[:,K_zero_test]
        

        K_pcrm_all6 = copy.deepcopy(dat['krad_mol'])
        #K_h2o = K_pcrm_all6[:,K_zero_test,0]
        if Salogq == 1:
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

        K_in_fast = copy.deepcopy(K_use)
        use_lev_fast = copy.deepcopy(K_zero_test)
       
        chan_order_curr = np.zeros([len(all_chan)],dtype=int)
        chan_total_dfs_curr = np.zeros([len(all_chan)])
        #order the channels according to DFS info
        for k in range(len(all_chan)):
            #print(k)
            select_chan_curr = np.zeros([(k+1)],dtype=int)
            dfs_vals = np.zeros([len(all_chan)-k])
            #will have to loop over dfs_vals to find the which channel to add. 
            

            #start with the first channel with the most DFS
            if k == 0:
                #if k=0 then don't need to worry about previously selected chan
                for m in range(len(dfs_vals)):
                    select_chan_curr[k] = all_chan[m]
                    
                    select_chan_in = copy.deepcopy(select_chan_curr)
                    #print(select_chan_in)

                    A_tqout,dfs_tqout,dfsdiag_tqout = calc_dfs_3opt(ret_type,atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in,Sa_in=Sa,NEDR_in=NEDR,Sa_logq=Salogq,K_in=K_in_fast,use_lev=use_lev_fast)

                    dfs_vals[m] = copy.deepcopy(dfs_tqout)

                maxi = np.argmax(dfs_vals)
                max_chan = all_chan[maxi]
                chan_order_curr[k] = max_chan
                chan_total_dfs_curr[k] = np.max(dfs_vals)
                
            # now need to account for previously selected channels
            else:
                select_chan_curr[0:k] = chan_order_curr[0:k]
                

                #this will find the remaining channels to test, 
                #(less the chosen channels)
                #should be the same dim as dfs_vals
                remain_chan = np.setdiff1d(all_chan,chan_order_curr)
    

                for m in range(len(dfs_vals)):
                    select_chan_curr[k] = remain_chan[m]
                    print(select_chan_curr)
                    #sort the channels in case they need to be in order
                    select_chan_in = copy.deepcopy(np.sort(select_chan_curr))
                    
                    
                    A_tqout,dfs_tqout,dfsdiag_tqout = calc_dfs_3opt(ret_type,atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in,Sa_in=Sa,NEDR_in=NEDR,Sa_logq=Salogq,K_in=K_in_fast,use_lev=use_lev_fast)

                    dfs_vals[m] = copy.deepcopy(dfs_tqout)

                maxi = np.argmax(dfs_vals)
                max_chan = remain_chan[maxi]
                
                chan_order_curr[k] = max_chan
                chan_total_dfs_curr[k] = np.max(dfs_vals)
                


        chan_order_all[i,j,:] = chan_order_curr
        chan_total_dfs_all[i,j,:] = chan_total_dfs_curr
            
        #raise ValueError()


        pres_use = pressure < surf_pres[i,j]
        q_pwv = q_in[pres_use]
        pres_pwv = pressure[pres_use]
        pwv[i,j] = q2pwv.q2pwv(q_pwv,pres_pwv)
        print(pwv[i,j])
        

dat = {}
dat['pwv'] = pwv
dat['chan_order'] = chan_order_all 
dat['chan_total_dfs'] = chan_total_dfs_all



if savedat == 1:  
    with h5py.File(outfilename,'w') as h:
        for v in dat:
            h[v] = dat[v]

