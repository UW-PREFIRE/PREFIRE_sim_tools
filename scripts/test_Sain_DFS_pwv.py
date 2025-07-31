import netCDF4
import numpy as np
import PREFIRE_sim_tools
from calc_dfs_auxtypein import calc_dfs_temp,calc_dfs_h2o,calc_dfs_temp_h2o
import q2pwv
import h5py
import pyPCRTM
import PREFIRE_sim_tools
from calc_dfs_Sa_in import calc_dfs_3opt

#set to 1 if Sa_in is in log space
Salogq = 1

#for testing Sa_in option
#vfilename = ('/home/nnn/projects/PREFIRE_sim_tools/scripts/'+
#                'test_Sa.h5')
#nfilename = ('/home/nnn/projects/PREFIRE_ATM_retrieval/scripts/'+
#                'delete_OE_static_data_Sa_nonlog.h5')
#nfilename = ('/home/nnn/projects/PREFIRE_ATM_retrieval/scripts/'+
#                'delete_OE_static_data_Sa_log.h5')

nfilename = ('/home/mmm/projects/PREFIRE_ATM_retrieval/scripts/OE_static_data_Sa_model_01.h5')
new = h5py.File(nfilename,'r')
Sa = new['Sa']

#or set Sa with ndim = 0, this will compute an analytic Sa for each profile
#Sa = np.array(1)


#if you are ready to save the data set savedat = 1 and define the outfilename
savedat = 0
outfilename = '/data/users/nnn/DFS_analysis/ERA5_A_DFS_diag_pwv_qlog_SaMod01_allchan_Ocean.h5'

#which channels to use? set 'select_chan_in' here
#use for all channels
select_chan_in = np.array(range(63)) 

# remove channel 7 only
#select_chan_in = np.array([4,5,6,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63])

# remove channels 4-5 and 37 and above
#select_chan_in = np.array([6,7,10,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])
# remove channels 18 and above (plus masked channels) to only do the MIR
#select_chan_in = np.array([4,5,6,7,10,11,12,13,14,15,16])

#this code will change NEDR to perform tests
#set the SRF file to use
srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'
# the ch7test
#srf_file='/home/mmm/projects/PREFIRE_sim_tools/data/PREFIRE_SRF_v0.10.4_ch7test_360_2021-03-28.nc'


#file 2 may be used for the NEDR_in tests
srf_file2='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc'
#srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.1_2020-02-21.nc'
#srf_file = '/home/mmm/projects/PREFIRE_sim_tools/data/PREFIRE_SRF_v0.09.2_lwtest_2020-08-15.nc'

#test out NEDR_in
##srfdata = netCDF4.Dataset(srf_file,'r')
##NEDR_srf = srfdata['NEDR'][:].astype(np.float32)
#for 10.4 and higher
##NEDR = NEDR_srf[:,0]/1.0
#raise ValueError()
##srfdata.close()

#replace the channels less than 30 microns with old srf NEDR
##srfdata2 = netCDF4.Dataset(srf_file2,'r')
##NEDR_srf2 = srfdata2['NEDR'][:].astype(np.float32)
#for 9.2 and lower
##NEDR[36:] = NEDR_srf2[36:]
#raise ValueError()
##srfdata2.close()

#use the default NEDR
NEDR = np.array(1)

#Choose which standard profile to use as to fill in profile data
# use subactric winter (4) ?
atm_num = 4

#choose which Aux-met data set to use 1= arctic ocean, 2=greenland, 3=antarctia, 4=tropic ocean
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

nc.close()


atrack,xtrack,levs = temp.shape
dfs_temp = np.zeros([atrack,xtrack]) - 9999.0
dfs_q = np.zeros([atrack,xtrack]) - 9999.0
dfs_tq = np.zeros([atrack,xtrack]) - 9999.0
pwv = np.zeros([atrack,xtrack]) - 9999.0
diag_t = np.zeros([atrack,xtrack,101]) - 9999.0
diag_q = np.zeros([atrack,xtrack,101]) - 9999.0
diag_tq = np.zeros([atrack,xtrack,202]) - 9999.0
A_tq = np.zeros([atrack,xtrack,202,202]) - 9999.0
A_t = np.zeros([atrack,xtrack,101,101]) - 9999.0
A_q = np.zeros([atrack,xtrack,101,101]) - 9999.0


F = pyPCRTM.PCRTM()
F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True,output_bt_flag=True,output_jacob_bt_flag=True)


for i in range(atrack):
    #print(i)
    for j in range(xtrack):
        print(i)
        temp_in = temp[i,j,:]
        q_in = q[i,j,:]
        st_in = surf_temp[i,j]
        sp_in = surf_pres[i,j]
        #A_tout,dfs_tout,dfsdiag_tout = calc_dfs_temp(atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in)
        A_tout,dfs_tout,dfsdiag_tout = calc_dfs_3opt(0,atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in,Sa_in=Sa,NEDR_in=NEDR,Sa_logq=Salogq)
        dfs_temp[i,j] = dfs_tout
        diag_t[i,j,:] = dfsdiag_tout
        #A_qout0,dfs_qout0,dfsdiag_qout0 = calc_dfs_h2o(atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in)
        A_qout,dfs_qout,dfsdiag_qout = calc_dfs_3opt(1,atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in,Sa_in=Sa,NEDR_in=NEDR,Sa_logq=Salogq)
        dfs_q[i,j] = dfs_qout
        diag_q[i,j,:] = dfsdiag_qout
        #A_tqout0,dfs_tqout0,dfsdiag_tqout0 = calc_dfs_temp_h2o(atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in)
        A_tqout,dfs_tqout,dfsdiag_tqout = calc_dfs_3opt(2,atm_num,temp_in,q_in,st_in,sp_in,F,srf_file=srf_file,use_chan=select_chan_in,Sa_in=Sa,NEDR_in=NEDR,Sa_logq=Salogq)
        #raise ValueError()
        dfs_tq[i,j] = dfs_tqout
        diag_tq[i,j,:] = dfsdiag_tqout
        A_t[i,j,:,:] = A_tout
        A_q[i,j,:,:] = A_qout
        A_tq[i,j,:,:] = A_tqout
        #raise ValueError() 
        pres_use = pressure < surf_pres[i,j]
        q_pwv = q_in[pres_use]
        pres_pwv = pressure[pres_use]
        pwv[i,j] = q2pwv.q2pwv(q_pwv,pres_pwv)
        print(pwv[i,j])
        

dat = {}
dat['pwv_interp'] = pwv
dat['dfs_temp'] = dfs_temp 
dat['dfs_q'] = dfs_q
dat['dfs_tq'] = dfs_tq
dat['surf_temp'] = surf_temp
dat['surf_pres'] = surf_pres
dat['dfs_diag_t'] = diag_t
dat['dfs_diag_q'] = diag_q
dat['dfs_diag_tq'] = diag_tq
dat['A_t'] = A_t
dat['A_q'] = A_q
dat['A_tq'] = A_tq



if savedat == 1:  
    with h5py.File(outfilename,'w') as h:
        for v in dat:
            h[v] = dat[v]

