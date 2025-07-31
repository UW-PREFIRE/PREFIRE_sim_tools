import netCDF4
import numpy as np
import pyPCRTM
import matplotlib.pyplot as plt 
import PREFIRE_sim_tools
import OE_prototypes
import os.path
from PREFIRE_sim_tools.datasim import file_creation
import copy
from PREFIRE_ATM_OE import OE_calclib

#set to 1 if you'd like to write data to file
writedat = 0

#set to 0 for the larger uncertainty values
#set to 1 for the smaller values that match the prior covarinace maxtrix
unc_case = 1

#set tq_case to reflect if you wish to perturb temp only (tq_case = 1)
# or perturb temp and q profiles (tq_case = 2)
# or perturb temp and q profiles and surface temp (tq_case = 3)
tq_case = 2

#set the SRF file to use
#srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc'

#number of channels in SRF output
nchan = 63

#Choose which standard profile to use as to fill in profile data
# use subactric winter (4) 
atm_num = 4

#choose the directory to write to
if unc_case == 0:
    top_outdir = '/data/users/nnn/datasim_examples/PERT_QlogSa_5K0p8_06162021/'
if unc_case == 1:
    top_outdir = '/data/users/nnn/datasim_examples/PERT_QlogSa_2K0p6_06162021/'


#choose which Aux-met data set to use 1=ocean, 2=greenland, 3=antarctia, 4=tropics
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

adat = {}
for varname in nc['Aux-Met'].variables:
    adat[varname] = nc['Aux-Met'][varname][:]
tdat = {}
for varname in nc['Geometry'].variables:
    tdat[varname] = nc['Geometry'][varname][:]


nc.close()

n_levels = 101

# this is the DFS analytic function.
# the PCRTM fixed pressure levels.
# need this to construct Sa.
P_levels = np.loadtxt('../data/plevs101.txt')

if unc_case == 0:
    # uncertainty for each of the 101 levels
    T_uncertainty = np.zeros(n_levels) + 5.0
    # uncertainty for the surface temp
    Ts_uncertainty = 5.0
    # uncertainty for Q in log space
    lnQ_uncertainty = 0.8


#to match the prior Mod01
if unc_case == 1:
    # uncertainty for each of the 101 levels
    T_uncertainty = np.zeros(n_levels) + 2.0
    # uncertainty for the surface temp
    Ts_uncertainty = 2.0
    # uncertainty for Q in log space
    lnQ_uncertainty = 0.6

# T_correlation, this is the exponential scale length in [hPa], since
# we are using pressure levels.
T_correlation = np.linspace(10.0, 200.0, n_levels)
Q_correlation = np.linspace(10.0, 200.0, n_levels)

Sa_t, Ra_t = OE_calclib.level_exp_cov_matrix(P_levels, T_uncertainty, T_correlation)
Sa_q, Ra_q = OE_calclib.level_exp_cov_matrix(P_levels, lnQ_uncertainty, Q_correlation)

xt = int(8.0)
at = int(1000.0)
lev = int(101.0)

temp_pert = np.zeros((at,xt,lev))
q_pert = np.zeros((at,xt,lev))
ts_pert = np.zeros((at,xt))

np.random.seed(1113)
for i in range(at):
    for j in range(xt):
        temp_random = OE_prototypes.random_correlated(temp[i,j,:],Sa_t,K=1)
        temp_pert[i,j,:] = temp_random[:,0]
        if tq_case >= 2:
            #make sure there are no unrealistic negative values in q. 
            thres_q = q[i,j,:] < 0
            q[i,j,thres_q] = 1e-5
            
            lnq = np.log(q[i,j,:])
            lnq_random = OE_prototypes.random_correlated(lnq,Sa_q,K=1)
            q_pert[i,j,:] = np.exp(lnq_random[:,0])

            #make sure there are no unrealistic negative values in q_pert. 
            thres = q_pert[i,j,:] < 0
            q_pert[i,j,thres] = 1e-7

            if tq_case == 3:
                ts_pert[i,j] = np.random.normal(surf_temp[i,j], Ts_uncertainty, 1)

adat['temp_profile'] = temp_pert
if tq_case >= 2:
    adat['wv_profile'] = q_pert
    if tq_case ==3:
        adat['surface_temp']  = ts_pert

# filename format *TPERT* is temp only. *TQPERT* is temp and q.  
#    *TQTSPERT* is temp and q and surf_temp(ts)
if tq_case == 1:
    Met_fstr = os.path.join(
        top_outdir,
        'PREFIRE_TPERT_AUX-MET_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
if tq_case == 2:
    Met_fstr = os.path.join(
        top_outdir,
        'PREFIRE_TQPERT_AUX-MET_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
if tq_case == 3:
    Met_fstr = os.path.join(
        top_outdir,
        'PREFIRE_TQTSPERT_AUX-MET_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
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
    dat = {'Geometry':tdat, 'Aux-Met':adat}
    file_creation.write_data_fromspec(dat, '/home/nnn/projects/PREFIRE_sim_tools/PREFIRE_sim_tools/datasim/file_specs.json', Met_filename)


plotshow = 0

if plotshow == 1:

    plt.figure(1, figsize=(5,10))
    plt.plot(q[67,2,:],P_levels,label='orig')
    plt.plot(q_pert[67,2,:],P_levels,label='randomnoise 1')
    #plt.plot(test[:,50],P_levels,label='randomnoise 2')
    plt.ylim(1100,0)
    #plt.xlim(200,300)
    plt.legend(loc = 'upper right')
    plt.show()

