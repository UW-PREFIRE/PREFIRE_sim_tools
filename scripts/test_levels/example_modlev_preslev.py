import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pyPCRTM
import PREFIRE_sim_tools
import OE_prototypes 
from PREFIRE_sim_tools.datasim import file_creation
from PREFIRE_sim_tools.utils import level_interp
from PREFIRE_ATM_utils import calc_pwv
import copy

#set to 1 if you want to save the plots. set the directory at the and of this script
plotsave = 0

# load the surface file
filepath = '/data/ERA5/ERA5_60Sto90S_single/ERA5_single_hourly_60Sto90S_2012-07-01.nc'
    
nc1 = netCDF4.Dataset(filepath, 'r')

#    nc1 = xr.open_dataset(filepath + date + '.nc')
    
sfc_pres = nc1['sp'][:,:,:].astype(np.float32)
skin_t = nc1['skt'][:,:,:].astype(np.float32)

#raise ValueError()

#load the pressure level file
presfile = '/data/ERA5/ERA5_ANT_plevel_20120101.nc'
nc1 = netCDF4.Dataset(presfile, 'r')

t1 = nc1['t'][:,:,:,:].astype(np.float32)
q1 = nc1['q'][:,:,:,:].astype(np.float32)
o31 = nc1['o3'][:,:,:,:].astype(np.float32)
time1 = nc1['time'][:].astype(np.float32)
lon1 = nc1['longitude'][:].astype(np.float32)
lat1 = nc1['latitude'][:].astype(np.float32)
level1 = nc1['level'][:].astype(np.float32)



#load coefficents for ERA5 pressure levels for the model level files
lv_coef_data = np.loadtxt(fname='/home/nnn/projects/ERA5_analysis/ERA5_get_data/model_level_coeff_numbers.csv', delimiter=',')
ac = lv_coef_data[1:,1]
bc = lv_coef_data[1:,2]


modfile = '/data/ERA5/ERA5_60Sto90S_modlev/monthly_TQO/ERA5_TQO_hourly_60Sto90S_2012-01.nc'
nc = netCDF4.Dataset(modfile, 'r')

#just look at the first 24 hours
starttime = 0
endtime =23
t = nc['t'][starttime:endtime,:,:,:].astype(np.float32)
q = nc['q'][starttime:endtime,:,:,:].astype(np.float32)
o3 = nc['o3'][starttime:endtime,:,:,:].astype(np.float32)
log_sp = nc['lnsp'][starttime:endtime,:,:,:].astype(np.float32)
lon = nc['longitude'][:].astype(np.float32)
lat = nc['latitude'][:].astype(np.float32)
time = nc['time'][starttime:endtime].astype(np.float32)

sfc_pres = np.exp(log_sp)
atm_pres = sfc_pres*0.0
ntime, nlevels, nlat, nlon = sfc_pres.shape

#for num in range(nprofs):
#just look at 0 UTC at 0 lon for all lats
for num in range(nlat):
    atm_pres[0,:,num,0] = (sfc_pres[0,0,num,0]*bc) + ac


#convert to hPa
pres = atm_pres/100.0
#convert to g/kg
q = q*1000.0
#convert from kg/kg to ppm
M_d = 28.87
M_O3 = 48.0
o3 = o3*(M_d/M_O3)*1e6

#skin_temp from bottom layer
##st = temp[:,-1]
#surface pressure in hPa
sp = sfc_pres[:,0,:,:]/100.0

#filename ending should match indices below
plotfile = '_24_96.png'

#lat_use = 30
#lat1_use = 120 

#lat_use = 28
#lat1_use = 112

#lat_use = 0
#lat1_use = 0

#lat_use = 26
#lat1_use = 104

lat_use = 24
lat1_use = 96

#lon is hardcoded to 0 for this test
# at time = 0

q_mod = q[0,:,lat_use,0]
t_mod = t[0,:,lat_use,0]
pres_mod = pres[0,:,lat_use,0]
sp_mod = sp[0,lat_use,0] #surface pressure to use
st_mod = t_mod[-1] #surface temp to use

q_plev = q1[0,:,lat1_use,0]*1000.0
t_plev = t1[0,:,lat1_use,0]

noshow =1

if noshow ==0:

    #plot the orginal T and Q profiles
    fignum = 2
    fig = plt.figure(fignum, figsize=(10,10))
    fig.clf()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(q_mod,pres_mod,'+',linestyle='dotted')
    ax1.plot(q_plev,level1,'+',linestyle='solid')
    ax1.set_ylim(1000,0)
    ax1.set_ylabel('Pressure [hPa]')
    ax1.set_xlabel('g/kg')
    ax1.set_title('Q Profile')



    ax2.plot(t_mod,pres_mod,'+',linestyle='dotted')
    ax2.plot(t_plev,level1,'+',linestyle='solid')
    ax2.set_ylim(1000,0)
    ax2.set_ylabel('Pressure [hPa]')
    ax2.set_xlabel('K')
    ax2.set_title('Temp Profile')


    if plotsave == 1:
        plt.savefig('/home/nnn/plots/PREFIRE/level_tests/Profs_orig_'+plotfile)
    else:
        plt.show()



#Here is where to interpolate to PCRTM levels
data_in_mod0 = {}
data_in_mod0['temp'] = np.transpose(t[0,:,:,:])
data_in_mod0['pressure_profile'] = np.transpose(pres[0,:,:,:])

data_in_mod1 = {}
data_in_mod1['q'] = np.transpose(q[0,:,:,:])
#data_in_mod1['o3'] = o3_profile
data_in_mod1['pressure_profile'] = np.transpose(pres[0,:,:,:])

intp = np.loadtxt('/home/nnn/projects/PREFIRE_sim_tools/data/plevs101.txt')

#raise ValueError()

dat_interp_mod = level_interp.pressure_interp(data_in_mod0,intp,surf_extrap_method=3)
temp_interp_mod = dat_interp_mod['temp'][0,lat_use,:]

dat_interp_mod1 = level_interp.pressure_interp(data_in_mod1,intp,surf_extrap_method=0)
q_interp_mod = dat_interp_mod1['q'][0,lat_use,:]
#o3_interp = dat_interp2['o3']

#test pwv function
pwv_test = calc_pwv.q2pwv(q_interp_mod,intp,sp_mod)
#raise ValueError()

#now interp the pressure level files
timenum,levnum,latnum,lonnum = q1.shape
level_tile = np.tile(level1,(lonnum,latnum,1))

data_in_plev0 = {}
data_in_plev0['temp'] = np.transpose(t1[0,:,:,:])
data_in_plev0['pressure_profile'] = level_tile

data_in_plev1 = {}
data_in_plev1['q'] = np.transpose(q1[0,:,:,:])*1000
#data_in_plev1['o3'] = o3_profile
data_in_plev1['pressure_profile'] = level_tile

#raise ValueError()

dat_interp_plev = level_interp.pressure_interp(data_in_plev0,intp,surf_extrap_method=0)
temp_interp_plev = dat_interp_plev['temp'][0,lat1_use,:]

dat_interp_plev1 = level_interp.pressure_interp(data_in_plev1,intp,surf_extrap_method=0)
q_interp_plev = dat_interp_plev1['q'][0,lat1_use,:]
#o3_interp = dat_interp_plev['o3']

t_plev= t1[0,:,0,0]
t_interp = np.interp(intp,level1,t_plev)

#raise ValueError()

fignum = 1
fig = plt.figure(fignum, figsize=(10,10))
fig.clf()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(q_interp_mod,intp,'+',linestyle='dotted')
ax1.plot(q_interp_plev,intp,'+',linestyle='solid')
ax1.set_ylim(1000,0)
ax1.set_ylabel('Pressure [hPa]')
ax1.set_xlabel('g/kg')
ax1.set_title('Q Profile')



ax2.plot(temp_interp_mod,intp,'+',linestyle='dotted')
ax2.plot(temp_interp_plev,intp,'+',linestyle='solid')
ax2.set_ylim(1000,0)
ax2.set_ylabel('Pressure [hPa]')
ax2.set_xlabel('K')
ax2.set_title('Temp Profile')


if plotsave == 1:
    plt.savefig('/home/nnn/plots/PREFIRE/level_tests/Profs_interp_'+plotfile)
else:
    plt.show()

#now convert to radiances
#set the SRF file to use
#srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc'
srf_file='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'

#number of channels in SRF output
nchan = 63



# load std as the met profile (effectively)
nc = netCDF4.Dataset('/home/nnn/projects/PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')


#Creates PCRTM wrapper
F = pyPCRTM.PCRTM()

F.init(2, True, True, True, False, True, False)

plev = F.plevels

#run PCRTM with the model level data
F.psfc = sp_mod
#     F.psfc = 1000.0
F.pobs = 0.005
F.sensor_zen = 0.0
#setting surface emissivity to 1
F.emis = 1.0 + np.zeros(F.num_monofreq, np.float32)

std_atm_names = ['tropic', 'midlat_summer', 'midlat_winter',
                 'subarc_summer', 'subarc_winter', 'us_standard']
#Choose which standard profile to use as to fill in profile data
# use subactric winter (4) 
a=4

#F.tskin = nc['temp'][-1,a]
F.tskin = st_mod

std_t_mod = nc['temp'][:,a].astype(np.float32)
# very approx conversion from ppm to q [g/kg]
q = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
std_q_mod = q.astype(np.float32)

#std_t_mod[10:101] = temp_interp_mod[10:101]
#std_q_mod[10:101] = q_interp_mod[10:101]
#F.tlev = copy.deepcopy(std_t_mod)
#F.h2o = copy.deepcopy(std_q_mod)
F.tlev = copy.deepcopy(temp_interp_mod)
F.h2o = copy.deepcopy(q_interp_mod)

F.co2 = nc['co2'][:,a].astype(np.float32)
F.o3 = nc['o3'][:,a].astype(np.float32)
F.n2o = nc['n2o'][:,a].astype(np.float32)
F.co = nc['co'][:,a].astype(np.float32)
F.ch4 = nc['ch4'][:,a].astype(np.float32)
        
#run the model
r = F.forward_rt()
    
trans_mod = copy.deepcopy(r['layer_trans'])
rad_mod = copy.deepcopy(r['rad'])
wn_pcrtm = copy.deepcopy(r['wn'])
bt_mod = copy.deepcopy(r['bt'])
plev = copy.deepcopy(F.plevels)



#raise ValueError()

#rerun after changing to plev data
std_t_pres = nc['temp'][:,a].astype(np.float32)
# very approx conversion from ppm to q [g/kg]
q = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
std_q_pres = q.astype(np.float32)

#F.tlev = nc['temp'][:,a].astype(np.float32)
# very approx conversion from ppm to q [g/kg]
#q = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
#F.h2o = q.astype(np.float32)

#std_t_pres[10:101] = temp_interp_plev[10:101]
#std_q_pres[10:101] = q_interp_plev[10:101]
#F.tlev = copy.deepcopy(std_t_pres)
#F.h2o = copy.deepcopy(std_q_pres)

#std_t_pres[10:101] = t_interp_plev[10:101]

raise ValueError()

F.tlev = copy.deepcopy(temp_interp_plev)
F.h2o = copy.deepcopy(q_interp_plev)

##F.tlev = nc['temp'][:,a].astype(np.float32)
# very approx conversion from ppm to q [g/kg]
##q = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
##F.h2o = q.astype(np.float32)

#run the model again
r2 = F.forward_rt()
    

trans_plev = copy.deepcopy(r2['layer_trans'])
rad_plev = copy.deepcopy(r2['rad'])
wn_pcrtm2 = copy.deepcopy(r2['wn'])
bt_plev = copy.deepcopy(r2['bt'])
plev2 = copy.deepcopy(F.plevels)

nc.close()

srffile='/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.10.4_360_2021-03-28.nc'

#find the TIRS convolved radiance
w_mod,wr_mod,tirs_mod,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(rad_mod, SRFfile=srffile,spec_grid='wl')
w_plev,wr_plev,tirs_plev,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(rad_plev, SRFfile=srffile,spec_grid='wl')

#convert to  BT
tirs_mod_swap = np.swapaxes(tirs_mod,0,1)
bt_mod_tirs,test1_mod = PREFIRE_sim_tools.TIRS.radiometry.btemp(tirs_mod_swap,spec_grid='wl',SRFfile=srffile)

#convert to  BT
tirs_plev_swap = np.swapaxes(tirs_plev,0,1)
bt_plev_tirs,test1_plev = PREFIRE_sim_tools.TIRS.radiometry.btemp(tirs_plev_swap,spec_grid='wl',SRFfile=srffile)

#raise ValueError()

msk = (bt_mod_tirs > 0)
msk=msk[0,:]

fignum = 3
fig = plt.figure(fignum, figsize=(12,10))
fig.clf()

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(1e4/wn_pcrtm,bt_mod,label='PCRTM model level')#,'+',linestyle='dotted')
ax1.plot(1e4/wn_pcrtm,bt_plev,label='PCRTM pressure levels')#,'+',linestyle='solid')

ax1.plot(w_mod[msk],bt_mod_tirs[0,msk],'<',label='TIRS model levels')
ax1.plot(w_plev[msk],bt_plev_tirs[0,msk],'<',label='TIRS pressure levels')

ax1.set_xlim(3,55)
ax1.set_ylabel('Brightness Temperature [K]')
ax1.set_xlabel('Wavelength [um]')
#ax1.set_title(' ')

ax1.legend(loc = 'upper right',fontsize=14)

ax2.plot(1e4/wn_pcrtm,bt_mod-bt_plev)#,'+',linestyle='dotted')
ax2.plot(w_mod[msk],bt_mod_tirs[0,msk]-bt_plev_tirs[0,msk],'<')

#ax2.plot(1e4/wn_pcrtm,bt_plev)#,'+',linestyle='solid')
ax2.set_xlim(3,55)
ax2.set_ylim(-1,2)
ax2.set_ylabel('Brightness Temperature Diff [K] (ML - PL)')
ax2.set_xlabel('Wavelength [um]')
#ax1.set_title(' ')

#plt.legend(loc = 'upper right',fontsize=14)


if plotsave == 1:
    plt.savefig('/home/nnn/plots/PREFIRE/level_tests/BT_Diff_'+plotfile)
else:
    plt.show()

raise ValueError()
