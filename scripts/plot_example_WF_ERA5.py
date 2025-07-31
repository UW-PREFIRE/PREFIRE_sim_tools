import numpy as np
import pyPCRTM
import netCDF4
import copy
import PREFIRE_sim_tools
import PREFIRE_sim_tools.utils
import matplotlib.pyplot as plt  
from scipy import interpolate
from matplotlib.ticker import ScalarFormatter

#this program will plot brightness temps on one panel and weight functions is another. 


#set this value to 1 if you wish to use ERA5 data. If set to zero you will use the standard profile set below. 
use_ERA5 = 1

#choose case to run. 
#Case 1 over open ocean July 2016
#Case 2 over Summit Greenland July 2016

ERA5_case = 1

if ERA5_case == 1:
    #case 1 over open ocean
    #use this to subsample regions
    #choose 75N  and 0E
    lat_use = 15
    lon_use = 180
    #choose an index for the time (hourly data) for the profile
    time_use = 54
    ERA5_file = '/data/users/nnn/ERA5/temp_q_ozone/ERA5_TQO_hourly_201607.nc'

if ERA5_case == 2:
    #case 2 over summit Greenland
    #use this to subsample regions
    #choose 73N  and 38W
    lat_use = 17
    lon_use = 142
    #choose an index for the time for the profile
    time_use = 147 #46 # 84 #12
    ERA5_file = '/data/users/nnn/ERA5/temp_q_ozone/ERA5_TQO_hourly_201601.nc'

std_atm_names = ['tropic', 'midlat_summer', 'midlat_winter',
                 'subarc_summer', 'subarc_winter', 'us_standard']

#choose standard profile for foundation of PCRTM input (0 = tropic, 1 =mls, etc.)
if ERA5_case == 1:
    a = 3
if ERA5_case == 2:
    a =4

#choose which SRF version to use 0 = v0.09.1, 1 = v0.09.2:
srfuse=1

srffiles = ['/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.1_2020-02-21.nc','/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc']

# the PCRTM fixed pressure levels.
P_levels_orig = np.loadtxt('../data/plevs101.txt')


F = pyPCRTM.PCRTM()

sensor_id = 2

F.init(sensor_id,
           output_jacob_flag=True,
           output_tr_flag=False,
           output_ch_flag=True,
           output_jacob_ch_flag=True,
           output_bt_flag=True,
           output_jacob_bt_flag=True)

F.psfc = 1000.0
F.pobs = 0.005
F.sensor_zen = 0.0
F.emis = 1.0 + np.zeros(F.num_monofreq, np.float32)



#color_names = ['b-', 'k-', 'g-','r-', 'c-', 'm-', 'y-']

nc = netCDF4.Dataset('../../PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')


F.z = nc['z'][:,a].astype(np.float32)
F.tskin = nc['temp'][-1,a] 

F.tlev = nc['temp'][:,a].astype(np.float32) 
# very approx conversion from ppm to q [g/kg]
q = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
F.h2o = q.astype(np.float32)
F.co2 = nc['co2'][:,a].astype(np.float32)
F.o3 = nc['o3'][:,a].astype(np.float32)
F.n2o = nc['n2o'][:,a].astype(np.float32)
F.co = nc['co'][:,a].astype(np.float32)
F.ch4 = nc['ch4'][:,a].astype(np.float32)



if use_ERA5 == 1:

   
    #use an ERA5 profile for temp and q
    #load coefficents for pressure levels
    lv_coef_data = np.loadtxt(fname='/home/nnn/projects/ERA5_analysis/ERA5_get_data/model_level_coeff_numbers.csv', delimiter=',')
    ac = lv_coef_data[1:,1]
    bc = lv_coef_data[1:,2]


    # load the variables, then close file
    #ncfile = '/data/users/nnn/ERA5/temp_q_ozone/ERA5_TQO_hourly_201607.nc'
    nc = netCDF4.Dataset(ERA5_file, 'r')
    lv = nc['level'][:].astype(np.float32)
    time = nc['time'][time_use].astype(np.float32)
    lat = nc['latitude'][lat_use].astype(np.float32)
    lon = nc['longitude'][lon_use].astype(np.float32)
    temp = nc['t'][time_use,:,lat_use,lon_use].astype(np.float32)
    q = nc['q'][time_use,:,lat_use,lon_use].astype(np.float32)
    o3 = nc['o3'][time_use,:,lat_use,lon_use].astype(np.float32)
    log_sp = nc['lnsp'][time_use,:,lat_use,lon_use].astype(np.float32) #log of surface pressure
    gp = nc['z'][time_use,:,lat_use,lon_use].astype(np.float32) #geopotential
    nc.close()

    sfc_pres = np.exp(log_sp)

    temp_sm = copy.deepcopy(temp)
    q_sm = copy.deepcopy(q)
    sfc_pres_sm = copy.deepcopy(sfc_pres)

#    temp_sm = temp[time_use,:,lat_use,lon_use]
#    q_sm = q[time_use,:,lat_use,lon_use]
#    sfc_pres_sm = sfc_pres[time_use,:,lat_use,lon_use]

    # find pressure at levels
    atm_pres = (sfc_pres_sm[0]*bc) + ac

    #convert to hPa
    pres = atm_pres/100.0
    #convert to g/kg
    q_sm = q_sm*1000.0
    
    #interpolate to 101 levels
    # the PCRTM fixed pressure levels.

    minlev = np.min(pres,axis=0)
    maxlev = np.max(pres,axis=0)

    t_out = interpolate.interp1d(pres[:],temp_sm[:])
    q_out = interpolate.interp1d(pres[:],q_sm[:])

    iuse  = (maxlev > P_levels_orig )*(P_levels_orig > minlev)
    t_new = t_out(P_levels_orig[iuse])
    q_new = q_out(P_levels_orig[iuse])
    temp_ERA5 = F.tlev
    temp_ERA5[iuse] = t_new
    q_ERA5 = F.h2o
    q_ERA5[iuse] = q_new

    p_new = P_levels_orig[iuse]
    
    F.tlev = temp_ERA5
    F.h2o = q_ERA5
    F.psfc = p_new[-1]
#raise ValueError('stop')

r = F.forward_rt()

#get the pressure levels of PCRTM
#P_levels_orig = copy.deepcopy(F.plevels)

    #determines the Jacobian
Kt_pcrm = copy.deepcopy(r['krad_t'])
K_zero_test = np.all(Kt_pcrm==0,axis=0)==0
Kt=Kt_pcrm[:,K_zero_test]
    

w,wr,yc,_ = PREFIRE_sim_tools.TIRS.srf.apply_SRF_wngrid(Kt, SRFfile=srffiles[srfuse],spec_grid='wl')


ym = np.ma.masked_invalid(yc)

t_curr = copy.deepcopy(F.tlev)
t_curr_use = t_curr[K_zero_test]


ym=ym.transpose()
yc = yc.transpose()


wn = copy.deepcopy(r['wn'])
wl = 10000.0/wn
bt =copy.deepcopy(r['bt'])

P_levels = P_levels_orig[K_zero_test]
P_levels_rev = P_levels[::-1]
levels = np.arange(0,98)
levels_rev = levels[::-1]



colors = ['r','g','b','purple','c','k','y']


#plt.show()
fignum = 4
fig = plt.figure(fignum, figsize=(10,10))
fig.clf()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_xlabel('BT [K]',fontsize=34)
#ax1.set_ylabel('Wavenumber [cm$^-1$]',fontsize=14)
ax1.set_ylabel('Wavelength [$\mu$m]',fontsize=34)
ax1.tick_params(labelsize=24)


#choose the indices of the wavelengths to plot
wn_sub = [6,12,19,24,30,40,46]

#fillzero = np.arange(7)*0.0+230.0

#interpolate to brightness temps
bt_out = interpolate.interp1d(wl,bt)
bt_new = bt_out(w[wn_sub])

ax1.plot(bt,wl)

if ERA5_case == 1:
    ax1.set_xlim(220,290)
if ERA5_case == 2:
    ax1.set_xlim(195,265)
ax1.set_ylim(3,55)
#ax1.set_title(std_atm_names[a])

if ERA5_case == 1:
    ax1.set_title('Ocean, Summer',fontsize=30)
    ax2.set_title('Ocean, Summer',fontsize=30)
    #dummy = 1
#    fig.suptitle('3 July 2016 Ocean',fontsize=28)
#    fig.suptitle('Ocean',fontsize=28)
if ERA5_case == 2:
    ax1.set_title('Greenland, Winter',fontsize=30)
    ax2.set_title('Greenland, Winter',fontsize=30)
    #fig.suptitle('7 Jan 2016 Summit, Greenland')
#    fig.suptitle('Greenland')
#fig.suptitle(std_atm_names[a])




for i, aname in enumerate(wn_sub):
    
    print(w[wn_sub[i]])

    ax1.plot(bt_new[i],w[wn_sub[i]],marker = '>',markersize = 15, color = colors[i])

    bb_dt =  PREFIRE_sim_tools.utils.blackbody.dradT_wavelen(t_curr_use,w[wn_sub[i]])
    wf = ym[:,wn_sub[i]]/bb_dt

    curr_wl = '{0:7.2f}'.format(w[wn_sub[i]])+' $\mu$m'
    ax2.plot(wf,P_levels,color = colors[i],label=curr_wl)


ax2.set_yscale('log')
ax2.set_ylim(1000,50)
ax2.set_xlim(0,0.1)
#ax2.xaxis.label.set_size(40)
#ax2.set_yticklabels(['1000','800','600','400','200'])
ax2.tick_params(labelsize=24)

for axis in [ax2.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

ax2.set_xlabel('WF',fontsize=34)
ax2.set_ylabel('log Pressure [mbar]',fontsize=34)
#ax2.set_title('Greenland Ocean',fontsize=34)

#ax2.set_title('legend in wavelength')
ax2.legend(loc = 'upper right',fontsize=20)

plt.tight_layout()
plt.show()
#plt.savefig('/home/nnn/plots/PREFIRE/WFs/ERA5_20160107_WF_wl_summit_bigax.png'
#plt.close()
