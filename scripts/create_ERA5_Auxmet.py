import netCDF4
import numpy as np
import glob
import h5py
import os.path
from PREFIRE_sim_tools.datasim import file_creation
from PREFIRE_sim_tools.utils import level_interp


# Created June 2020
#This program will read in ERA5 data from a specific region and create Aux-met files for the PREFILE project. 
# A subsample of 8000 (or set 'siz') profiles will be written to the file in the array [1000,8]. 
#case1  = ocean (lat = -5 to 5, lon = 70N to 80N)
#case2 = Greenland (lat = -45 to -35, lon = 70N to 80N )
#case3 = Antarctica (lat = -180 to 179, lon = 80S to 85S)
#case4 = Tropics (lat = 160 to 170, lon = 5S to 5N)

#choose a case,siz,and if you want to write the data to netCDF file
#1=ocean, 2 = Greenland, 3 = Antarctica, 4 = Tropcis
case = 4
#pick the total number of random profiles to write
siz =8000
#set equal to 1 if you are ready to write the data to Aux_met file
writedat = 1
#set directory to write the files to:
top_outdir = '/data/users/nnn/datasim_examples/'



#begin program
#
#load coefficents for ERA5 pressure levels
lv_coef_data = np.loadtxt(fname='/home/nnn/projects/ERA5_analysis/ERA5_get_data/model_level_coeff_numbers.csv', delimiter=',')
ac = lv_coef_data[1:,1]
bc = lv_coef_data[1:,2]

if case ==1:
    file_list = glob.glob('/data/users/nnn/ERA5/temp_q_ozone/ERA5_TQO_hourly_70Nto90N_2016*.nc')
if case ==2:
    file_list = glob.glob('/data/users/nnn/ERA5/temp_q_ozone/ERA5_TQO_hourly_70Nto90N_2016*.nc')
if case ==3:
    file_list = glob.glob('/data/users/nnn/ERA5/temp_q_ozone/ERA5_TQO_hourly_70Sto90S_2016*.nc')
if case ==4:
    file_list = glob.glob('/data/users/nnn/ERA5/temp_q_ozone/ERA5_TQO_hourly_n5Sto20N_160to180_2016*.nc')

#use this to subsample regions
if case ==1:
    lat_bottom = 21
    lat_top = 10
    lon_start = 175
    lon_end = 186
if case ==2:
    lat_bottom = 21
    lat_top = 10
    lon_start = 135
    lon_end = 146
if case ==3:
    lat_bottom = 16
    lat_top = 10
    lon_start = 0
    lon_end = 360
if case ==4:
    lat_bottom = 26
    lat_top = 15
    lon_start = 0
    lon_end = 11

for num,file_path in enumerate(file_list):
    print(file_path)
    nc = netCDF4.Dataset(file_path, 'r')
    if num==0:
        temp = nc['t'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        q = nc['q'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        o3 = nc['o3'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        log_sp = nc['lnsp'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        gp = nc['z'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        lon = nc['longitude'][lon_start:lon_end].astype(np.float32)
        lat = nc['latitude'][lat_top:lat_bottom].astype(np.float32)
        time = nc['time'][:].astype(np.float32)
        
        
    else:
        temp_2 = nc['t'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        q_2 = nc['q'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        o3_2 = nc['o3'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        log_sp_2 = nc['lnsp'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        gp_2 = nc['z'][:,:,lat_top:lat_bottom,lon_start:lon_end].astype(np.float32)
        time_2 = nc['time'][:].astype(np.float32)

        temp = np.vstack((temp,temp_2))
        q = np.vstack((q,q_2))
        o3 = np.vstack((o3,o3_2))
        log_sp = np.vstack((log_sp,log_sp_2))
        gp = np.vstack((gp,gp_2))
        time = np.hstack((time,time_2))
        

sfc_pres = np.exp(log_sp)


timesiz,levsiz,latsiz,lonsiz = temp.shape


np.random.seed(165)
randomtime = np.random.randint(timesiz,size=siz)
np.random.seed(244)
randomlat = np.random.randint(latsiz,size=siz)
np.random.seed(998)
randomlon = np.random.randint(lonsiz,size=siz)

temp_ran = temp[randomtime,:,randomlat,randomlon]
q_ran = q[randomtime,:,randomlat,randomlon]
o3_ran = o3[randomtime,:,randomlat,randomlon]
gp_ran = gp[randomtime,:,randomlat,randomlon]
sfc_pres_ran = sfc_pres[randomtime,:,randomlat,randomlon]
lat_ran = lat[randomlat]
lon_ran = lon[randomlon]
time_ran = time[randomtime]

atm_pres_ran = sfc_pres_ran*0.0
nprofs, nlevels = sfc_pres_ran.shape
for num in range(nprofs):
    atm_pres_ran[num,:] = (sfc_pres_ran[num,0]*bc) + ac

#convert to hPa
pres_ran = atm_pres_ran/100.0
#convert to g/kg
q_ran = q_ran*1000.0
#convert from kg/kg to ppm
M_d = 28.87
M_O3 = 48.0
o3_ran = o3_ran*(M_d/M_O3)*1e6

#skin_temp from bottom layer
st_ran = temp_ran[:,-1]
#surface pressure in hPa
sp_ran = sfc_pres_ran[:,0]/100.0

#split it up into 8 "xtracks" for profiles in the atrack
xt = int(8.0)
profs,lev = temp_ran.shape
at = int(profs/xt)

skin_temp = np.zeros((at,xt))
temp_profile = np.zeros((at,xt,lev))
wv_profile = np.zeros((at,xt,lev))
surface_pressure = np.zeros((at,xt))
pres_profile = np.zeros((at,xt,lev))
o3_profile = np.zeros((at,xt,lev))
time_UTC = np.zeros((at,xt))
latitude = np.zeros((at,xt))
longitude = np.zeros((at,xt))

for i in range(xt):
    print(i)
    skin_temp[:,i] = st_ran[i*at:i*at+at]
    surface_pressure[:,i] = sp_ran[i*at:i*at+at]
    temp_profile[:,i,:] = (temp_ran[i*at:i*at+at,:])
    wv_profile[:,i,:] = (q_ran[i*at:i*at+at,:])
    o3_profile[:,i,:] = (o3_ran[i*at:i*at+at,:])
    pres_profile[:,i,:] = (pres_ran[i*at:i*at+at,:])
    time_UTC[:,i] = time_ran[i*at:i*at+at]
    latitude[:,i] = lat_ran[i*at:i*at+at]
    longitude[:,i] = lon_ran[i*at:i*at+at]


#Here is where to interpolate to PCRTM levels
data_in1 = {}
data_in1['temp'] = temp_profile
data_in1['pressure_profile'] = pres_profile

data_in2 = {}
data_in2['q'] = wv_profile
data_in2['o3'] = o3_profile
data_in2['pressure_profile'] = pres_profile

intp = np.loadtxt('/home/nnn/projects/PREFIRE_sim_tools/data/plevs101.txt')

dat_interp1 = level_interp.pressure_interp(data_in1,intp,surf_extrap_method=3)
temp_interp = dat_interp1['temp']

dat_interp2 = level_interp.pressure_interp(data_in2,intp,surf_extrap_method=0)
wv_interp = dat_interp2['q']
o3_interp = dat_interp2['o3']

#create contant co2 and ch4 values
co2_profile = (np.arange(8000).reshape((1000,8)))*0.0 + 400.0
ch4_profile = (np.arange(8000).reshape((1000,8)))*0.0 + 1.8

# first, create the analysis data dictionary according to file_specs.json
adat = {}

if case == 1:
    adat['analysis_name'] = np.array(['ERA5_ocean'])
if case == 2:
    adat['analysis_name'] = np.array(['ERA5_Greenland'])
if case == 3:
    adat['analysis_name'] = np.array(['ERA5_Antarctica'])
if case == 4:
    adat['analysis_name'] = np.array(['ERA5_Tropics'])

FillValue = -9999

adat['surface_pressure'] = surface_pressure
adat['surface_temp'] = skin_temp
adat['skin_temp'] = skin_temp
adat['pressure_profile'] = intp 
adat['temp_profile'] = temp_interp
adat['wv_profile'] = wv_interp
adat['o3_profile'] = o3_interp
adat['xco2'] = co2_profile
adat['xch4'] = ch4_profile

sensor_zenith = (np.arange(8000).reshape((1000,8)))*0.0
land_fraction = (np.arange(8000).reshape((1000,8)))*100.0

#tdat will go in the geometry file 
tdat = {}
tdat['time_UTC'] = time_UTC
tdat['latitude'] = latitude
tdat['longitude'] = longitude
tdat['sensor_zenith'] = sensor_zenith
tdat['land_fraction'] = land_fraction



Met_fstr = os.path.join(
        top_outdir,
        'PREFIRE_TEST_AUX-MET_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
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


savedat = 0

if savedat == 1:
    outfilename = '/data/users/nnn/ERA5/h5/ERA5_Aux_met_test_antarctica.h5'
    with h5py.File(outfilename,'w') as h:
        for v in adat:
            h[v] = adat[v]
