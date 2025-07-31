import netCDF4
import numpy as np
import pyPCRTM
import matplotlib.pyplot as plt 
import PREFIRE_sim_tools
import OE_prototypes
import os.path
from PREFIRE_sim_tools.datasim import file_creation
import copy
import pcrtm_surface_emis
from numpy.random import seed
from numpy.random import randint

#set to 1 if you want to write data to file
writedat = 0


#choose the directory to write to
top_outdir = '/data/users/nnn/datasim_examples/var_surfemis/'

#choose which Aux-met data set to use 1=ocean, 2=greenland, 3=antarctia, 4=tropics
case = 1


#set these thresholds to use for surface emis types
high_stemp = 277.0
low_stemp = 269.0


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

#copy adat so we can add values later
ancdat = {}
ancdat['Anc-SimTruth'] = adat

at,xt,lev = temp.shape

#The surface types are as follows for the function mix_emis_pcrtm:
#    1 = pure water
#    2 = fine snow
#    3 = medium snow
#    4 = coarse snow 
#    5 = ice

#use pure water for cases where surface temp is greater than high_stemp
wemis,pcrtm_emis = pcrtm_surface_emis.mix_emis_pcrtm(1,1,100.0)


# use snow mixtures for cases where the surface temp is less than low_temp
# mix together different snow types
seed(111)
snonly_type1=randint(2,5,[at,xt]) #this produces cases 2-4
snonly_type2=randint(2,5,[at,xt])
snonly_perc = randint(0,100,[at,xt])

#use snow mixtures and pure water for the surface temp between the high and low thresholds
#mix together different snow types and a chance for melt ponds
seed(133)
snpw_type1=randint(1,5,[at,xt]) # this produces cases 1-4
snpw_type2=randint(1,5,[at,xt])
snpw_perc = randint(0,100,[at,xt])

#create arrays to save the values used to create the resultant surface emissivity
surface_emis = np.zeros((at,xt,740))-999.0
stype1 = np.zeros((at,xt))-999.0
stype2 = np.zeros((at,xt))-999.0
spercent1 = np.zeros((at,xt))-999.0

#For arctic ocean region use pure water above the high theshold
#    snow mixutre below the low threshold
#    and snow and water in between
if case == 1:
    for i in range(at):
        for j in range(xt):
            if surf_temp[i,j] > high_stemp:
                surface_emis[i,j,:] = wemis
                stype1[i,j] = 1
                stype2[i,j] = 1
                spercent1[i,j] = 100

            if surf_temp[i,j] < low_stemp:
                outemis,outpcrtm_emis = pcrtm_surface_emis.mix_emis_pcrtm(snonly_type1[i,j],snonly_type2[i,j],snonly_perc[i,j],mix_pcrtm_emis_wn=pcrtm_emis)
                surface_emis[i,j,:] = outemis
                stype1[i,j] = snonly_type1[i,j]
                stype2[i,j] = snonly_type2[i,j]
                spercent1[i,j] = snonly_perc[i,j]

            if surf_temp[i,j] >= low_stemp and surf_temp[i,j] <= high_stemp:
                snoutemis,outpcrtm_emis = pcrtm_surface_emis.mix_emis_pcrtm(snpw_type1[i,j],snpw_type2[i,j],snpw_perc[i,j],mix_pcrtm_emis_wn=pcrtm_emis)
                surface_emis[i,j,:] = snoutemis
                stype1[i,j] = snpw_type1[i,j]
                stype2[i,j] = snpw_type2[i,j]
                spercent1[i,j] = snpw_perc[i,j]

#for Greenland and Antarctica just use a mixture of snow type
if case == 2 or case == 3:
    for i in range(at):
        for j in range(xt):
            outemis,outpcrtm_emis = pcrtm_surface_emis.mix_emis_pcrtm(snonly_type1[i,j],snonly_type2[i,j],snonly_perc[i,j],mix_pcrtm_emis_wn=pcrtm_emis)
            surface_emis[i,j,:] = outemis
            stype1[i,j] = snonly_type1[i,j]
            stype2[i,j] = snonly_type2[i,j]
            spercent1[i,j] = snonly_perc[i,j]

#for Tropics just use pure water
if case == 4:
    for i in range(at):
        for j in range(xt):
            surface_emis[i,j,:] = wemis
            stype1[i,j] = 1
            stype2[i,j] = 1
            spercent1[i,j] = 100

#create a new variable to store the surfacetype fractions
#The surface types are as follows:
#    1 = pure water = index0
#    2 = fine snow = index1
#    3 = medium snow = index2
#    4 = coarse snow = index3
#    5 = ice = index4
surface_type_frac = np.zeros((at,xt,5))

for i in range(at):
    for j in range(xt):
        si1 = int(stype1[i,j]-1)
        si2 = int(stype2[i,j]-1)
        surface_type_frac[i,j,si1] = spercent1[i,j]/100.0
        #if si2=si1 then the total fraction will add to 1.0
        surface_type_frac[i,j,si2] = surface_type_frac[i,j,si2] + (100.0-spercent1[i,j])/100.0
            

#append data to ancdat
ancdat['Anc-SimTruth']['surf_emis'] = surface_emis
ancdat['Anc-SimTruth']['surf_type_fractions'] = surface_type_frac

Met_fstr = os.path.join(
        top_outdir,
        'PREFIRE_TEST_ANC-SimTruth_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
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
#    dat = {'Geometry':tdat, 'Anc-SimTruth':ancdat}
#    file_creation.write_data_fromspec(dat, '/home/nnn/projects/PREFIRE_sim_tools/PREFIRE_sim_tools/datasim/file_specs.json', Met_filename)
    #dat = {'Geometry':tdat, 'Anc-SimTruth':ancdat}
    file_creation.write_data_fromspec(ancdat, '/home/nnn/projects/PREFIRE_sim_tools/PREFIRE_sim_tools/datasim/file_specs.json', Met_filename)

raise ValueError()


