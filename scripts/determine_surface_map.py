#
# Attempt to determine the spatial map of surface surftypes.
#
# The method is to load the monthly surface emis. climatology
# file, then look for the array matches between the emissivity
# spectrum at some lat-lon grid cell, and the emissivity spectra
# within "by-type" file.
# These matches give the type used for that lat-lon grid cell,
# assuming that the climatology file does not mix surface types
# following some sub grid cell fractional mixing.
#
# A., Oct 12 2020
#

import netCDF4
import numpy as np

month_file = 'surface_emissivity_1cm_0.5x0.5_53deg_month08.nc'
output_file = 'surface_surftypes_0.5x0.5_53deg_month08.nc'
surftype_file = 'surface_emissivity_for_11types_53deg_2.nc'

with netCDF4.Dataset(surftype_file, 'r') as ncC:
    wn_surftype = ncC['wn'][:]
    emis_bytype = []
    # stick one element as "padding" because the emis spectra
    # are 1-ordered.
    emis_bytype.append(np.zeros((wn_surftype.shape[0],1)))
    for n in range(1,10):
        emis_bytype.append(ncC['emis'+str(n)][:][:,np.newaxis])
    emis_bytype.append(ncC['emis10'][:].T)
    emis_bytype.append(ncC['emis11'][:][:,np.newaxis])
emis_bytype = np.concatenate(emis_bytype, axis=1)

with netCDF4.Dataset(month_file, 'r') as nc:
    emis_map = nc['emissivity'][:]
    wn_month = nc['wn'][:]
    lat = nc['lat'][:]
    lon = nc['lon'][:]
assert np.all(wn_surftype == wn_month)

# create an array that is a boolean, specifying if the s'th
# surftype is equal to the spectrum in the monthly climatology
# array at lat/lon (i,j). This will yield an array of
# shape (nlat, nlon, ntype), surftype_bmap
# use broadcasting to do this in one step
# emis_map axes will be (lat, lon, wn, 1)
# emis_bytype will be (1, 1, wn, surftype)
emis_map = emis_map[..., np.newaxis]
emis_bytype = emis_bytype[np.newaxis, np.newaxis, ...]
surftype_bmap = np.all(emis_map == emis_bytype, axis=2)

# convert this to a lat/lon shape integer array, with
# an integer specifying the surface type.
surftype_imap = np.zeros(surftype_map.shape[:2], np.int)
for i in range(surftype_bmap.shape[-1]):
    surftype_imap[surftype_bmap[:,:,i]] = i

with netCDF4.Dataset(output_file, 'w'):
    nc.createDimension('lat', lat.shape[0])
    nc.createDimension('lon', lon.shape[0])
    nc.createDimension('surftype', surftype_bmap.shape[-1])
    nc.createVariable('lat', lat.dtype, ('lat',))
    nc.createVariable('lon', lon.dtype, ('lon',))
    # it doesn't like bool - use u8 ?
    #nc.createVariable('surftype_bmap', surftype_bmap.dtype,
    nc.createVariable('surftype_bmap', np.uint8,
                      ('lat', 'lon', 'surftype'))
    nc.createVariable('surftype_imap', surftype_imap.dtype,
                      ('lat', 'lon'))
    nc['lat'][:] = lat
    nc['lon'][:] = lon
    nc['surftype_imap'][:] = surftype_imap
    nc['surftype_bmap'][:] = surftype_bmap.astype(np.uint8)
