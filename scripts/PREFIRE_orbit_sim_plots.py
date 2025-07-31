import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
import h5py
import netCDF4

###################################################################
# example 1: plot the high resolution simulation spectrum with the 
# simulated TIRS channels.

# high res data stored in a MATLAB file.
# there are 3 times (03, 06, 09 UTC), two locations (Antarctic, 
# greenland), and two simulations (origin and with_pond).

# the slices in the full (6144, 12288) array are:
# greenland = [5427:5495, 10889:10957]
# antartica = [478:546, 2185:2253]
#
highres_file = (
    '/data/users/xc/GFDL_radiance_simulation/data_for_pond_study/'+
    'GFDL_20160801_03UTC_Antarctic_origin.mat')


# low res channel radiances
# there are additional dirs to match origin and with_pond
lowres_file = (
    '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
    'TIRS_SRF_v0.09.2_origin/03UTC/TIRS_radiance_20160801_03UTC.h5' )

# get a high res spectrum.
m = loadmat(highres_file)
hr_wavenum_rad = m['rad'][0,0,:]
hr_wavenum = m['wn_PCRTM'][:,0]

# convert to per wavelen (in um), 1e4 needed for unit conversion
hr_wavelen = 1e4 / hr_wavenum
# radiance in [W / (m2 sr um)], 1e6 needed for unit conversion
hr_wavelen_rad = hr_wavenum_rad / hr_wavelen**2 * 1e4

# get the TIRS channel radiances
with h5py.File(lowres_file, 'r') as h:
    TIRS_wavelen = h['wavelen'][:]
    TIRS_radiance = h['rad'][478, 2185, :]
    radiance_image = h['rad'][400:700, 2000:2400, 6]
    image_lat = h['lat'][400:700, 2000:2400]
    image_lon = h['lon'][400:700, 2000:2400]

# plot high res wavenum spectrum
fig, ax = plt.subplots()
ax.plot(hr_wavenum, hr_wavenum_rad)
ax.set_ylabel('Radiance [W / (m^2 sr cm^-1)]')
ax.set_xlabel('Wavenumber [cm^-1]')
fig.savefig('example_plot_wavenum_radiance.png')

# plot wavelen
fig, ax = plt.subplots()
ax.plot(hr_wavelen, hr_wavelen_rad)
ax.set_ylabel('Radiance [W / (m^2 sr um)]')
ax.set_xlabel('Wavelength [um]')
ax.plot(TIRS_wavelen, TIRS_radiance, 'ob')
ax.set_xlim(0, 55)
fig.savefig('example_plot_wavelen_radiance.png')


###################################################################
# example 2: plot TIRS footprints

# orbit simulation file: this uses the global model data (as above),
# but applies spatial averaging according to the simulated sensor
# footprints.
# the global data pixels are approx 3km X 3km, the TIRS sensor footprints
# are approx 15km x 15km, so they are approx averages of 5x5 of the
# global sim pixels.


orbit_file = (
    '/data/users/mmm/datasim_examples/TIRS_SRF_v0.09.1_origin/'+
    '03UTC/PREFIRE_SAT1_1B-RAD_S00_R00_20190317084606_00011.nc')
with netCDF4.Dataset(orbit_file, 'r') as nc:
    vlat = nc['Geometry/latitude_vertices'][8280:8300,:,:]
    vlon = nc['Geometry/longitude_vertices'][8280:8300,:,:]
    c12_rad = nc['Radiance/spectral_radiance'][8280:8300,:,12]
    

fig, ax = plt.subplots()
ax.pcolormesh(image_lon, image_lat, radiance_image, cmap='magma')
ii = [0, 1, 2, 3, 0]
fp_index = 3
for n in range(vlat.shape[0]):
    ax.plot(vlon[n,fp_index,:], vlat[n,fp_index,:], marker='x', color='white')
    ax.plot(vlon[n,fp_index,ii], vlat[n,fp_index,ii], color='white')

ax.set_xlim(63, 66)
ax.set_ylim(-75.5,-74.25)

fig.savefig('example_plot_TIRS_footprints.png')
