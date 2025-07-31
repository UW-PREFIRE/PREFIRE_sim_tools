"""
construct orbit/geo data, from CALIOP L1b, or
from PREFIRE orbit sim (TBD)
"""

import datetime
import os.path

import numpy as np
import h5py
import pyhdf.SD

from . import file_creation

def _segment_orbit_lat(lat):
    """
    segments the orbit sim data by the latitude.
    This is assumed to be a 1D vector (sat ground track lat),
    and this is a bunch of orbits concatenated together.
    Segmenting is done on the ascending orbit passes of the equator.
    """
    msk = np.diff(lat) > 0
    msk = np.logical_and(msk, lat[:-1] < 0)
    msk = np.logical_and(msk, lat[1:] > 0)

    slices = []
    idx = msk.nonzero()[0]

    for i in range(idx.shape[0]-1):
        slices.append(slice(idx[i]+1, idx[i+1]+1))

    return slices


def _prepare_times(ref_time_utc, nsamples):
    """
    sort copy-paste kludge here, because the times of the samples
    are not currently written out to fit

    outputs the (N,7) shaped array we want to populate the time_UTC
    in the product file. This is integers:
    yr, mn, day, hr, min, sec, millsecond
    """
    # sample time in seconds.
    TIRS_tau = 0.707

    time_utc = np.zeros((nsamples, 7), np.int16)
    yr  = int(ref_time_utc[0])
    m   = int(ref_time_utc[1])
    dy  = int(ref_time_utc[2]) 
    hr  = int(ref_time_utc[3])
    mn  = int(ref_time_utc[4])
    sec = int(ref_time_utc[5])
    microsec = int( 1e6*(ref_time_utc[5] - sec) )
    dt0 = datetime.datetime(yr, m, dy, hr, mn, sec, microsec)

    for n in range(nsamples):
        dt = dt0 + datetime.timedelta(seconds=TIRS_tau * n)
        time_utc[n,0] = dt.year
        time_utc[n,1] = dt.month
        time_utc[n,2] = dt.day
        time_utc[n,3] = dt.hour
        time_utc[n,4] = dt.minute
        time_utc[n,5] = dt.second
        time_utc[n,6] = int(np.round(dt.microsecond/1000))

    return time_utc


def preprocess_PREFIRE_L1_orbit_sim(matfile):
    """
    basically one-use function, to take the large output file
    from Brian D.'s PREFIRE_L1 orbit sim, cut it into orbit chunks, and
    then write out quasi-L1b files.
    """

    spec_file = 'file_specs.json'

    with h5py.File(matfile,'r') as h:
        sat_lat = h['satint/geod_lat'][:,0]
        time_utc0 = h['sat/utc'][0]

    matfile_only = os.path.split(matfile)[1]
    outpath_fstr = (
        '/data/users/mmm/datasim_orbits/'+
        matfile_only.replace('.mat', '_prep_segment{0:02d}.nc4'))
    ss = _segment_orbit_lat(sat_lat)

    gdat = {}

    time_utc = _prepare_times(time_utc0, sat_lat.shape[0])

    for n, s in enumerate(ss):

        gdat['time_UTC'] = time_utc[s]

        with h5py.File(matfile, 'r') as h:
            
            gdat['latitude'] = h['ground/P'][s,:]
            gdat['longitude'] = h['ground/Q'][s,:]
            gdat['latitude_geoid'] = h['ground/P'][s,:]
            gdat['longitude_geoid'] = h['ground/Q'][s,:]
            # slice 1: on 2nd dimension, is because Ps stores the center point
            # then the 4 corners (Ps[:,0,:,0] should be the same as P)
            # 0 on trailing dimension, this is the spectral dimension that
            # is currently redundant.
            gdat['latitude_vertices'] = np.rollaxis(h['ground/Ps'][s,1:,:,0], 1, 3)
            gdat['longitude_vertices'] = np.rollaxis(h['ground/Qs'][s,1:,:,0], 1, 3)
            
            gdat['subsat_latitude'] = h['satint/geod_lat'][s,0]
            gdat['subsat_longitude'] = h['satint/geod_lon'][s,0]
            gdat['sat_altitude'] = h['satint/geod_alt'][s,0]

        out_file = outpath_fstr.format(n)
        dat = {'Geometry':gdat}
        file_creation.write_data_fromspec(dat, spec_file, out_file)
    

def load_CALIOP_geo(hfile):
    """
    loads the key geometry variables from CALIOP L1b file.

    Parameters
    ----------
    hfile : str
        path to CALIOP L1b hdf file.

    Returns
    dat : dict
        python dictionary with the loaded geometry data.
        values are numpy ndarrays.
    """
    h = pyhdf.SD.SD(hfile, pyhdf.SD.SDC.READ)
    dat = {}
    var_list = [
        'Latitude', 'Subsatellite_Latitude',
        'Longitude', 'Subsatellite_Longitude',
        'Spacecraft_Altitude',
        'Solar_Azimuth_Angle', 'Solar_Zenith_Angle',
        'Surface_Elevation','IGBP_Surface_Type',
        'GMAO_Surface_Elevation', 'Land_Water_Mask',
        'Earth-Sun_Distance',
        'Profile_Time', 'Profile_UTC_Time',
        'Viewing_Zenith_Angle', 'Viewing_Azimuth_Angle'
    ]

    for dname in var_list:
        d = h.select(dname)
        # to remove the singleton dimension, all arrays are [N,1]
        tmp_vals = d.get()[:,0]
        if 'scale_factor' in d.attributes():
            sf = np.float32(d.attributes()['scale_factor'])
            tmp_vals = tmp_vals * sf
        if 'add_offset' in d.attributes():
            ao = np.float32(d.attributes()['add_offset'])
            tmp_vals = tmp_vals + ao
        dat[dname] = tmp_vals
        del d

    h.end()

    return dat

def _extract_digits_from_UTCfloat(x, d0, dn):
    # helper to extract digits from the CALIOP UTC float time
    tmp = x / 10**d0 - x // 10**d0
    d = np.trunc(tmp * 10**dn).astype(np.int16)
    return d

def _along_track_approx_distance(lat, lon, R):
    # approx along-orbit distance traveled since the first point.
    # this is a poor man's great circle distance (assumes spherical earth)
    th_r = np.deg2rad(lat)
    phi_r = np.deg2rad(lon)
    cos_angle = (
        np.sin(th_r)*np.sin(th_r[0]) + 
        np.cos(th_r)*np.cos(th_r[0]) * np.cos(phi_r-phi_r[0]) )
    invalid_msk = cos_angle > 1.0
    if np.any(invalid_msk):
        cos_angle[invalid_msk] = 1.0

    angle = np.arccos(cos_angle)
    R_earth = 6378.0
    dist = angle * R

    return dist


def generate_test_orbit_from_CALIOP(cdat):
    """
    generates synthetic TIRS from CALIOP geo data.
    Right now this is stupid-simple:
    replicates the CALIOP orbit footprint into 8 footprints, 
    each being separated by some approx surface offset, only in 
    latitude (so it is easier.) Assumes all zeniths are the same,
    and solar angles don't change.

    This generates just the lat/lon centroids for the footprints, along
    with the along track space craft alt, and others.

    Parameters
    ----------
    cdat : dict
        python dictionary with the CALIOP geometry data (from load_CALIOP_geo())

    Returns
    -------
    tdat : dict
        python dictionary containing the simulated TIRS geometry.
        These keys will match those in the TIRS file specification.
    """

    tdat = {}

    # quasi-fixed parameters describing footprint spacing.
    fp_size_km = 14.0
    fp_xtrack_gap_km = 30.0
    # for testing, set this to a big number (creates many fewer
    # footprints along track.)
    #fp_atrack_step_km = 1000.0
    fp_atrack_step_km = fp_size_km / 3.0
    num_xtrack = 8
    # remainder should derive from these.

    # to find the atrack spacing of the CALIOP data: 
    # use helper function, to find 'along track distance' relative to the
    # first point in the orbit. Then find the median delta, ignoring the
    # first 20 (these seem noisy), and only use the first half.
    # since we are now using whole CALIOP orbits the delta is negative for 
    # the second half.
    R_earth = 6378.0
    npts_half = cdat['Latitude'].shape[0] // 2
    dist = _along_track_approx_distance(
        cdat['Latitude'][:npts_half], cdat['Longitude'][:npts_half], R_earth)
    caliop_atrack_delta = np.median(np.diff(dist[10:]))
    num_atrack_step = int(np.ceil(fp_atrack_step_km/caliop_atrack_delta))

    # num along track will just be whatever number we get from the 
    # CALIOP track after the sparse subsample.
    subslice = slice(num_atrack_step // 2, None, num_atrack_step)
    lat1 = cdat['Latitude'][subslice]
    lon1 = cdat['Longitude'][subslice]
    num_atrack = lat1.shape[0]

    # make offset array spaced by input parameters, and centered.
    fp_offsets_km = np.arange(num_xtrack) * (fp_size_km + fp_xtrack_gap_km)
    fp_offsets_km = fp_offsets_km - np.mean(fp_offsets_km)

    # to facilitate shapes for broadcasting.
    shape2D = (num_atrack, num_xtrack)
    shape1a = (num_atrack, 1)
    shape1x = (1, num_xtrack)

    tdat = {}

    tdat['obs_id'] = np.zeros(shape2D, np.int64)
    id_ramp = np.arange(1,num_atrack+1, dtype=np.int64)*100
    tdat['obs_id'][:] = (
        id_ramp.reshape((num_atrack,1)) +
        np.arange(1,9).reshape((1,8)))

    degree_per_km = 360.0 / (R_earth * 2 * np.pi)
    lon_offsets = ( 
        (fp_offsets_km * degree_per_km).reshape(shape1x) /
        (np.cos(np.deg2rad(lat1))).reshape(shape1a) )

    tdat['latitude'] = lat1.reshape(shape1a) + np.zeros(shape1x)
    tdat['longitude'] = lon1.reshape(shape1a) + lon_offsets

    # for vertices, just make these a square box equal to fp_size_km
    # corners: NE, SE, SW, NW; e.g., clockwise, from 0 azimuth (North)
    # lat corner offsets are fixed, so we can use broadcasting.
    vert_dlat = 0.5*fp_size_km * degree_per_km
    corner_lat = np.array([vert_dlat, -vert_dlat, -vert_dlat, vert_dlat])
    corner_lat = corner_lat.reshape((1,1,4))
    # lon corners will change through orbit.
    vert_dlon = (vert_dlat / np.cos(np.deg2rad(lat1))).reshape((num_atrack,1,1))
    corner_lon = np.zeros(tdat['latitude'].shape + (4,))
    corner_lon[...,0:2] = vert_dlon
    corner_lon[...,2:4] = -vert_dlon

    tdat['latitude_vertices'] = (
        tdat['latitude'][:,:,np.newaxis] + corner_lat )
    tdat['longitude_vertices'] = (
        tdat['longitude'][:,:,np.newaxis] + corner_lon )

    tdat['latitude_geoid'] = lat1.reshape(shape1a) + np.zeros(shape1x)
    tdat['longitude_geoid'] = lon1.reshape(shape1a) + lon_offsets

    tdat['time_tai93'] = cdat['Profile_Time'][subslice]
    tdat['time_UTC'] = np.zeros((num_atrack, 7), dtype=np.int16)
    UTC_ftime = cdat['Profile_UTC_Time'][subslice]

    tdat['time_UTC'][:,0] = _extract_digits_from_UTCfloat(UTC_ftime, 6, 2)+2000
    tdat['time_UTC'][:,1] = _extract_digits_from_UTCfloat(UTC_ftime, 4, 2)
    tdat['time_UTC'][:,2] = _extract_digits_from_UTCfloat(UTC_ftime, 2, 2)
    UTC_fday = UTC_ftime - np.trunc(UTC_ftime)
    UTC_fhr = UTC_fday * 24.0
    UTC_fmin = (UTC_fhr - np.trunc(UTC_fhr)) * 60.0
    UTC_fsec = (UTC_fmin - np.trunc(UTC_fmin)) * 60.0
    UTC_fms = (UTC_fsec - np.trunc(UTC_fsec)) * 1000.0
    tdat['time_UTC'][:,3] = np.trunc(UTC_fhr)
    tdat['time_UTC'][:,4] = np.trunc(UTC_fmin)
    tdat['time_UTC'][:,5] = np.trunc(UTC_fsec)
    tdat['time_UTC'][:,6] = np.trunc(UTC_fms)
    
    tdat['sensor_zenith'] = (
        cdat['Viewing_Zenith_Angle'][subslice][:,np.newaxis] + np.zeros(shape1x))
    tdat['sensor_azimuth'] = (
        cdat['Viewing_Azimuth_Angle'][subslice][:,np.newaxis] + np.zeros(shape1x))

    tdat['solar_zenith'] = (
        cdat['Solar_Zenith_Angle'][subslice][:,np.newaxis] + np.zeros(shape1x))
    tdat['solar_azimuth'] = (
        cdat['Solar_Azimuth_Angle'][subslice][:,np.newaxis] + np.zeros(shape1x))
    # this is in AU in CALIOP, maybe that's easier?
    tdat['solar_distance'] = (
        cdat['Earth-Sun_Distance'][subslice][:,np.newaxis] + np.zeros(shape1x))

    tdat['subsat_latitude'] = cdat['Subsatellite_Latitude'][subslice]
    tdat['subsat_longitude'] = cdat['Subsatellite_Longitude'][subslice]

    tdat['sat_altitude'] = cdat['Spacecraft_Altitude'][subslice]

    # don't actually know how to do this at the moment, so make it Fill.
    tdat['solar_beta_angle'] = np.array([-9999.0])

    # Land fraction, altitude, altitude std, will be filled by GFDL sampling

    return tdat
