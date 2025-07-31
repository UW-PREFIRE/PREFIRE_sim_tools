from datetime import datetime
import os.path

import netCDF4
import h5py
import numpy as np
from shapely.geometry import Polygon

from PREFIRE_sim_tools import PCRTM_utils

_GFDL_data_fstr = (
    '/data/users/xc/GFDL_fields_3km/{0:s}/'+
    '{1:s}_C3072_12288x6144.fre.day1_03UTC.nc4' )
# it has different name root, so just hardcode it
_GFDL_topo_file = (
    '/data/users/xc/GFDL_fields_3km/'+
    'orog_filt_C3072_24576x12288.fre.nc' )
    
def _local_slices(lat, lon, vlat, vlon):
    """helper function to compute row and column slices for the two-dimensional
    lat/lon arrays. Inputs are lat, lon (the 1D vectors defining the equal angle
    grids), and vlat, vlon, the vertices of the lat-lon box that will be extracted.
    the return slices give the bounding sub-array that contains the vertices."""

    a, b = lat.searchsorted([vlat.min(), vlat.max()])
    if a > 0:
        a -= 1
    # decided to do this in the caller.
    #if b < lat.shape[0]:
    #    b += 1
    slice1 = slice(a, b)

    a, b = lon.searchsorted([vlon.min(), vlon.max()])
    if a > 0:
        a -= 1
    # decided to do this in the caller.
    #if b < lon.shape[0]:
    #    b += 1
    slice2 = slice(a, b)

    return slice1, slice2

def _store_grid_weights(slices, weights, norms, wfile):
    """
    helper to write out the computed grid weights.
    """
    shape2D = norms.shape
    islice_starts = np.zeros(shape2D, np.int32)
    islice_stops = np.zeros(shape2D, np.int32)
    jslice_starts = np.zeros(shape2D, np.int32)
    jslice_stops = np.zeros(shape2D, np.int32)
    weight_shapes = np.zeros(shape2D + (2,), np.int32)

    for a,b in np.ndindex(norms.shape):
        si, sj = slices[a,b]
        # various awkward checks for vertex groups that failed
        # In those cases, the slices are None, and weights is an
        # integer 0 (not an array). Note that all conditions happen
        # together, so we only need to check one.
        # in that condition, set the various outputs to -1. Will check
        # this condition when loading.
        if si is None:
            islice_starts[a,b] = -1
            islice_stops[a,b] = -1
            jslice_starts[a,b] = -1
            jslice_stops[a,b] = -1
        else:
            islice_starts[a,b] = si.start
            islice_stops[a,b] = si.stop
            jslice_starts[a,b] = sj.start
            jslice_stops[a,b] = sj.stop
            weight_shapes[a,b,:] = weights[a,b].shape

    num_weights = weight_shapes[:,:,0] * weight_shapes[:,:,1]
    max_num = np.max(num_weights)
    weights_3D = np.zeros(shape2D + (max_num,)) - 1

    for a,b in np.ndindex(norms.shape):
        if num_weights[a,b] > 0:
            weights_3D[a,b,:num_weights[a,b]] = weights[a,b].flatten()

    with h5py.File(wfile, 'w') as h:
        h['weights_3D'] = weights_3D
        h['num_weights'] = num_weights
        h['weight_shapes'] = weight_shapes
        h['norms'] = norms
        h['islice_starts'] = islice_starts
        h['islice_stops'] = islice_stops
        h['jslice_starts'] = jslice_starts
        h['jslice_stops'] = jslice_stops


def _load_grid_weights(wfile):
    """
    helper to reload the weights data from an h5, and put it back into
    the numpy object arrays
    """
    with h5py.File(wfile, 'r') as h:
        weights_3D = h['weights_3D'][:]
        num_weights = h['num_weights'][:]
        weight_shapes = h['weight_shapes'][:]
        norms = h['norms'][:]
        islice_starts = h['islice_starts'][:]
        islice_stops = h['islice_stops'][:]
        jslice_starts = h['jslice_starts'][:]
        jslice_stops = h['jslice_stops'][:]        

    shape2D = norms.shape
    slices = np.zeros(shape2D, np.object)
    weights = np.zeros(shape2D, np.object)

    for a,b in np.ndindex(shape2D):
        if islice_starts[a,b] == -1:
            si = None
            sj = None
        else:
            si = slice(islice_starts[a,b], islice_stops[a,b])
            sj = slice(jslice_starts[a,b], jslice_stops[a,b])
            # use copy so the original can be GC'ed eventually
            tmp = weights_3D[a,b,:num_weights[a,b]].copy()
            weights[a,b] = tmp.reshape(weight_shapes[a,b])
            slices[a,b] = (si, sj)

    return slices, weights, norms


def _compute_grid_weights(lat_vertices, lon_vertices, 
                          clat, clon):
    """
    this is the core function that does the footprint overlapping,
    to compute a weighting array on the high res GFDL grid for each
    TIRS footprint.

    Parameters
    ----------
    lat_vertices : ndarray
        The vertex latitudes for the TIRS footprints, shaped (N, 8, 4) for
        the 8 footprints, with the last dimension (shape 4) containing the
        vertices (e.g. the corners of the footprint rectangle).

    lon_vertices : ndarray
        The vertex longitudes

    clat : ndarray
        The corner latitudes for the high resolution grid.
        If the high res field is shaped (A,B), in Lat, Lon, the
        clat, clon arrays should be shaped (A+1,), (B+1), respectively.

    clon : ndarray
        The corner longitudes.

    Returns
    -------

    slices : ndarray, object dtype
        array containing 2-element tuples (lat slice, lon slice) for each footprint.
        this is the slice pair into the high res grid. This is an object array
        because each element is a python tuple with 2 items.

    weights : ndarray, object dtype
        Each element contains the weighting array, with a shape matching what
        would be extracted using the slice pair at this footprint. This is an object
        array because it can be ragged - the footprints will contain different
        numbers of high res points depending on the location.

    norms : ndarray
        the sum of the weights; a float array.

    Note
    ----
    all return arrays are shaped (N,8), matching the number of obs and
        footprints in the input vertex arrays.

    """

    # this is the core data describing the mapping and weights from the
    # high res (3km GFDL) to the sensor grid; the slices are pairs of
    # row-slice, column-slice, in the high res grid, for the "thumbnail"
    # containing that footprint; weights are the fractional areas of
    # each high res pixel overlapped with the footprint;
    # norms are the total area encompassed by the footprint (in terms of
    # sq. degrees).
    # 

    shape2D = lat_vertices.shape[:2]

    slices = np.zeros(shape2D, np.object)
    weights = np.zeros(shape2D, np.object)
    norms = np.zeros(shape2D)
    normsz = np.zeros(shape2D)

    # new for L1b Simulations:
    # have to reorder the vertices. From L1b Sim, 
    # these are ordered:
    # #1: ahead-port; #2: behind-port; #3 ahead-starboard, #4 behind-starb.
    # for ascending, this is 1-NW, 2-SW, 3-NE, 4-SE.
    # for the Polygon, needs to be convex. (old way, for ascending, was:
    # 1-NE, 2-SE, 3-SW, 4-NW.
    #
    # soo... force a reorder of the L1Sim format, to change to
    # (NW, SW, SE, NE). This will break the cold for the old stuff - but
    # I don't think that matters, old stuff won't be used anymore. If we need
    # again, we should change the CALIOP synthetic footprints instead, to match
    # the L1 convention.
    reorder_idx = [0,1,3,2]

    for ij in np.ndindex(shape2D):

        # note the [ij] on the 3D vertex arrays is equivalent to
        # [ij[0], ij[1], :]
        # copy the lon, since it is altered.
        vlat = lat_vertices[ij][reorder_idx]
        vlon = lon_vertices[ij][reorder_idx]

        # here, we need to account for the GFDL longitude domain
        # (0,360) versus the input orbit track data domain (-180,180).
        if np.any(vlon < 0):
            if np.sum(vlon < 0) == 4:
                # this will wrap the footprint around to the correct domain
                # to match GFDL.
                vlon = vlon + 360
            else:
                # in this case, the footprint is straddling 0 longitude,
                # which means it would wrap around the GFDL domain.
                # for now, just bail on these FP.
                slices[ij] = None, None
                continue

        si, sj = _local_slices(clat, clon, vlat, vlon)

        # the slices for the corner lat/lon needs to be +1 on the stop index.
        # e.g., for a lon/x slice of [10:15] in the data array, this is 
        # defined by corner values with a slice of [10:16].
        csi = slice(si.start, si.stop+1)
        csj = slice(sj.start, sj.stop+1)
        clat_i = clat[csi]
        clon_j = clon[csj]

        # weights should be sized one less than the corner arrays.
        wt_ij = np.zeros((clat_i.shape[0]-1, clon_j.shape[0]-1))
        fpPoly = Polygon(zip(vlat, vlon))

        # subtract 1, because the data in clat_i, clon_j represents the
        # lat/lon bin edges for each sample in the Model Met. fields
        for k,l in np.ndindex(wt_ij.shape):
            lat_bdry = clat_i[[k+1, k, k, k+1, k+1]]
            lon_bdry = clon_j[[l+1, l+1, l, l, l+1]]
            gridPoly = Polygon(zip(lat_bdry, lon_bdry))
            wt_ij[k,l] = fpPoly.intersection(gridPoly).area
            #plt.plot(lon_bdry, lat_bdry, '--')
            #if k==0 and l==0:
            #    print(str(fpPoly))
            #    print(str(gridPoly))

        slices[ij] = (si,sj)

        weights[ij] = wt_ij
        norms[ij] = np.sum(wt_ij)

    return slices, weights, norms


def _apply_grid_weights(z, slices, weights, norms):
    """
    helper function to apply the grid weights to an input array z,
    which is some 2D field from the GFDL.
    The slices, weights, norms, are the returns from _compute_grid_weights()
    above, and they contain the indexing to extract the (nframe, 8) FOVs.

    returns both the per-FOV mean and std dev in a (nframe, 8) array.
    """

    shape2D = norms.shape
    z_mean = np.zeros(shape2D)
    z_var = np.zeros(shape2D)

    for ij in np.ndindex(shape2D):

        si, sj = slices[ij]
        wt_ij = weights[ij]
        # some might be None, if it straddles the lon wrap.
        if si:
            z_mean[ij] = np.sum(wt_ij * z[si, sj]) / norms[ij]
            z_var[ij] = np.sum(wt_ij * (z[si, sj]-z_mean[ij])**2) / norms[ij]

    z_stdv = np.sqrt(z_var)

    return z_mean, z_stdv


def _apply_grid_weights_profile_var(ncv, slices, weights, norms):
    """
    similar to _apply_grid_weights() above, but instead of a single
    2D ndarray, it is applied to a netCDF variable that is 3D.
    """

    shape2D = norms.shape
    nlev = ncv.shape[0]

    z_mean = np.zeros(shape2D + (nlev,))
    z_stdv = np.zeros(shape2D + (nlev,))

    for k in range(nlev):
        tmp = _apply_grid_weights(ncv[k,:,:], slices, weights, norms)
        z_mean[:,:,k] = tmp[0]
        z_stdv[:,:,k] = tmp[1]

    return z_mean, z_stdv


def compute_fullspectra(gridweight_file, sim_coef_fstr):
    """
    from the stored gridweights file, given a path to the raw
    PCscore output from the GFDL-PCRTM run, recompute the full
    resolution spectra.

    The input sim_coef_fstr should be a format string where the
    latitude row number can be input (recall the PCscore data was
    stored per-latitude row)

    Internally loads the PC data, from a hardcoded path (local subdir
    data in this project repository)

    returns data dictionary with keys:
    wavenum: 1D wavenumber vector, shape (5421,)
    radiance: 3D radiance array, shape (along track, cross track, 5421)
    radiance_stdv: same shape as radiance

    radiance will be the mean radiance spectrum, over the grouping of
    grid cells and grid weights specified in the gridweight file, for
    each atrack-xtrack scene.
    radiance_stdv is the std deviation over those same grid cells.
    """
    pcdat = PCRTM_utils.read_PC_file('../../data/PCRTM_Pcnew_id2.dat')

    slices, weights, norms = _load_grid_weights(gridweight_file)
    shape2D = slices.shape

    dat = {}
    dat['wavenum'] = np.arange(50, 2760.01, 0.5)
    shape3D = shape2D + dat['wavenum'].shape
    dat['radiance'] = np.zeros(shape3D)
    dat['radiance_stdv'] = np.zeros(shape3D)

    # sorta hack to trick the profile variable apply functions to work
    # on these data (we treat the PCRTM spectra as a atmospheric profile)
    #
    # we cannot load the entire PCRTM resolution array, so just do it
    # for the small group of GFDL grid cells needed to map one
    # TIRS scene (at array index ij)
    # so we basically make another 3D array that has placeholder axes
    # for the lat/lon dimensions, and convert the slices to be local
    # to the small group.
    single_weights = np.zeros((1,1), dtype=np.object)
    single_norms = np.zeros((1,1))
    single_slices = np.zeros((1,1), dtype=np.object)
    single_slices[0,0] = (slice(None), slice(None))

    t0 = datetime.now()

    for i,j in np.ndindex(shape2D):

        # 500 frames is approx 2 hour of run time, I think.
        if (j==0) and (i%500 == 0):
            dt = datetime.now()-t0
            print('{:4d} {:1d} {:s}'.format(i,j,str(dt)))

        # for failed vertices, skip the calculation and use FillValue.
        if norms[i,j] == 0:
            dat['radiance'][i,j,:] = -9999
            dat['radiance_stdv'][i,j,:] = -9999
            continue

        si, sj = slices[i,j]
        PCRTM_rad = load_PCRTM_spectra(si, sj, sim_coef_fstr, pcdat)

        # change shape (a,b,5421) to (5421,a,b)
        PCRTM_radr = np.rollaxis(PCRTM_rad, 2, 0)
        single_norms[0,0] = norms[i,j]
        single_weights[0,0] = weights[i,j]
        # output is shaped (1,1,5421)
        PCRTM_rad_mean, PCRTM_rad_stdv = _apply_grid_weights_profile_var(
            PCRTM_radr, single_slices, single_weights, single_norms)

        # Note - here we did have to split apart the tuple to have this work
        # correctly. We could not just use ij as the loop variable, as done in
        # many other places in this module.
        # (mixing a tuple and the slice for dimen 3 does something different)
        dat['radiance'][i,j,:] = PCRTM_rad_mean[0,0]
        dat['radiance_stdv'][i,j,:] = PCRTM_rad_stdv[0,0]

    return dat


def load_PCRTM_spectra(si, sj, coefs_fstr, pcdat):
    
    nPC_perband = [100,100,100,100]
    nw = 5421

    ni = si.stop - si.start
    nj = sj.stop - sj.start

    irange = range(si.start, si.stop)
    jrange = range(sj.start, sj.stop)

    PCRTM_rad = np.zeros((ni, nj, nw))

    for i in irange:
        rad_i = PCRTM_utils.radiance_from_PCscores(
            pcdat, coefs_fstr.format(i+1), recslice=sj)
        PCRTM_rad[i-si.start,:,:] = rad_i

    return PCRTM_rad

        
def orbit_extraction(tdat, sim_file_fstr, weights_filename=None,
                     zweights_filename=None):
    """
    extracts the GFDL fields, according to a orbit track data dictionary
    (tdat), e.g. from orbit_tracks.load_CALIOP_geo(),

    Parameters
    ----------
    tdat : dict
        Dictionary containing the orbit track data. See output from
        orbit_tracks.load_CALIOP_geo().
    sim_file_fstr : str
        format string, that when combined with a channel number, gives
        the full path and filename to a GFDL radiance at TIRS channel.
        Example:
        "/data/users/mmm/GFDL_radiance_simulation_conversion/<sim_subdir>/
             TIRS_radiance_20160801_03UTC_{0:02d}.h5"
    weights_filename : str or None
        if specified, a filename for an intermediate h5 file that will contain
        the weights, slices, norms arrays.
        If None (the default), these arrays are not saved.
    zweights_filename : str or None
        same as weights_filename, but for the grids derived for the surface
        topography file (which is a different, higher spatial resolution grid.)

    Returns
    -------
    adat : dict
        The extracted NWP analysis data. The key names matching the file_specs
        in the JSON file for the Aux-Met data stream.
    rdat : dict
        The extracted radiance data. The key names match the file_specs in the
        JSON file for the Radiance data stream.

    """

    # surf files are 300 MB each, so we can just load them entirely
    with netCDF4.Dataset(_GFDL_data_fstr.format('ps', 'ps'), 'r') as nc:
        lat = nc['grid_yt'][:]
        lon = nc['grid_xt'][:]
        ps = nc['ps'][:]
        # Pa -> hPa conversion
        ps *= 0.01

    with netCDF4.Dataset(_GFDL_data_fstr.format('ts', 'ts'), 'r') as nc:
        ts = nc['ts'][:]

    with netCDF4.Dataset(_GFDL_topo_file, 'r') as nc:
        latz = nc['lat'][:]
        lonz = nc['lon'][:]
        latz_bnds = nc['lat_bnds'][:]
        lonz_bnds = nc['lon_bnds'][:]
        zs = nc['orog_filt'][:]

    dlat = lat[1]-lat[0]
    clat = np.r_[-90.0, lat+0.5*dlat]
    dlon = lon[1]-lon[0]
    clon = np.r_[  0.0, lon+0.5*dlon]

    # the bin edges are kept in [n,2] shaped arrays, such that 
    # lat_bnds[k,0] == lat_bnds[k-1,1]
    # combine these into a single clat array like the others
    dlatz = latz[1] - latz[0]
    dlonz = lonz[1] - lonz[0]
    clatz = np.r_[ latz_bnds[0,0], latz_bnds[:,1] ]
    clonz = np.r_[ lonz_bnds[0,0], lonz_bnds[:,1] ]

    bdry_i = [0, 1, 2, 3, 0]

    slices, weights, norms = _compute_grid_weights(
        tdat['latitude_vertices'], tdat['longitude_vertices'], clat, clon)

    # different slices, weights, for z (the altitude), since this
    # is on a different (and much higher) spatial resolution.
    slices_z, weights_z, norms_z = _compute_grid_weights(
        tdat['latitude_vertices'], tdat['longitude_vertices'], clatz, clonz)

    if weights_filename is not None:
        _store_grid_weights(slices, weights, norms, weights_filename)
    if zweights_filename is not None:
        _store_grid_weights(slices_z, weights_z, norms_z,
                            zweights_filename)

    ps_mean, ps_stdv = _apply_grid_weights(ps, slices, weights, norms)
    ts_mean, ts_stdv = _apply_grid_weights(ts, slices, weights, norms)
    alt_mean, alt_stdv = _apply_grid_weights(zs, slices_z, weights_z, norms_z)

    # "free" memory
    ps = 0
    ts = 0
    zs = 0

    # now do the water vapor and temp profiles, one level at a time.
    # this is on the same grid as the ps/ts data, so we can reuse those weights.
    shape2D = tdat['latitude'].shape

    with netCDF4.Dataset(_GFDL_data_fstr.format('q','q_plev'), 'r') as nc:
        # required due to bogus "missing_value" attribute
        nc.set_auto_mask(False)
        # note that the pressure levels are constant in this model, so we just
        # copy out the constant array via broadcasting it.
        plev = nc['plev'][:]
        nlev = plev.shape[0]
        p_prof = np.zeros(shape2D+(nlev,))
        p_prof[:] = plev[np.newaxis, np.newaxis, :]
        q_prof_mean, q_prof_stdv = _apply_grid_weights_profile_var(
            nc['q_plev'], slices, weights, norms)

    with netCDF4.Dataset(_GFDL_data_fstr.format('t','t_plev'), 'r') as nc:
        # required due to bogus "missing_value" attribute
        nc.set_auto_mask(False)
        t_prof_mean, t_prof_stdv = _apply_grid_weights_profile_var(
            nc['t_plev'], slices, weights, norms)

    with netCDF4.Dataset(_GFDL_data_fstr.format('qi','qi_plev'), 'r') as nc:
        # required due to bogus "missing_value" attribute
        nc.set_auto_mask(False)
        qi_prof_mean, qi_prof_stdv = _apply_grid_weights_profile_var(
            nc['qi_plev'], slices, weights, norms)

    with netCDF4.Dataset(_GFDL_data_fstr.format('ql','ql_plev'), 'r') as nc:
        # required due to bogus "missing_value" attribute
        nc.set_auto_mask(False)
        ql_prof_mean, ql_prof_stdv = _apply_grid_weights_profile_var(
            nc['ql_plev'], slices, weights, norms)

    # kg/kg to g/kg.
    q_prof_mean *= 1e3
    q_prof_stdv *= 1e3
    qi_prof_mean *= 1e3
    qi_prof_stdv *= 1e3
    ql_prof_mean *= 1e3
    ql_prof_stdv *= 1e3

    # repeat for TIRS radiances.
    nchannel = 63
    rad_mean = np.zeros(shape2D + (nchannel,))
    rad_stdv = np.zeros(shape2D + (nchannel,))
    rad_unc = np.zeros(shape2D + (nchannel,))
    # initial detector flag as all "bad". flip to zero when there
    # is TIRS channel radiance simulated.
    detector_flag = np.zeros((8,nchannel), np.uint16) - 1
    detector_ID = np.zeros((8,nchannel), np.uint8)
    detector_ID[:] = np.arange(1,64).astype(np.uint8)
    wavelen = np.zeros((8,nchannel))

    # this is sort of gross, but derives the channel mapping based on what
    # is actually on disk. There must be 54 of them, otherwise a problem.
    for c in range(63):
        rad_file = sim_file_fstr.format(c+1)
        try:
            with h5py.File(rad_file,'r') as h:
                rad_c = h['rad'][:]
        except OSError:
            continue

        tmp = _apply_grid_weights(rad_c, slices, weights, norms)
        rad_mean[:,:,c] = tmp[0]
        rad_stdv[:,:,c] = tmp[1]
        detector_flag[:,c] = 0

    # extract the NEDR from the SRF file, then broadcast it to the rad
    # uncertainty array.
    SRF_file = '/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.03_2019-10-30.nc'

    with netCDF4.Dataset(SRF_file, 'r') as nc:
        center_wavelens = (
            nc['channel_wavelen1'][:] +
            nc['channel_wavelen2'][:] )
        NEDR = nc['NEDR'][:]
    rad_unc[:] = NEDR[np.newaxis, np.newaxis, :]

    # first, create the analysis data dictionary
    adat = {}

    adat['analysis_name'] = np.array(['GFDL_3km'])
    adat['analysis_id'] = np.array([1])

    FillValue = -9999

    adat['altitude'] = alt_mean
    adat['altitude_stdev'] = alt_stdv
    adat['surface_pressure'] = ps_mean
    adat['surface_temp'] = ts_mean
    adat['pressure_profile'] = p_prof
    adat['temp_profile'] = t_prof_mean
    adat['wv_profile'] = q_prof_mean

    # altitude is the only one on a different grid, which had
    # a separate norms array (I think these end up being floating point
    # equal, in the end, though).
    z_msk = norms_z == 0
    if np.any(z_msk):
        adat['altitude'][z_msk] = FillValue
        adat['altitude_stdev'][z_msk] = FillValue

    z_msk = norms == 0
    if np.any(z_msk):
        adat['surface_pressure'][z_msk] = FillValue
        adat['surface_temp'][z_msk] = FillValue
        adat['pressure_profile'][z_msk,:] = FillValue
        adat['temp_profile'][z_msk,:] = FillValue
        adat['wv_profile'][z_msk,:] = FillValue

    adat['skin_temp'] = adat['surface_temp'].copy()

    # Now, radiance stuff.
    rdat = {}
    rdat['spectral_radiance'] = rad_mean
    rdat['spectral_radiance_unc'] = rad_unc
    if np.any(z_msk):
        rdat['spectral_radiance'][z_msk,:] = FillValue
        rdat['spectral_radiance_unc'][z_msk,:] = FillValue

    rdat['detector_flag'] = detector_flag
    rdat['detector_ID'] = detector_ID
    rdat['radiance_quality_flag'] = np.zeros(shape2D, np.int16)

    # finally, extra Sim stuff: primarily, sub-FOV standard deviations
    # and condensed water
    sdat = {}
    sdat['spectral_radiance_stdv'] = rad_stdv
    sdat['pressure_profile'] = adat['pressure_profile'][0,0,:]
    sdat['qi_mean'] = qi_prof_mean
    sdat['qi_stdv'] = qi_prof_stdv
    sdat['ql_mean'] = ql_prof_mean
    sdat['ql_stdv'] = ql_prof_stdv
    sdat['ts_mean'] = ts_mean
    sdat['ts_stdv'] = ts_stdv
    sdat['t_mean'] = t_prof_mean
    sdat['t_stdv'] = t_prof_stdv
    sdat['q_mean'] = q_prof_mean
    sdat['q_stdv'] = q_prof_stdv

    return adat, rdat, sdat
