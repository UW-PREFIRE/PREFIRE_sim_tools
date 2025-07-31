"""
functions to convert the PCRTM forward run (in PC scores)
into TIRS radiance

This repackages the PC scores grouped by latitude bands into
single global data arrays, one for each TIRS spectral band.
"""

import os.path
import glob
import shutil
import multiprocessing
import datetime
import itertools
import warnings

import numpy as np
import tables
import h5py

from scipy.io.matlab import loadmat

import PREFIRE_sim_tools

def melt_pond_testruns(clearsky=False):
    """
    simple hard-coded function to insert the melt-pond
    simulation output into the corresponding GFDL global arrays.

    This does a loop over the 3 timesteps (hours), the two SRF,
    and two provided simulations ("origin" and "with_pond");
    for each case, we copy the corresponding data from the
    original runs, and simply overwrite the lat-lon covered by
    these two simulations.
    The two spatial locations (greenland, antarctica) are both merged
    into the global array.
    """

    # for slice definitions, I just hand copied X.'s ancillary
    # information in the readmes.
    # Store in a dictionary with key equal to the location name
    # (which is also used in the filenames), to make it easy to loop.

    lon_slices = {}
    lat_slices = {}

    # Greenland: 2x2 deg with center at lat0,lon0 = 70, -40    
    lon_slices['greenland'] = slice(10889, 10957)
    lat_slices['greenland'] = slice( 5427,  5495)

    # Antarctica: 2x2 deg with center at lat0,lon0 = -75, 65
    lon_slices['Antarctic'] = slice( 2185,  2253)
    lat_slices['Antarctic'] = slice(  478,   546)
 
    if clearsky:
        input_fstr = (
            '/data/users/xc/GFDL_radiance_simulation/data_for_pond_study_clrsky/'+
            'GFDL_{ymd:s}_{hour:02d}UTC_{location:s}_{sim:s}_clrsky.mat')
    else:
        input_fstr = (
            '/data/users/xc/GFDL_radiance_simulation/data_for_pond_study/'+
            'GFDL_{ymd:s}_{hour:02d}UTC_{location:s}_{sim:s}.mat')
    # write to the same location put in a new directory with "sim"
    # as a suffix. (origin or with_pond)
    if clearsky:
        outdir_fstr = (
            '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
            'TIRS_SRF_v{srf_ver:s}_{sim:s}_clrsky/{hour:02d}UTC')
    else:
        outdir_fstr = (
            '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
            'TIRS_SRF_v{srf_ver:s}_{sim:s}/{hour:02d}UTC')

    # the original files.
    if clearsky:
        orig_fstr = (
            '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
            'TIRS_SRF_v{srf_ver:s}_clrsky/{hour:02d}UTC/TIRS_radiance_{ymd:s}_{hour:02d}UTC_c{ch:02d}.h5')
    else:
        orig_fstr = (
            '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
            'TIRS_SRF_v{srf_ver:s}/{hour:02d}UTC/TIRS_radiance_{ymd:s}_{hour:02d}UTC_c{ch:02d}.h5')

    ymd = '20160801'
    # only 03UTC exists for clear sky
    #timesteps = (3,6,9)
    timesteps = (3,)

    srf_dir = '/data/rttools/TIRS_ancillary'
    SRF_file_fstr = os.path.join(
        srf_dir, 'PREFIRE_SRF_v{0:s}_2020-02-21.nc')

    SRF_vers = ('0.09.1', '0.09.2')
    locations = ('Antarctic', 'greenland')
    sims = ('origin', 'with_pond')

    SRFdata = None
    all_vars = (timesteps, SRF_vers, sims)

    for hour, SRF_ver, sim in itertools.product(*all_vars):
        
        print('creating output file for hour '+
              '{0:02d} sim {1:s} SRF {2:s}'.format(hour, sim, SRF_ver))

        outdir = outdir_fstr.format(srf_ver=SRF_ver, sim=sim, hour=hour)
        SRFfile = SRF_file_fstr.format(SRF_ver)

        radc = {}

        for loc in locations:
            in_file = input_fstr.format(
                ymd=ymd, hour=hour, location=loc, sim=sim)

            m = loadmat(in_file)
            npixel = np.prod(m['rad'].shape[:2])

            # these files have PCRTM spectra converted to W instead
            # of the conventional mW.
            # downstream assumes mW, so convert here.
            y = m['rad'] * 1e3
            # convert (nlat, nlot, nspec) to (nspec, nlat*nlon)
            y = np.rollaxis(y,2,0).reshape((5421,npixel))
            wl, wlr, rad_c, SRFdata = PREFIRE_sim_tools.TIRS_srf.apply_SRF_wngrid(
                y, SRFfile=SRFfile, SRFdata=None)
            radc[loc] = rad_c.reshape((63,) + m['rad'].shape[:2])

        for c in range(63):

            if np.all(np.isnan(radc['greenland'][c,:])):
                continue

            # note the channel index switch to 1-indexed here.
            orig_f = orig_fstr.format(
                ymd=ymd, srf_ver=SRF_ver, hour=hour, ch=c+1)

            new_f = os.path.join(outdir, os.path.split(orig_f)[-1])
            shutil.copyfile(orig_f, new_f)

            with h5py.File(new_f, 'a') as h:
                for loc in locations:
                    lon_s = lon_slices[loc]
                    lat_s = lat_slices[loc]
                    h['rad'][lat_s, lon_s] = radc[loc][c,:,:]


def batch_serial_loop():
    """
    simple batch loop to run the different timesteps and TIRS SRF
    Those are hardcoded.

    this does the global sims, and must be run before the melt pond
    function above (which sort of patches-in the melt pond areas)
    """

    ymd = '20160801'
    # clrsky only for hour (3,) right now
    timesteps = (3,6,9)
    src_topdir = '/data/users/xc/GFDL_radiance_simulation/data'
    dst_topdir = '/data/users/mmm/GFDL_radiance_simulation_conversion'
    srf_dir = os.path.expanduser('~/projects/PREFIRE_sim_tools/data/')

    SRF_file_fstr = os.path.join(
        #srf_dir, 'PREFIRE_SRF_v{0:s}_2020-02-21_PCRTM_grid.nc')
        srf_dir, 'PREFIRE_SRF_v{0:s}_360_2021-03-28_PCRTM_grid.nc')

    # probably don't need to do both SRFs anymore
    #SRF_vers = '0.09.1', '0.09.2'
    #SRF_vers = ('0.09.2',)
    SRF_vers = ('0.10.4',)

    for hour, SRF_ver in itertools.product(timesteps, SRF_vers):

        SRF_file = SRF_file_fstr.format(SRF_ver)

        # do the all sky first
        src_dir_fstr = os.path.join(src_topdir, '{0:02d}UTC')
        src_dir = src_dir_fstr.format(hour)
        dst_dir_fstr = os.path.join(dst_topdir, 'TIRS_SRF_v{0:s}', '{1:02d}UTC')
        dst_dir = dst_dir_fstr.format(SRF_ver, hour)
        print('computing timestep = '+ str(hour))
        print('  SRF_file = '+SRF_file)
        print('  src_dir = '+src_dir)
        print('  dst_dir = '+dst_dir)
        serial_loop(ymd, hour, SRFfile=SRF_file, clearsky=False,
                    src_dir=src_dir, dst_dir=dst_dir)
        print()

        # clear sky exists for 03 only here.
        if hour == 3:
            src_dir_fstr = os.path.join(src_topdir, '{0:02d}UTC_clrsky')
            src_dir = src_dir_fstr.format(hour)
            dst_dir_fstr = os.path.join(
                dst_topdir, 'TIRS_SRF_v{0:s}_clrsky', '{1:02d}UTC')
            dst_dir = dst_dir_fstr.format(SRF_ver, hour)
            print('computing clearsky  timestep = '+ str(hour))
            print('  SRF_file = '+SRF_file)
            print('  src_dir = '+src_dir)
            print('  dst_dir = '+dst_dir)
            serial_loop(ymd, hour, SRFfile=SRF_file, clearsky=True,
                        src_dir=src_dir, dst_dir=dst_dir)
            print()

    # repeats for 15h timestep.
    # source dir is slightly different.
    timesteps = (15,)
    src_topdir = '/data/users/xc/GFDL_radiance_simulation_globe/data'

    for hour, SRF_ver in itertools.product(timesteps, SRF_vers):

        SRF_file = SRF_file_fstr.format(SRF_ver)

        # do the all sky first
        src_dir_fstr = os.path.join(src_topdir, '{0:02d}UTC')
        src_dir = src_dir_fstr.format(hour)
        dst_dir_fstr = os.path.join(dst_topdir, 'TIRS_SRF_v{0:s}', '{1:02d}UTC')
        dst_dir = dst_dir_fstr.format(SRF_ver, hour)
        print('computing timestep = '+ str(hour))
        print('  SRF_file = '+SRF_file)
        print('  src_dir = '+src_dir)
        print('  dst_dir = '+dst_dir)
        serial_loop(ymd, hour, SRFfile=SRF_file, clearsky=False,
                    src_dir=src_dir, dst_dir=dst_dir)
        print()

        # clear sky exists for 15 only here.
        if hour == 15:
            src_dir_fstr = os.path.join(src_topdir, '{0:02d}UTC_clrsky')
            src_dir = src_dir_fstr.format(hour)
            dst_dir_fstr = os.path.join(
                dst_topdir, 'TIRS_SRF_v{0:s}_clrsky', '{1:02d}UTC')
            dst_dir = dst_dir_fstr.format(SRF_ver, hour)
            print('computing clearsky  timestep = '+ str(hour))
            print('  SRF_file = '+SRF_file)
            print('  src_dir = '+src_dir)
            print('  dst_dir = '+dst_dir)
            serial_loop(ymd, hour, SRFfile=SRF_file, clearsky=True,
                        src_dir=src_dir, dst_dir=dst_dir)
            print()




def serial_loop(
        ymd, hour, SRFfile=None, imax=6144,
        clearsky = False,
        src_dir='/data/users/xc/GFDL_radiance_simulation/data',
        dst_dir='/data/users/mmm/GFDL_radiance_simulation_conversion'):
    """
    serial loop over all files for one global snapshot.
    Loops are over the lat bands (For the source PCRTM score files).
    Writes out global arrays at TIRS channels.

    keywords allow some flexibility in terms of output location
    and number of lat slices.

    reads from /data/users/xc/GFDL_radiance_simulation/data

    writes to /data/users/mmm/GFDL_radiance_simulation_conversion/
    by default.

    This takes on order 15 minutes to do the entire global dataset.
    """

    nPC_perband = [100,100,100,100]

    if clearsky:
        fstr = os.path.join(
            src_dir, 'PC_GFDL_{ymd:s}_{hour:02d}UTC_ilat{index:04d}_clrsky.par_spec.ieee')
    else:
        fstr = os.path.join(
            src_dir, 'PC_GFDL_{ymd:s}_{hour:02d}UTC_ilat{index:04d}.par_spec.ieee')

    t0 = datetime.datetime.now()
    imax = 6144
    idiv = 256
    jmax = 12288

    # for testing = takes ~ 10 seconds
    #imax = 64
    #idiv = 4

    Nc = 63

    rad = np.zeros((imax, jmax, Nc), np.float32)
    lat = np.zeros((imax, jmax), np.float64)
    lon = np.zeros((imax, jmax), np.float64)
    SRFdata = None

    for i in range(imax):

        if (i % idiv) == 0:
            print('{0:5d} of {1:5d} etime: {2:s}'.format(
                i, imax, str(datetime.datetime.now()-t0)))

        sdat = PREFIRE_sim_tools.PCRTM_utils.read_score_file(
            nPC_perband, fstr.format(ymd=ymd, hour=hour, index=i+1))

        S = np.concatenate(sdat['PCscore'], axis=1)

        wl, wlr, rad_i, SRFdata = PREFIRE_sim_tools.TIRS_srf.apply_SRF_PCscore(
            S, SRFdata=SRFdata, spec_grid='wl', SRFfile=SRFfile)

        rad[i,:,:] = rad_i.T
        lat[i,:] = sdat['lat']
        lon[i,:] = sdat['lon']

        if i == 0:
            channel_mask = np.all(rad_i==0, axis=1)

    output_name_root = os.path.join(
        dst_dir,
        'TIRS_radiance_{ymd:s}_{hour:02d}UTC'.format(ymd=ymd,hour=hour))

    write_converted_output(rad, lat, lon, channel_mask, output_name_root)
    

def write_converted_output(rad, lat, lon, channel_mask, outfile_root):

    FilterObj = tables.Filters(complevel=1, complib='zlib')

    shape2D = rad.shape[:2]

    with tables.open_file(outfile_root+'_latlon.h5', 'w') as h:
        h.create_carray('/', 'lat', tables.Float64Atom(), shape2D, 
                        filters=FilterObj)
        h.create_carray('/', 'lon', tables.Float64Atom(), shape2D, 
                        filters=FilterObj)
        h.root.lat[:] = lat
        h.root.lon[:] = lon

    for c in range(63):

        if channel_mask[c]:
            continue
        
        with tables.open_file(outfile_root+'_c{0:02d}.h5'.format(c+1), 'w') as h:
            h.create_carray('/', 'rad', tables.Float32Atom(), shape2D, 
                            filters=FilterObj)
            h.root.rad[:] = rad[:,:,c]

    # write out a full channel set as well.
    # this reproduces what is done in join_channels().
    # approx wavelengths
    # this takes a lot of memory - might not be wise?

    wdelta = 6.0/7.0
    wgrid = np.arange(1,64) * wdelta
    c_idx = np.nonzero(channel_mask==0)[0]
    wavelen = wgrid[c_idx-1]
    # note this used to be off by one - I forgot to do the +1
    # to convert this to 1-ordered channel positions.
    c_number = c_idx + 1

    filename3D = outfile_root + '.h5'

    shape3D = shape2D + c_idx.shape
    rad_valid_channels = rad[:,:,c_idx]

    with tables.open_file(filename3D, 'w') as h:

        h.create_carray('/', 'rad', tables.Float32Atom(), 
                        shape3D, filters=FilterObj)

        h.root.rad[:] = rad_valid_channels

        h.create_carray('/', 'lat', tables.Float64Atom(),
                        lat.shape, filters=FilterObj)
        h.create_carray('/', 'lon', tables.Float64Atom(), 
                        lon.shape, filters=FilterObj)
        h.root.lat[:] = lat
        h.root.lon[:] = lon

        h.create_array('/', 'wavelen', wavelen)
        h.create_array('/', 'channel_number', c_number)


def _patch_channel_number_output(TIRS_file):
    """
    the updated write_converted_output() function had a bug, I was not
    adding 1 to the channel_number output before writing it.
    This is a one-use function to patch that array, so I don't have
    to re-run the whole thing just to fix thise index numbers.
    """

    with h5py.File(TIRS_file, 'r+') as h:
        c_idx = h['channel_number'][:]
        if c_idx[0] == 3:
            print('patching channel numbers')
            c_idx += 1
            h['channel_number'][:] = c_idx

def example_plot(fignum=10):
    """
    make some simple pcolormesh plots, using a sparse ::10 slice to reduce
    the data size sent to matplotlib.

    """

    import matplotlib.pyplot as plt

    fig = plt.figure(fignum, figsize=(12,8))
    fig.clf()

    axs = [fig.add_subplot(3,2,p) for p in range(1,7)]
    chn_i = [i+1 for i in [10, 11, 18, 27, 33, 50]]

    with h5py.File('../data/PREFIRE_SRF_v0.03_2019-10-30.nc') as h:
        wl_grid = h['wavelen'][:]
        wn = 1e4/wl_grid
        SRF = h['srf'][:]

    wlc = np.zeros(63)
    wnc = np.zeros(63)
    for n in range(63):
        if SRF[:,n].sum() == 0:
            continue
        wlc[n] = np.trapz(SRF[:,n]*wl_grid, wl_grid)/np.trapz(SRF[:,n], wl_grid)
        wnc[n] = np.trapz(SRF[:,n]*wn, wn)/np.trapz(SRF[:,n], wn)

    islice = slice(None, None, 10)
    jslice = slice(None, None, 10)
    llfile = (
        '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
        'test_TIRS_radiance_latlon.h5')
    with h5py.File(llfile, 'r') as h:
        lat = h['lat'][islice, jslice]
        lon = h['lon'][islice, jslice]

    # hardcoded roll size - need to figure out how to compute from the sizes...
    roll_shift = 614
    lon = np.roll(lon, roll_shift, 1)
    fstr = (
        '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
        'test_TIRS_radiance_c{0:02d}.h5')

    for i in range(6):

        with h5py.File(fstr.format(chn_i[i]), 'r') as h:
            rad = h['rad'][islice, jslice]
        rad = np.roll(rad, roll_shift, 1)
        print('Making ad hoc x 10 scaling to correct units')
        rad *= 10

        pm = axs[i].pcolormesh(lon, lat, rad, cmap='inferno')
        cb = plt.colorbar(pm, ax=axs[i])
        axs[i].set_title('TIRS channel {0:2d}, '.format(chn_i[i])+
                         'wl={0:5.1f} um,  wn={1:4.0f} cm-1'.format(
                             wlc[chn_i[i]-1], wnc[chn_i[i]-1]))
        if i == 3:
            cb.ax.set_ylabel('Radiance [W/(m2 sr um)]', rotation=270,
                             va='bottom')
    axs[2].set_ylabel('Latitude')
    axs[4].set_xlabel('Longitude')
    axs[5].set_xlabel('Longitude')


def join_channels(data_dir):
    """
    helper to concatenate the channels back into a single 3D array.
    this makes certain analysis/plotting a little easier.
    """

    flist = glob.glob(os.path.join(
        data_dir,'TIRS_radiance_????????_??UTC_c??.h5'))
    flist.sort()

    rad_list = []
    c_list = []
    for f in flist:
        c_list.append( int(f[-5:-3]) )
        with h5py.File(f, 'r') as h:
            rad_list.append(h['rad'][:])

    filename3D = flist[0].replace('_c04.h5', '.h5')

    filenameLL = flist[0].replace('c04.h5', 'latlon.h5')
    # this doesnt change, so we can maybe just hardcode this to one file.
    # for some reason the newer sims were not creating this file?
    #filenameLL = (
    #    '/data/users/mmm/GFDL_radiance_simulation_conversion/'+
    #    'test_TIRS_radiance_latlon.h5')

    with h5py.File(filenameLL, 'r') as h:
        lat = h['lat'][:]
        lon = h['lon'][:]

    # approx wavelengths
    wdelta = 6.0/7.0
    wgrid = np.arange(1,64) * wdelta
    c_idx = np.array(c_list)
    wavelen = wgrid[c_idx-1]

    FilterObj = tables.Filters(complevel=1, complib='zlib')
    with tables.open_file(filename3D, 'w') as h:

        # we don't have to make the full 3D array in memory first,
        # but it is much faster to create the compressed h5 array that
        # way rather than filling it one channel at a time.
        shape3D = rad_list[0].shape + (len(c_list),)
        rad3D = np.zeros(shape3D, dtype=np.float32)
        for i in range(len(c_list)):
            rad3D[:,:,i] = rad_list[i]

        h.create_carray('/', 'rad', tables.Float32Atom(), 
                        shape3D, filters=FilterObj)
        h.root.rad[:] = rad3D

        h.create_carray('/', 'lat', tables.Float64Atom(),
                        lat.shape, filters=FilterObj)
        h.create_carray('/', 'lon', tables.Float64Atom(), 
                        lon.shape, filters=FilterObj)
        h.root.lat[:] = lat
        h.root.lon[:] = lon

        h.create_array('/', 'wavelen', wavelen)
        h.create_array('/', 'channel_number', c_idx)
