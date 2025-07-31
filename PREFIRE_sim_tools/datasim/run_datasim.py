"""
Script to run datasim over a set of orbits given by CALIOP L1b data.
This is not generalized - it assumes it will run in the datasim subdirectory
of PREFIRE_sim_tools, and has various hardcoded paths to data output and input.

run from ipython with %run, or from the shell with 'python run_datasim.py'

"""

from PREFIRE_sim_tools.datasim import GFDL_extraction
from PREFIRE_sim_tools.datasim import orbit_tracks
from PREFIRE_sim_tools.datasim import file_creation
from PREFIRE_sim_tools.utils import level_interp

from glob import glob
import numpy as np
from datetime import datetime
import os.path

import netCDF4

import multiprocessing

def pressure_interp(dat):
    """
    small wrapper to call the pressure interp.
    """
    # GFDL has fixed P levels, so we only need to pick one.
    p = dat['pressure_profile'][0,0,:]
    intp = np.loadtxt('../../data/plevs101.txt')
    dat_intp = level_interp.fixed_pressure_interp(dat, p, intp)
    dat_intp['pressure_profile'] = intp
    return dat_intp

def _wrap_run(inputs):
    run(*inputs)

def batch_run_meltpond(pool_size=6):
    lon_shifts_orig =  [0.0]*8 + [6.0, -106.0, 226.0, -17.0]
    lon_shifts_pond = [None]*8 + [6.0, -106.0, 226.0, -17.0]
    inputs_list = [
        # for the clrsky, only hour 3 is currently run.
        ('TIRS_SRF_v0.09.1_origin_clrsky', 3, lon_shifts_orig),
        ('TIRS_SRF_v0.09.1_with_pond_clrsky', 3, lon_shifts_pond),
        ('TIRS_SRF_v0.09.2_origin_clrsky', 3, lon_shifts_orig),
        ('TIRS_SRF_v0.09.2_with_pond_clrsky', 3, lon_shifts_pond),
        #('TIRS_SRF_v0.09.1_origin', 3, lon_shifts_orig),
        #('TIRS_SRF_v0.09.1_origin', 6, lon_shifts_orig),
        #('TIRS_SRF_v0.09.1_origin', 9, lon_shifts_orig),
        #('TIRS_SRF_v0.09.1_with_pond', 3, lon_shifts_pond),
        #('TIRS_SRF_v0.09.1_with_pond', 6, lon_shifts_pond),
        #('TIRS_SRF_v0.09.1_with_pond', 9, lon_shifts_pond),
        #('TIRS_SRF_v0.09.2_origin', 3, lon_shifts_orig),
        #('TIRS_SRF_v0.09.2_origin', 6, lon_shifts_orig),
        #('TIRS_SRF_v0.09.2_origin', 9, lon_shifts_orig),
        #('TIRS_SRF_v0.09.2_with_pond', 3, lon_shifts_pond),
        #('TIRS_SRF_v0.09.2_with_pond', 6, lon_shifts_pond),
        #('TIRS_SRF_v0.09.2_with_pond', 9, lon_shifts_pond),
        ]
    
    P = multiprocessing.Pool(pool_size)
    P.map(_wrap_run, inputs_list)


def load_L1orbitsim(ncfile):
    """
    loads a PREFIRE L1 orbit sim, and then adapts the dictionary
    so that this will run with the rest of the code - fortunately I had
    set this up to look like L1b geo already, so not much to do.
    """

    tdat = {}
    vlist = ('time_UTC', 'latitude_vertices', 'longitude_vertices',
             'latitude', 'longitude', 'latitude_geoid', 'longitude_geoid', 
             'subsat_latitude', 'subsat_longitude', 'sat_altitude')
    with netCDF4.Dataset(ncfile, 'r') as nc:
        for v in vlist:
            tdat[v] = nc['Geometry'][v][:]
             
    # don't actually know how to do this at the moment, so make it Fill.
    tdat['solar_beta_angle'] = np.array([-9999.0])

    # none of these are populated yet
    vlist = ('sensor_zenith', 'sensor_azimuth',
             'solar_zenith', 'solar_azimuth', 'solar_distance')
    for v in vlist:
        tdat[v] = np.zeros(tdat['latitude'].shape) - 9999.0
    tdat['time_tai93'] = np.zeros(tdat['latitude'].shape[0]) - 9999.0

    return tdat

def _parse_time_UTC(tdat):
    """
    from the time_UTC array in tdat, derive the ymd and hms strings that
    will be used in the filename for the output product
    """

    t = tdat['time_UTC'][0]
    ymd = '{:04d}{:02d}{:02d}'.format(t[0], t[1], t[2])
    hms = '{:02d}{:02d}{:02d}'.format(t[3], t[4], t[5])

    return ymd, hms
    
def run_fullspec_calc(input_dir, output_dir, run_slice=None):
    """
    run the full spectrum calc over an already completed simulation run
    (at least, the gridweight files need to exist.)

    input directory is checked for SAT?_gridweights files; 
    for each gridweight file, an output file is created with the same
    gridding.

    Note: For the TIRS simulations, the output will depend on both the SRF
    and the GFDL simulation timestep (03UTC, etc), whereas the fullspec
    will only depend on the GFDL timestep. So, the output directory tree
    is a little different.

    I am planning something like the following:

    /data/users/mmm/datasim_output
        .
        |-TIRS_SRF_v0.10.4
        |---03UTC
        |---06UTC
        |---09UTC
        |-TIRS_SRF_v0.10.4_withpond (maybe - would only be for SAT1)
        |---03UTC
        |---06UTC
        |---09UTC
        |-FullPCRTM
        |---03UTC
        |---06UTC
        |---09UTC

    For future SRF updates, new subdirs would be created (TIRS_SRF_vN.NN),
    but the FullPCRTM would not change, assuming we keep reusing the same
    orbit sims.
    """

    input_files1 = glob(os.path.join(input_dir, 'SAT1_gridweights_*.h5'))
    input_files2 = glob(os.path.join(input_dir, 'SAT2_gridweights_*.h5'))
    input_files1.sort()
    input_files2.sort()

    output_files1 = []
    for f in input_files1:
        fonly = os.path.split(f)[-1]
        output_fonly = fonly.replace('SAT1_gridweights_',
                                     'SAT1_fullspectra_')
        output_fonly = output_fonly.replace('.h5', '.nc')
        output_files1.append(os.path.join(output_dir,output_fonly))

    output_files2 = []
    for f in input_files2:
        fonly = os.path.split(f)[-1]
        output_fonly = fonly.replace('SAT2_gridweights_',
                                     'SAT2_fullspectra_')
        output_fonly = output_fonly.replace('.h5', '.nc')
        output_files2.append(os.path.join(output_dir,output_fonly))

    # attempt to autocreate the format string for the PC score files...
    # this is likely to be a brittle problem point...
    tmp = os.path.split(input_dir)
    # this handles if the input_dir has a trailing pathsep.
    if tmp[1] == '':
        tmp = os.path.split(tmp[0])
    timestep = tmp[1]

    if timestep in ('03UTC', '06UTC', '09UTC'):
        sim_dir = '/data/users/xc/GFDL_radiance_simulation/data/'+timestep
    elif timestep == '15UTC':
        sim_dir = '/data/users/xc/GFDL_radiance_simulation_globe/data/'+timestep
    else:
        raise ValueError('Could not match timestamp from input_dir '+
                         'with known GFDL simuations')

    sim_coef_fstr = os.path.join(
        sim_dir, 'PC_GFDL_20160801_'+timestep+'_ilat{0:04d}.par_spec.ieee')

    input_files = (input_files1 + input_files2)
    output_files = (output_files1 + output_files2)

    if run_slice:
        input_files = input_files[run_slice]
        output_files = output_files[run_slice]

    nfiles = len(input_files)

    for n, (inf, outf) in enumerate(zip(input_files, output_files)):

        if os.access(outf, os.F_OK):
            print('skipping existing file: '+outf)
            continue

        t0 = datetime.now()
        print('Starting Run {:d} of {:d}'.format(n+1, nfiles))
        print('Input:  '+inf)
        print('output: '+outf)

        dat = GFDL_extraction.compute_fullspectra(inf, sim_coef_fstr)
        file_creation.write_data_fromspec(
            {'Anc-Fullspectra':dat}, 'file_specs.json', outf)

        dt = datetime.now()-t0
        print('etime: ' + str(dt))
        print('')


def run_L1orbitsim(sim_subdir, sim_hr):
    """
    run the sim over the data in a particular subdir.
    this is a copy of run(), but rewritten to work from the new
    orbit data produced by the PREFIRE_L1 orbit simulator.

    The code here is not factored well, so I just copy/pasted.

    Otherwise this works the same as run(), but without a lon_shifts

    """
    top_simdir = '/data/users/mmm/GFDL_radiance_simulation_conversion'
    top_outdir = '/data/users/mmm/datasim_output'

    hr_dir = '{0:02d}UTC'.format(sim_hr)

    Met_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'PREFIRE_SAT{sat:1d}_AUX-MET_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
    Rad_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'PREFIRE_SAT{sat:1d}_1B-RAD_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
    AncSim_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'PREFIRE_SAT{sat:1d}_ANC-SIM_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
    Wt_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'SAT{sat:1d}_gridweights_{ymd:s}{hms:s}_{granule:05d}.h5')
    zWt_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'SAT{sat:1d}_zgridweights_{ymd:s}{hms:s}_{granule:05d}.h5')

    sim_file_fstr = os.path.join(
        top_simdir, sim_subdir, hr_dir, 
        'TIRS_radiance_20160801_{0:02d}UTC'.format(sim_hr) +
        '_c{0:02d}.h5')

    orbit_dir = '/data/users/mmm/datasim_orbits'
    flist1 = glob(orbit_dir + '/PREFIRE_orbit_simrw_550_98_140_*.nc4')
    flist2 = glob(orbit_dir + '/PREFIRE_orbit_simrw_550_98_200_*.nc4')
    flist1.sort()
    flist2.sort()

    t0 = datetime.now()
    sat1granule = 0
    sat2granule = 0

    for m, f in enumerate(flist1 + flist2):

        # crude delineation between SAT1 and SAT2, based on RAAN, 
        # which is in the orbitsim filename. SAT1 = 140, SAT2 = 200
        if '140' in f:
            sat = 1
            sat1granule += 1
            granule = sat1granule
        elif '200' in f:
            sat = 2
            sat2granule += 1
            granule = sat2granule
        else:
            print('Cannot determine satellite number, assuming SAT1')
            sat = 1
            sat1granule += 1
            granule = sat1granule

        tdat = load_L1orbitsim(f)

        ymd, hms = _parse_time_UTC(tdat)

        Met_filename = Met_fstr.format(
            sat=sat, ymd=ymd, hms=hms, granule=granule)
        Rad_filename = Rad_fstr.format(
            sat=sat, ymd=ymd, hms=hms, granule=granule)
        AncSim_filename = AncSim_fstr.format(
            sat=sat, ymd=ymd, hms=hms, granule=granule)
        weights_filename = Wt_fstr.format(
            sat=sat, ymd=ymd, hms=hms, granule=granule)
        zweights_filename = zWt_fstr.format(
            sat=sat, ymd=ymd, hms=hms, granule=granule)

        if os.access(Rad_filename, os.R_OK):
            print('skipping file, already exists: '+Rad_filename)
            continue

        adat, rdat, sdat = GFDL_extraction.orbit_extraction(
            tdat, sim_file_fstr, weights_filename=weights_filename,
            zweights_filename=zweights_filename)

        adat_intp = pressure_interp(adat)

        # moves the altitude into Geometry (we have to extract this from
        # the Aux-Met data)
        for v in ('altitude', 'altitude_stdev'):
            tdat[v] = adat[v]

        print('Writing files:')
        print(Rad_filename)
        print(Met_filename)
        print(AncSim_filename)

        dat = {'Geometry':tdat, 'Aux-Met':adat_intp}
        file_creation.write_data_fromspec(dat, 'file_specs.json', Met_filename)

        dat = {'Geometry':tdat, 'Radiance':rdat}
        file_creation.write_data_fromspec(dat, 'file_specs.json', Rad_filename)

        dat = {'Geometry':tdat, 'Anc-Sim':sdat}
        file_creation.write_data_fromspec(dat, 'file_specs.json', AncSim_filename)

        dt = datetime.now()-t0
        print('Completed file ' + str(m+1) + ' elapsed time: '+str(dt))


def run(sim_subdir, sim_hr, lon_shifts):
    """
    run the sim over the data in a particular subdir.

    the sim_subdir and sim_hr control the input GFDL data source
    (e.g., what meteorology is used in there)

    The lon_shifts is a list, which also controls how many orbits are
    simulated. There are currently 8 possible orbits from the CALIOP 
    sample that I am using; these are used in a wrapping fashion.
    For example if there are 10 elements in lon_shifts then we would
    use orbits [0,1,2,3,4,5,6,7,0,1] to get to 10; the last two elements
    (that are wrapped back to CALIOP orbits 0, 1) will have +1 day added
    to the time.
    A value of None in lon shifts means that file is skipped.

    """
    top_simdir = '/data/users/mmm/GFDL_radiance_simulation_conversion'
    top_outdir = '/data/users/mmm/datasim_examples'

    hr_dir = '{0:02d}UTC'.format(sim_hr)

    Met_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'PREFIRE_SAT1_AUX-MET_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
    Rad_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'PREFIRE_SAT1_1B-RAD_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')
    AncSim_fstr = os.path.join(
        top_outdir, sim_subdir, hr_dir,
        'PREFIRE_SAT1_ANC-SIM_S00_R00_{ymd:s}{hms:s}_{granule:05d}.nc')

    sim_file_fstr = os.path.join(
        top_simdir, sim_subdir, hr_dir, 
        'TIRS_radiance_20160801_{0:02d}UTC'.format(sim_hr) +
        '_c{0:02d}.h5')

    flist = glob('/data/users/mmm/CALIOP_L1b_sample/CAL*.hdf')
    flist.sort()
    nfiles = len(flist)
    n_caliop_orbits = nfiles // 2

    if nfiles != 16:
        raise ValueError('This is hardcoded to work with 16 files!'+
                         ' The CALIOP sample dir contents are changed?')

    t0 = datetime.now()

    for m, lon_shift in enumerate(lon_shifts):

        if lon_shift is None:
            continue

        n = m % 8
        d = m // 8

        cal_time = flist[n*2].split('.')[-2]
        day_string = '{0:02d}'.format(int(cal_time[8:10]) + d)
        ymd = cal_time[:4] + cal_time[5:7] + day_string

        cal_time = flist[n*2].split('.')[-2][11:]
        hms = cal_time[:2] + cal_time[3:5] + cal_time[6:8]
        
        # concatenates the ascending & descending CALIOP passes into 1 orbit
        dat_asc = orbit_tracks.load_CALIOP_geo(flist[n*2])
        dat_des = orbit_tracks.load_CALIOP_geo(flist[n*2+1])
        dat = {}
        for k in dat_asc:
            dat[k] = np.concatenate([dat_asc[k], dat_des[k]])

        # apply shift to hit the 'melt ponds'
        for k in ('Subsatellite_Longitude', 'Longitude'):
            shifted_lon = dat[k] + lon_shift
            shifted_lon = np.mod(shifted_lon + 180, 360) - 180
            dat[k] = shifted_lon

        tdat = orbit_tracks.generate_test_orbit_from_CALIOP(dat)
        adat, rdat, sdat = GFDL_extraction.orbit_extraction(tdat, sim_file_fstr)

        adat_intp = pressure_interp(adat)

        # moves the altitude into Geometry (we have to extract this from
        # the Aux-Met data)
        for v in ('altitude', 'altitude_stdev'):
            tdat[v] = adat[v]

        Met_filename = Met_fstr.format(ymd=ymd, hms=hms, granule=m+1)
        Rad_filename = Rad_fstr.format(ymd=ymd, hms=hms, granule=m+1)
        AncSim_filename = AncSim_fstr.format(ymd=ymd, hms=hms, granule=m+1)

        print('Writing files:')
        print(Rad_filename)
        print(Met_filename)
        print(AncSim_filename)

        dat = {'Geometry':tdat, 'Aux-Met':adat_intp}
        file_creation.write_data_fromspec(dat, 'file_specs.json', Met_filename)

        dat = {'Geometry':tdat, 'Radiance':rdat}
        file_creation.write_data_fromspec(dat, 'file_specs.json', Rad_filename)

        dat = {'Geometry':tdat, 'Anc-Sim':sdat}
        file_creation.write_data_fromspec(dat, 'file_specs.json', AncSim_filename)

        dt = datetime.now()-t0
        print('Completed file ' + str(m+1) + ' elapsed time: '+str(dt))


if __name__ == "__main__":
    raise NotImplementedError("I do not think this part is up to date...")
    import sys
    if len(sys.argv) == 1:
        print('usage: python run_datasim.py <pool_size>')
    pool_size = int(sys.argv[1])
    batch_run_meltpond(pool_size=pool_size)
