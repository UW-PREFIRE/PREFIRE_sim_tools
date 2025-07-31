import glob, os, datetime

import numpy as np
import netCDF4
import h5py

import pyatmlib
from pyrtmwrap.aer import lblrtm_utils
import pyrtmwrap

import matplotlib.pyplot as plt

# timing
# LBLRTM v12.8  7mol_0-4000                   7.35
# LBLRTM v12.8  7mol_0-4000_ExBrd             7.32
# LBLRTM v12.8  7mol_0-4000_LineRej           2.46
# LBLRTM v12.8  allmol_0-4000                16.21
# LBLRTM v12.8  allmol_0-4000_ExBrd          16.12
# LBLRTM v12.8  allmol_0-4000_LineRej         8.98
# LBLRTM v12.10 7mol_0-4000                   8.34
# LBLRTM v12.10 7mol_0-4000_ExBrd             8.26
# LBLRTM v12.10 7mol_0-4000_LineRej           3.10
# LBLRTM v12.10 allmol_0-4000                19.63
# LBLRTM v12.10 allmol_0-4000_ExBrd          18.35
# LBLRTM v12.10 allmol_0-4000_LineRej        10.50

def create_lblrtm_inputs():
    """
    helper to create LBLRTM inputs.
    returns kw, profiles, surf_temps
    kw - keyword dictionary for lblrtm_utils.run
    profiles - list of 6 (each std atmosphere) profile data dict
    surf_temps - list of 6 surface temps.

    CO2 is set to 400 ppm, vertically constant
    """
    nc = netCDF4.Dataset('../../PREFIRE_sim_tools/data/'+
                         'Modtran_standard_profiles_PCRTM_levels.nc','r')

    profiles = []
    surf_temps = []

    surf_emis = 1.0
    surf_pres = 1013.0
    obs_pres = 0.005
    sensor_zen = 0.0

    # load all std atm data into a dictionary that LBLRTM wrapper will use.
    for a in range(6):
        
        surf_temps.append(nc['temp'][-1,a])

        profiles.append({})
        profiles[a]['pres'] = np.loadtxt('../data/plevs101.txt')
        profiles[a]['temp'] = nc['temp'][:,a]
        # very approx conversion from ppm to q [g/kg]
        profiles[a]['H2O'] = nc['h2o'][:,a] * 0.622 * 1e-6 * 1e3
        profiles[a]['CO2'] = np.zeros(profiles[a]['temp'].shape)+400.0
        profiles[a]['O3'] = nc['o3'][:,a]
        profiles[a]['N2O'] = nc['n2o'][:,a]
        profiles[a]['CO'] = nc['co'][:,a]
        profiles[a]['CH4'] = nc['ch4'][:,a]

    nc.close()

    # throw away the first N levels that are below the surface P.
    for profile in profiles:
        # add 1 here - this should make the surf_pres within
        # the layer described by the bottom two levels.
        k = np.searchsorted(profile['pres'], surf_pres) + 1
        # note, here we do the TOA->surf flip as well
        # (LBLRTM will want this ordered surf to TOA, the PCRTM
        # input data is ordered reverse.)
        for var in profile:
            profile[var] = profile[var][:k][::-1]

        z = pyatmlib.profiles.compute_altitude(
            profile['pres'], profile['temp'], profile['H2O'])
        profile['alt'] = z

    # for the std atm 2, LBLRTM actually seg faults, I think it was for
    # the top most layer radiance merge. Just removing the top layer seemed
    # to fix this.
    # I did check the delta in radiance by removing this level - it is
    # very small, and only changes the center of the 15 um CO2 band.
    for profile in profiles:
        for var in profile:
            profile[var] = profile[var][:-1]

    profile_units = {}
    profile_units['H2O'] = 'g/kg'
    for m in ('CO2', 'O3', 'N2O', 'CO', 'CH4'):
        profile_units[m] = 'ppmv'

    kw = {}

    # bug in tapes.tape5 means you cannot have a blank comment for now.
    kw['comment'] = 'version_testing'
    kw['run_type'] = 'rad'

    kw['profile_units'] = profile_units
    kw['nmol'] = 7
    kw['level_units'] = 'pressure'
    kw['path_start'] = surf_pres
    kw['path_end'] = obs_pres
    kw['path_angle'] = 180.0 - sensor_zen

    kw['wmin'] = 50.0
    kw['wmax'] = 2000.0
    kw['surf_emis'] = surf_emis

    return kw, profiles, surf_temps

def do_lblrtm_versiontest_runs():
    """
    run a bunch of tests on lblrtm 12.10 vs 12.8
    """
    kw, profiles, surf_temps = create_lblrtm_inputs()

    tape3_list = glob.glob(
        '/data/rttools/lblrtm/lblrtm_v12.8/hitran/TAPE3*')
    tape3_list.sort()
    os.environ['LBL_HOME'] = '/data/rttools/lblrtm/lblrtm_v12.8'

    timing = {}

    for a in range(6):

        kw['atm_flag'] = a + 1
        kw['profile'] = profiles[a]
        kw['levels'] = profile['pres']
        kw['surf_temp'] = surf_temps[a]

        for tape3 in tape3_list:

            kw['tape3'] = tape3
            tape3_desc = os.path.split(tape3)[1][11:]

            t0 = datetime.datetime.now()
            dat = lblrtm_utils.run(**kw)
            dt = datetime.datetime.now() - t0
            if tape3_desc in timing:
                timing[tape3_desc] += dt.total_seconds()
            else:
                timing[tape3_desc] = dt.total_seconds()

            output_file = os.path.join(
                '/data/users/mmm/lblrtm_output/version_tests',
                'lblrtm_v12.8_atm{0:1d}_'+tape3_desc+'.h5')
            output_file = output_file.format(a+1)

            with h5py.File(output_file, 'w') as h:
                h['wn'] = dat['wn']
                h['rad'] = dat['rad']
                h['bt'] = lblrtm_utils.lblrad2bt(dat['wn'], dat['rad'])

    for tape3_desc in sorted(list(timing.keys())):
        print('LBLRTM v12.8  {0:21s} '.format(tape3_desc) +
              '{0:12.2f}'.format(timing[tape3_desc]/60.0))

    # copy paste to do v12.10
    tape3_list = glob.glob(
        '/data/rttools/lblrtm/lblrtm_v12.10/hitran/TAPE3*')
    tape3_list.sort()
    os.environ['LBL_HOME'] = '/data/rttools/lblrtm/lblrtm_v12.10'

    timing = {}

    for a in range(6):

        kw['atm_flag'] = a + 1
        kw['profile'] = profiles[a]
        kw['levels'] = profile['pres']
        kw['surf_temp'] = surf_temps[a]

        for tape3 in tape3_list:

            kw['tape3'] = tape3
            tape3_desc = os.path.split(tape3)[1][11:]

            t0 = datetime.datetime.now()
            dat = lblrtm_utils.run(**kw)
            dt = datetime.datetime.now()-t0
            if tape3_desc in timing:
                timing[tape3_desc] += dt.total_seconds()
            else:
                timing[tape3_desc] = dt.total_seconds()

            output_file = os.path.join(
                '/data/users/mmm/lblrtm_output/version_tests',
                'lblrtm_v12.10_atm{0:1d}_'+tape3_desc+'.h5')
            output_file = output_file.format(a+1)

            with h5py.File(output_file, 'w') as h:
                h['wn'] = dat['wn']
                h['rad'] = dat['rad']
                h['bt'] = lblrtm_utils.lblrad2bt(dat['wn'], dat['rad'])

    for tape3_desc in sorted(list(timing.keys())):
        print('LBLRTM v12.10 {0:21s} '.format(tape3_desc) +
              '{0:12.2f}'.format(timing[tape3_desc]/60.0))


def compute_diffs():
    """
    load from version comparison tests (hardcoded data paths)
    """
    flist_v12p8 = glob.glob('/data/users/mmm/lblrtm_output/'+
                            'version_tests/lblrtm_v12.8_atm*')
    flist_v12p10 = glob.glob('/data/users/mmm/lblrtm_output/'+
                             'version_tests/lblrtm_v12.10_atm*')
    flist_v12p8.sort()
    flist_v12p10.sort()

    version_rmsd = np.zeros((6,6))
    version_maxd = np.zeros((6,6))
    version_maxdwn = np.zeros((6,6))
    brd_rmsd = np.zeros((6,2))
    rej_rmsd = np.zeros((6,2))

    for n, (f8,f10) in enumerate(zip(flist_v12p8, flist_v12p10)):
        print(os.path.split(f8)[1])
        with h5py.File(f8,'r') as h:
            wn8 = h['wn'][:]
            bt8 = h['bt'][:]
        with h5py.File(f10,'r') as h:
            wn10 = h['wn'][:]
            bt10 = h['bt'][:]
        if wn8.shape[0] != wn10.shape[0]:
            print('   shape mismatch')
            continue
        if np.any(wn8 != wn10):
            print('   wn mismatch')
        a = n // 6
        l = n % 6
        btd = (bt10 - bt8).astype(np.float64)
        version_rmsd[a,l] = btd.std()
        k = np.abs(btd).argmax()
        version_maxdwn[a,l] = wn8[k]
        version_maxd[a,l] = btd[k]

        if f8.endswith('_0-4000.h5'):
            bt8_ref = bt8
            bt10_ref = bt10
        else:
            btd8 = (bt8 - bt8_ref).astype(np.float64)
            btd10 = (bt10 - bt10_ref).astype(np.float64)
            if f8.endswith('_0-4000_ExBrd.h5'):
                brd_rmsd[a,0] = btd8.std()
                brd_rmsd[a,1] = btd10.std()
            if f8.endswith('_0-4000_LineRej.h5'):
                rej_rmsd[a,0] = btd8.std()
                rej_rmsd[a,1] = btd10.std()


    return version_rmsd, version_maxd, version_maxdwn, brd_rmsd, rej_rmsd


def get_wvfc_uncertainty():
    """
    data from Mlawer 2019 - [N,3] shaped array, the 3 columns
    are wavenumber, WV f continuum, % uncertainty
    """
    wvfc_uncertainty = np.array([
        [0, 9.31E-23, 4],
        [10, 9.38E-23, 10],
        [20, 9.41E-23, 18],
        [30, 9.97E-23, 12],
        [40, 1.17E-22, 12],
        [50, 1.38E-22, 12],
        [60, 1.54E-22, 15],
        [70, 1.51E-22, 18],
        [80, 1.39E-22, 21],
        [90, 1.22E-22, 25],
        [100, 1.06E-22, 30],
        [110, 9.15E-23, 30],
        [120, 8.01E-23, 30],
        [130, 6.97E-23, 30],
        [140, 6.05E-23, 30],
        [150, 5.24E-23, 30],
        [160, 4.56E-23, 30],
        [170, 3.76E-23, 30],
        [180, 3.05E-23, 30],
        [190, 2.56E-23, 30],
        [200, 2.20E-23, 30],
        [210, 1.81E-23, 30],
        [220, 1.48E-23, 27],
        [230, 1.16E-23, 24],
        [240, 8.77E-24, 21],
        [250, 7.10E-24, 21],
        [260, 5.90E-24, 21],
        [270, 5.17E-24, 21],
        [280, 4.19E-24, 21],
        [290, 3.40E-24, 21],
        [300, 2.68E-24, 21],
        [310, 2.06E-24, 21],
        [320, 1.84E-24, 21],
        [330, 1.46E-24, 21],
        [340, 1.09E-24, 21],
        [350, 8.49E-25, 21],
        [360, 6.42E-25, 21],
        [370, 4.80E-25, 21],
        [380, 3.80E-25, 21],
        [390, 2.94E-25, 14],
        [400, 2.22E-25, 7],
        [410, 1.97E-25, 7],
        [420, 1.68E-25, 7],
        [430, 1.50E-25, 7],
        [440, 1.34E-25, 7],
        [450, 1.11E-25, 7],
        [460, 9.66E-26, 7],
        [470, 8.73E-26, 7],
        [480, 7.47E-26, 7],
        [490, 6.30E-26, 7],
        [500, 5.31E-26, 7],
        [510, 4.83E-26, 7],
        [520, 4.21E-26, 7],
        [530, 3.61E-26, 7],
        [540, 3.06E-26, 7],
        [550, 2.67E-26, 7],
        [560, 2.30E-26, 7],
        [570, 1.97E-26, 7],
        # I missed this part of the table the first time,
        # but since the uncertainty is still 7%, it shouldn't
        # impact any of the calcs.
        [580, 1.63E-26, 7],
        [590, 1.37E-26, 7],
        [600, 1.17E-26, 7] ] )
    return wvfc_uncertainty


def do_wvca_uncertainty_runs():
    """
    do the main uncertainty runs.
    this basically runs LBLRTM with a sweep in the continuum
    multiplier, basically [-30% to +30%] in steps of 1%.

    writes data to hardcoded paths:
    /data/users/mmm/lblrtm_output/wvca_tests/

    foreign continua are:
    wv_fcontinuum_scalings_atmN.h5
    self continua are:
    wv_scontinuum_scalings_atmN.h5
    """
    kw, profiles, surf_temps = create_lblrtm_inputs()

    kw['wmin'] =  50.0
    kw['wmax'] = 1800.0

    pcts = np.arange(-30.0, 31.0, 1.0)
    pnum = pcts.shape[0]

    for a in range(6):

        kw['atm_flag'] = a + 1
        kw['profile'] = profiles[a]
        kw['levels'] = kw['profile']['pres']
        kw['surf_temp'] = surf_temps[a]

        out_file = (
            '/data/users/mmm/lblrtm_output/wvca_tests/'+
            'wv_fcontinuum_scalings_atm{0:1d}.h5'.format(a+1))
        #if os.access(out_file, os.R_OK):
        #    print('skipping atm '+str(a+1))
        #    continue

        out_dat = {}

        kw['continua_mult'] = np.ones(7)
        t0 = datetime.datetime.now()
        for p,pct in enumerate(pcts):
            kw['continua_mult'][1] = 1 + pct*0.01
            dat = lblrtm_utils.run(**kw)
            if p == 0:
                out_dat['wn'] = dat['wn']
                out_dat['rad'] = np.zeros((dat['rad'].shape[0],pnum))
                out_dat['bt'] = np.zeros((dat['rad'].shape[0],pnum))
            out_dat['rad'][:,p] = dat['rad']
            out_dat['bt'][:,p] = lblrtm_utils.lblrad2bt(
                dat['wn'], dat['rad'])
        print('atm fc ', a+1, str(datetime.datetime.now()-t0))

        with h5py.File(out_file, 'w') as h:
            for k in out_dat:
                h.create_dataset(k, data=out_dat[k], compression=1)
                h[k][:] = out_dat[k]
            h['continuum_scaling_pct'] = pcts

    for a in range(6):

        kw['atm_flag'] = a + 1
        kw['profile'] = profiles[a]
        kw['levels'] = kw['profile']['pres']
        kw['surf_temp'] = surf_temps[a]

        out_file = (
            '/data/users/mmm/lblrtm_output/wvca_tests/'+
            'wv_scontinuum_scalings_atm{0:1d}.h5'.format(a+1))
        #if os.access(out_file, os.R_OK):
        #    print('skipping atm '+str(a+1))
        #    continue

        kw['continua_mult'] = np.ones(7)
        t0 = datetime.datetime.now()
        for p,pct in enumerate(pcts):
            kw['continua_mult'][0] = 1 + pct*0.01
            dat = lblrtm_utils.run(**kw)
            if p == 0:
                out_dat['wn'] = dat['wn']
                out_dat['rad'] = np.zeros((dat['rad'].shape[0],pnum))
                out_dat['bt'] = np.zeros((dat['rad'].shape[0],pnum))
            out_dat['rad'][:,p] = dat['rad']
            out_dat['bt'][:,p] = lblrtm_utils.lblrad2bt(
                dat['wn'], dat['rad'])
        print('atm sc ', a+1, str(datetime.datetime.now()-t0))

        with h5py.File(out_file, 'w') as h:
            for k in out_dat:
                h.create_dataset(k, data=out_dat[k], compression=1)
            h['continuum_scaling_pct'] = pcts


def _get_weighted_idx(wn, uncert_grid, pct_grid):
    # helper function, computes indices and weights into the
    # perturbation array for the pre-computed continuum mult
    # sweep (the -30 to +30% multipler run done above)
    #
    # wn is the wavenumber grid for the LBLRTM simulations; 
    # uncert_grid is the output from get_wvfc_uncertainty()
    # pct_grid is the perturbation magnitude grid from the
    # stored sweep runs (the -30 to +30 )
    #
    # by default, this uses the end points for extrapolations;
    # that seems sufficient in this context - so for wn > 570 1/cm,
    # the highest wavenumber in the grid, it will just reuse the
    # value at 570 1/cm (7%)
    uncert_intp = np.interp(wn, uncert_grid[:,0], uncert_grid[:,2])
    frac_index = np.interp(
        uncert_intp, pct_grid, np.arange(pct_grid.shape[0]))
    int_index =  np.vstack([
        np.floor(frac_index).astype(int),
        np.ceil(frac_index).astype(int)]).T
    wt = np.ceil(frac_index) - frac_index
    return int_index, wt

def create_fcontinuum_diff_specs():
    """
    load the pre-computed foreign continuum sweep for each std atm;
    apply the uncertainty assessment from Mlawer to get the
    expected radiance perturbation.
    """
    uncert_grid_p = get_wvfc_uncertainty()
    uncert_grid_n = uncert_grid_p.copy()
    uncert_grid_n[:,2] *= -1
    fstr = ('/data/users/mmm/lblrtm_output/wvca_tests/'+
            'wv_fcontinuum_scalings_atm{0:1d}.h5')
    outfile_fstr = ('/data/users/mmm/lblrtm_output/wvfc_perturb/'+
                    'wvfc_perturb_atm{0:1d}.h5')
    for a in range(1,7):
        with h5py.File(fstr.format(a), 'r') as h:
            pct_grid = h['continuum_scaling_pct'][:]
            i0 = np.nonzero(pct_grid==0)[0][0]
            wn = h['wn'][:]
            rad = h['rad'][:]
        idx_p, wts_p = _get_weighted_idx(wn, uncert_grid_p, pct_grid)
        idx_n, wts_n = _get_weighted_idx(wn, uncert_grid_n, pct_grid)
        rad0 = rad[:, i0]
        # i is just a dummy index, so we get an array with the same
        # shape as one spectrm, where the element i is equal
        # to element [i,k] from the perturb rad grid.
        i = np.arange(rad.shape[0])
        radp = rad[i,idx_p[:,0]]*wts_p + rad[i,idx_p[:,1]]*(1-wts_p)
        radn = rad[i,idx_n[:,0]]*wts_n + rad[i,idx_n[:,1]]*(1-wts_n)
        with h5py.File(outfile_fstr.format(a), 'w') as h:
            h.create_dataset('wn', data=wn, compression=1)
            h.create_dataset('rad0', data=rad0, compression=1)
            h.create_dataset('radp', data=radp, compression=1)
            h.create_dataset('radn', data=radn, compression=1)

def _apply_TIRS_SRFs(mono_wl, rad_wl, srf_wl, srf):
    radc = np.zeros(srf.shape[1])
    # applies an extra point at the shortwave end, that is zero.
    for c in range(radc.shape[0]):
        if np.isnan(srf[0,c]):
            radc[c] = -9999
            continue
        mono_srf = np.interp(mono_wl, srf_wl, srf[:,c])
        srf_integ = np.trapz(mono_srf, mono_wl)
        rad_integ = np.trapz(rad_wl * mono_srf, mono_wl)
        radc[c] = rad_integ / srf_integ
    return radc

def create_final_TIRS_files():
    """
    apply TIRS srfs to the perturbation spectrum 
    from create_fcontinuum_diff_specs.

    also creates netCDF outputs from the original h5, 
    since these can have easier metadata (units)
    it is added to the h5 files.
    """
    fstr = ('/data/users/mmm/lblrtm_output/wvfc_perturb/'+
            'wvfc_perturb_atm{0:1d}.h5')
    fstr_out = ('/data/users/mmm/lblrtm_output/wvfc_perturb/'+
                'wvfc_perturb_atm{0:1d}_TIRS.nc')

    SRFfile = ('/data/rttools/TIRS_ancillary/'+
               'PREFIRE_SRF_v0.09.2_2020-02-21.nc')
    with netCDF4.Dataset(SRFfile, 'r') as nc:
        srf_wl = nc['wavelen'][:]
        srf = nc['srf_normed'][:]

    for a in range(1,7):

        dat = {}
        with h5py.File(fstr.format(a),'r') as h:
            dat['wn'] = h['wn'][:]
            dat['rad0'] = h['rad0'][:]
            dat['radp'] = h['radp'][:]
            dat['radn'] = h['radn'][:]
            dat['wl'], dat['rad0_wl'] = lblrtm_utils.lblrad2wl(dat['wn'], dat['rad0'])
            _, dat['radp_wl'] = lblrtm_utils.lblrad2wl(dat['wn'], dat['radp'])
            _, dat['radn_wl'] = lblrtm_utils.lblrad2wl(dat['wn'], dat['radn'])

        TIRS_rad0 = _apply_TIRS_SRFs(dat['wl'],dat['rad0_wl'],srf_wl,srf)
        TIRS_radp = _apply_TIRS_SRFs(dat['wl'],dat['radp_wl'],srf_wl,srf)
        TIRS_radn = _apply_TIRS_SRFs(dat['wl'],dat['radn_wl'],srf_wl,srf)

        # change to W/m2 from W/cm2.
        TIRS_rad0 *= 1e4
        TIRS_radp *= 1e4
        TIRS_radn *= 1e4

        vkw = dict(dimensions=('mono_spec',), zlib=True, complevel=1)

        with netCDF4.Dataset(fstr_out.format(a), 'w') as nc:
            nc.createDimension('mono_spec', dat['wn'].shape[0])
            nc.createDimension('TIRS_channels', 63)

            for v in dat:
                ncv = nc.createVariable(v, np.float, **vkw)
                ncv[:] = dat[v]

            ncv = nc.createVariable('TIRS_rad0', np.float, ('TIRS_channels',))
            ncv[:] = TIRS_rad0
            ncv = nc.createVariable('TIRS_radp', np.float, ('TIRS_channels',))
            ncv[:] = TIRS_radp
            ncv = nc.createVariable('TIRS_radn', np.float, ('TIRS_channels',))
            ncv[:] = TIRS_radn

            nc['wn'].setncattr('units', '1/cm')
            nc['wl'].setncattr('units', 'um')
            nc['wn'].setncattr('description', 'wavenumber grid for monochromatic spectrum')
            nc['wl'].setncattr('description', 'wavelength grid for monochromatic spectrum')

            nc['rad0'].setncattr('units', 'W / (cm^2 sr cm^-1)')
            nc['radp'].setncattr('units', 'W / (cm^2 sr cm^-1)')
            nc['radn'].setncattr('units', 'W / (cm^2 sr cm^-1)')
            nc['rad0'].setncattr('description', 'monochromatic radiance spectrum at baseline')
            nc['radp'].setncattr('description', 'monochromatic radiance spectrum with positive water vapor foreign continuum perturbation')
            nc['radn'].setncattr('description', 'monochromatic radiance spectrum with negative water vapor foreign continuum perturbation')

            nc['rad0_wl'].setncattr('units', 'W / (cm^2 sr um)')
            nc['radp_wl'].setncattr('units', 'W / (cm^2 sr um)')
            nc['radn_wl'].setncattr('units', 'W / (cm^2 sr um)')
            nc['rad0_wl'].setncattr('description', 'monochromatic radiance spectrum at baseline')
            nc['radp_wl'].setncattr('description', 'monochromatic radiance spectrum with positive water vapor foreign continuum perturbation')
            nc['radn_wl'].setncattr('description', 'monochromatic radiance spectrum with negative water vapor foreign continuum perturbation')

            nc['TIRS_rad0'].setncattr('units', 'W / (m^2 sr um)')
            nc['TIRS_radp'].setncattr('units', 'W / (m^2 sr um)')
            nc['TIRS_radn'].setncattr('units', 'W / (m^2 sr um)')
            nc['TIRS_rad0'].setncattr('description', 'TIRS channel radiance at baseline')
            nc['TIRS_radp'].setncattr('description', 'TIRS channel radiance with positive water vapor foreign continuum perturbation')
            nc['TIRS_radn'].setncattr('description', 'TIRS channel radiance with negative water vapor foreign continuum perturbation')

            nc.setncattr(
                'Description',
                'Water vapor foreign continuum perturbations are taken '+
                'from Mlawer 2019, JGR, Table 5.\n'+
                'Forward radiance simulations using LBLRTM v12.8, '+
                'for standard atmosphere '+str(a))
            nc.setncattr(
                'Contact',
                'email')
            nc.setncattr(
                'CreationDate',
                datetime.datetime.now().strftime('%d %b %Y'))
                         

def sample_plots():

    fstr = '/data/users/mmm/lblrtm_output/wvfc_perturb/wvfc_perturb_atm{0:1d}_TIRS.nc'
    TIRS_rad0 = []
    TIRS_radp = []
    TIRS_radn = []
    for a in range(1,7):
        with netCDF4.Dataset(fstr.format(a),'r') as nc:
            TIRS_rad0.append(nc['TIRS_rad0'][:])
            TIRS_radp.append(nc['TIRS_radp'][:])
            TIRS_radn.append(nc['TIRS_radn'][:])

    SRFfile = ('/data/rttools/TIRS_ancillary/'+
               'PREFIRE_SRF_v0.09.2_2020-02-21.nc')
    with netCDF4.Dataset(SRFfile, 'r') as nc:
        cwl1 = nc['channel_wavelen1'][:]
        cwl2 = nc['channel_wavelen2'][:]
        NEDR = nc['NEDR'][:]
    cwl = (cwl1 + cwl2) / 2.0

    fignum0 = 10
    fignum1 = 11
    fignum2 = 12
    plt.figure(fignum0).clf()
    plt.figure(fignum1).clf()
    plt.figure(fignum2).clf()
    fig0, ax0 = plt.subplots(1,1,num=fignum0)
    fig1, ax1 = plt.subplots(1,1,num=fignum1)
    fig2, ax2 = plt.subplots(1,1,num=fignum2)

    alabels = ['Trop', 'MLS', 'MLW', 'SAS', 'SAW', 'USStd']
    for a in range(5):
        obj, = ax0.plot(cwl, TIRS_rad0[a], label=alabels[a])
        obj, = ax1.plot(cwl, TIRS_radn[a]-TIRS_rad0[a], label=alabels[a])
        ax1.plot(cwl, TIRS_radp[a]-TIRS_rad0[a], label='_nolabel_', color=obj.get_color())
        ax2.plot(cwl, TIRS_radn[a]+TIRS_radp[a]-2*TIRS_rad0[a],label=alabels[a])

    ax0.set_ylim(-0.1,9)
    ax1.plot(cwl, NEDR, '-xr', label='NEDR')
    ax1.plot(cwl, -NEDR, '-xr', label='_nolabel_')

    for ax in (ax0, ax1, ax2):
        ax.set_xlabel('Wavelength [um]')
        ax.grid(1)
        ax.legend()
    #for fig in (fig0, fig1, fig2):
    #    fig.tight_layout()
    #plt.draw()
    ax0.set_ylabel('Radiance [W/(m^2 sr um)]')
    ax1.set_ylabel('$\\Delta$ Radiance [W/(m^2 sr um)]')
    ax2.set_ylabel('$\\Delta$ Radiance [W/(m^2 sr um)]')

    fig0.savefig('plots/wvca_test_fig0_rad0spectra.png')
    fig1.savefig('plots/wvca_test_fig1_diffspectra.png')
    fig2.savefig('plots/wvca_test_fig2_ddffspectra.png')

if __name__ == "__main__":
    do_wvca_uncertainty_runs()
