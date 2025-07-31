import os.path

import h5py
import numpy as np
import matplotlib.pyplot as plt

import PREFIRE_sim_tools
from PREFIRE_sim_tools.utils import level_interp

def extract_base_profile():
    """ one-time function to extract the test profile - this was
    quasi-randomly selected from some of N.'s ERA5 arctic
    data, I just picked one that had a surface pressure near to
    level 97 from P101 (1013.95 hPa).
    """

    src_file = '/data/users/nnn/ERA5/h5/ERA5_Aux_met_test_not_interp.h5'
    i1, j1 = 4, 5
    i2, j2 = 7, 4
    with h5py.File(src_file,'r') as h:
        Tprof1 = h['temp_profile'][i1,j1,:]
        Pprof1 = h['pressure_profile'][i1,j1,:]
        Tprof2 = h['temp_profile'][i2,j2,:]
        Pprof2 = h['pressure_profile'][i2,j2,:]

    dst_file = os.path.join(
        PREFIRE_sim_tools.paths._data_dir,
        'level_interp_test_profile.h5')
    with h5py.File(dst_file, 'w') as h:
        h['Tprof1'] = Tprof1
        h['Pprof1'] = Pprof1
        h['Tprof2'] = Tprof2
        h['Pprof2'] = Pprof2


def make_test_profs(profile='1'):

    dst_file = os.path.join(
        PREFIRE_sim_tools.paths._data_dir,
        'level_interp_test_profile.h5')
    with h5py.File(dst_file, 'r') as h:
        Tprof = h['Tprof'+profile][:]
        Pprof = h['Pprof'+profile][:]

    inversion_scaling = [
        0.998, 0.995, 0.985, 0.982,
        0.980, 0.980, 0.980, 0.980, 0.980 ]

    sc = np.ones(137)
    testTprofs = np.zeros((137,3))

    for n,k in enumerate((4,7,9)):
        sc[-k:] = inversion_scaling[:k]
        testTprofs[:,n] = Tprof * sc
    
    testPprofs = np.zeros((137,3))
    testPprofs[:] = Pprof[:,np.newaxis]

    p101file = os.path.join(
        PREFIRE_sim_tools.paths._data_dir, 'plevs101.txt')
    p101 = np.loadtxt(p101file)

    dat = {}
    dat['pressure_profile'] = testPprofs.T.reshape((3,1,137))
    dat['temp_profile'] = testTprofs.T.reshape((3,1,137))

    # now skipping method 1, the linear extrap - this doesn't work
    # all that well in practice
    dat_intp0 = level_interp.pressure_interp(dat, p101, surf_extrap_method=0)
    #dat_intp1 = level_interp.pressure_interp(dat, p101, surf_extrap_method=1)
    dat_intp2 = level_interp.pressure_interp(dat, p101, surf_extrap_method=2)
    dat_intp3 = level_interp.pressure_interp(dat, p101, surf_extrap_method=3)

    dat_intp = (dat_intp0, dat_intp2, dat_intp3)

    return dat_intp, testPprofs, testTprofs


def plot_test_profs(dat_intp, P, T, fignum=22):
    markers = '*x+'
    fig = plt.figure(fignum, figsize=(12,5))
    axs = [fig.add_subplot(1,3,p) for p in range(1,4)]
    methods = (0, 2, 3)
    for p in range(3):
        for n in range(3):    
            obj, = axs[p].plot(T[:,n], P[:,n], marker=markers[n])
            axs[p].plot(dat_intp[p]['temp_profile'][n,0,:],
                        dat_intp[0]['pressure_profile'],
                        '--d', color=obj.get_color())
        Tsurf_string = 'Tsurf = {0:5.1f}, {1:5.1f}, {2:5.1f}'.format(
            dat_intp[p]['temp_profile'][0,0,-1],
            dat_intp[p]['temp_profile'][1,0,-1],
            dat_intp[p]['temp_profile'][2,0,-1])
        axs[p].set_title('intp method ' + str(methods[p]) + '\n'+
                         Tsurf_string)
        axs[p].grid(1)
        if p == 0:
            axs[p].set_ylabel('Pressure [hPa]')
        if p == 1:
            axs[p].set_xlabel('Temperature [K]')
    return axs

def batch_plot_test_profs():

    fignum=22
    fig = plt.figure(fignum, figsize=(12,5))
    fig.clf()

    dat_intp, testPprofs, testTprofs = make_test_profs('1')
    axs = plot_test_profs(dat_intp, testPprofs, testTprofs, fignum)
    for ax in axs:
        ax.set_ylim(1020,950)
        ax.set_xlim(270.0, 282.5)
    fig.savefig('plots/level_interp_test_nearsurf_synthetic_cases1.png')

    fig.clf()
    dat_intp, testPprofs, testTprofs = make_test_profs('2')
    axs = plot_test_profs(dat_intp, testPprofs, testTprofs, fignum)
    for ax in axs:
        ax.set_ylim(1010,940)
        ax.set_xlim(270.0, 282.5)
    fig.savefig('plots/level_interp_test_nearsurf_synthetic_cases2.png')


def _level_plot_helper(v101, p101, v, p, prange, fig, logp=False):
    fig.clf()
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    if logp:
        ax1.semilogy(v, p, '-o', label='NWP profile')
    else:
        ax1.plot(v, p, '-o', label='NWP profile')
    ax1.plot(v101, p101, '-x', ms=9, label='interp to PCRTM-101')
    ax1.set_ylabel('Pressure [hPa]')
    ax1.legend()
    ax1.legend(loc='best',prop={'size':9})
    ax1.set_ylim(prange)

    v_reintp = np.interp(p, p101, v101)
    if logp:
        ax2.semilogy(v_reintp-v, p, label='NWP - interp to PCRTM-101')
    else:
        ax2.plot(v_reintp-v, p, label='NWP - interp to PCRTM-101')
    xlims = ax2.get_xlim()
    max_x = np.abs(xlims).max()
    ax2.set_xlim(-max_x, max_x)
    ax2.legend(loc='best',prop={'size':9})
    ax2.set_ylim(prange)

    for pk in p101:
        ax1.axhline(pk, color='gray', ls='-', alpha=0.2)
        ax2.axhline(pk, color='gray', ls='-', alpha=0.2)

    return ax1, ax2

def plot_real_profs(T, P, T101, P101, Tintp, savefigs=False):
    # plots the 3 worst cases from real profiles
    # 1 worst case overall
    # 2 worst case in lower atmo (P > 13 hPa, roughly)
    # 3 worst case in surface-containing layer

    fignum = 10
    DT = np.abs(T - Tintp)
    surf_layer_maxabs_DT = np.zeros(DT.shape[:2])
    for a,b in np.ndindex(DT.shape[:2]):
        k = P101.searchsorted(P[a,b,-1])
        layer_bnd = P101[k-1:k+1]
        msk = np.logical_and(P[a,b,:] >= layer_bnd[0], P[a,b,:] <= layer_bnd[1])
        surf_layer_maxabs_DT[a,b] = np.max(np.abs(DT[a,b,msk]))

    # case 1
    fig1 = plt.figure(fignum+1, figsize=(8,6))
    a,b = np.unravel_index(DT.max(axis=2).argmax(), T.shape[:2])
    _level_plot_helper(T101[a,b,:], P101, T[a,b,:], P[a,b,:], 
                       (12.0, 0.002), fig1, logp=True)

    # case 2
    fig2 = plt.figure(fignum+2, figsize=(8,6))
    a,b = np.unravel_index(DT[:,:,20:].max(axis=2).argmax(), T.shape[:2])
    ax1, ax2 = _level_plot_helper(T101[a,b,:], P101, T[a,b,:], P[a,b,:],
                                  (1050.0, 950), fig2, logp=False)
    ax1.set_xlim(265, 285)

    # case 3
    fig3 = plt.figure(fignum+3, figsize=(8,6))
    a,b = np.unravel_index(surf_layer_maxabs_DT.argmax(), T.shape[:2])

    ax1, ax2 = _level_plot_helper(T101[a,b,:], P101, T[a,b,:], P[a,b,:], 
                                  (1030,950), fig3, logp=False)
    ax1.set_xlim(250, 265)

    if savefigs:
        fig1.savefig('level_interp_test_case1.png')
        fig2.savefig('level_interp_test_case2.png')
        fig3.savefig('level_interp_test_case3.png')


def run_batch_test(src_file=None):
    
    if src_file is None:
        src_file = '/data/users/nnn/ERA5/h5/ERA5_Aux_met_test_not_interp.h5'
    with h5py.File(src_file,'r') as h:
        T = h['temp_profile'][:]
        P = h['pressure_profile'][:]

    adat = {}
    adat['pressure_profile'] = P
    adat['temp_profile'] = T

    P101file = os.path.join(
        PREFIRE_sim_tools.paths._data_dir, 'plevs101.txt')
    P101 = np.loadtxt(P101file)

    adat_intp = level_interp.pressure_interp(adat, P101)

    # reinterpolate the derived T profiles (on p101) back to
    # the original 137 levels. this gives a sense of the errors.

    T101 = adat_intp['temp_profile']
    Tintp = np.zeros_like(T)

    for a,b in np.ndindex(Tintp.shape[:2]):
        Tintp[a,b,:] = np.interp(P[a,b,:], P101, T101[a,b,:])

    return T, P, T101, P101, Tintp


def batch_run_layer_stats():
   ddir = '/data/users/nnn/ERA5/h5/'
   hfiles = (
       ddir + 'ERA5_greenland_Aux_met_test_not_interp.h5',
       ddir + 'ERA5_Antarctica_Aux_met_test_not_interp.h5',)
   for hfile in hfiles:
       run_layer_stats(hfile)
       fig = plt.gcf()
       fig.suptitle(hfile.split('/')[-1][:-14])
       fig.savefig('plots/'+hfile.split('_')[-6]+'_surfT_stats.png')

       
def run_layer_stats(hfile):

    with h5py.File(hfile) as h:
        T = h['temp_ERA'][:]
        P = h['pres_ERA'][:]
        P101 = h['pressure_profile'][:]

    Tlayer_true = np.zeros(T.shape[:2] + (100,))
    Tlayer_aprx = np.zeros(T.shape[:2] + (100,4))
    layerP = np.vstack([P101[:-1], P101[1:]]).T

    surf_Tlayer_true = np.zeros(T.shape[:2])
    surf_Tlayer_aprx = np.zeros(T.shape[:2] + (4,))
    P_surf = P[:,:,-1]
    surf_dP = P_surf - P101[P101.searchsorted(P_surf)-1] 
    surf_dT = np.zeros_like(surf_Tlayer_aprx)

    for i,j in np.ndindex(Tlayer_true.shape[:2]):

        Tlayer_true[i,j,:] = level_interp.compute_layer_avg(
            T[i,j,:], P[i,j,:], layerP)
        k = np.nonzero(np.isfinite(Tlayer_true[i,j,:]))[0][-1]
        surf_Tlayer_true[i,j] = Tlayer_true[i,j,k]

        for m in range(4):
            T101 = level_interp._interp_helper(T[i,j,:], P[i,j,:], P101, m)
            Tlayer_aprx[i,j,:,m] = level_interp.compute_layer_avg(
                T101, P101, layerP)
            surf_Tlevels = T101[k:k+2]
            surf_T = np.interp(P[i,j,-1], P101[k:k+2], T101[k:k+2])
            surf_Tlayer_aprx[i,j,m] = 0.5*(surf_T+surf_Tlevels[0])
            surf_dT[i,j,m] = T[i,j,-1] - T101[-1]

    plot_layer_stats(surf_dT, surf_Tlayer_true, surf_Tlayer_aprx)

    return (Tlayer_true, Tlayer_aprx,
            surf_Tlayer_true, surf_Tlayer_aprx, surf_dP, surf_dT)


def plot_layer_stats(surf_dT, surf_Tlayer_true, surf_Tlayer_aprx):

    fig, ax = plt.subplots(2,1,figsize=(8,6))

    for m in (0,2,3):

        hh, hb = np.histogram(surf_dT[:,:,m].flatten(), 100)
        hbincs = (hb[1:] + hb[:-1])/2
        ax[0].semilogy(hbincs, hh, '-x')
        ax[0].set_xlabel('$\\Delta$ T_surf: ERA - interpolated [K]')
        ax[0].grid(1)

        layer_dT = surf_Tlayer_true - surf_Tlayer_aprx[:,:,m]
        hh, hb = np.histogram(layer_dT.flatten(), 100)
        hbincs = (hb[1:] + hb[:-1])/2
        ax[1].semilogy(hbincs, hh, '-x')
        ax[1].set_xlabel('$\\Delta$ surf Layer temp: '+
                         'ERA - interpolated [K]')
        ax[1].grid(1)

    ax[0].legend(['Copy Tsurf', 'Layer Avg',
                  'Copy Tsurf + Layer Avg'])
