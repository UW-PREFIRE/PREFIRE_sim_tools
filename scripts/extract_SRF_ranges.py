import os.path
import sys

import netCDF4
import h5py
import numpy as np

def srf_pctile_wvrange(wavelen, srf, pct):
    maxval = np.max(srf)
    threshval = maxval * pct
    m = srf >= threshval
    i = m.nonzero()[0]
    wvrange = wavelen[i[0]], wavelen[i[-1]]
    return wvrange

def get_all_wvranges(wavelen, srf, pct):
    wvranges = np.zeros(srf.shape[1:] + (2,))
    for c, s in np.ndindex(srf.shape[1:]):
        srf_cs = srf[:,c,s]
        if np.sum(srf_cs) == 0:
            continue
        wvranges[c,s,:] = srf_pctile_wvrange(wavelen, srf_cs, pct)
    return wvranges

def get_selected_pct_wvranges(srf_file):
    with netCDF4.Dataset(srf_file, 'r') as nc:
        nc.set_auto_mask(False)
        srf = nc['srf'][:]
        wavelen = nc['wavelen'][:]
        ch_wave1 = nc['channel_wavelen1'][:]
        ch_wave2 = nc['channel_wavelen2'][:]
    wvranges_theo = np.dstack([ch_wave1, ch_wave2])
    wvranges_50 = get_all_wvranges(wavelen, srf, 0.50)
    wvranges_05 = get_all_wvranges(wavelen, srf, 0.05)
    return wvranges_theo, wvranges_50, wvranges_05

def write_wvranges_to_csv(wvranges, wvr_file):
    
    with open(wvr_file, 'w') as f:
        #f.write('{:4s},{:12s},{:12s},{:12s},{:12s},{:12s},{:12s}\n'.format(
        #    'Ch#', 'wl1 theo', 'wl2 theo', 'wl1 50%', 'wl2 50%', 'wl1 1%', 'wl2 1%'))
        #fstr = '{:4d},{:12.8f},{:12.8f},{:12.8f},{:12.8f},{:12.8f},{:12.8f}\n'
        f.write('{:>4s},{:>12s},{:>12s},{:>12s},{:>12s}\n'.format(
            'Ch#', 'wl1 50%', 'wl2 50%', 'wl1 5%', 'wl2 5%'))
        fstr = '{:4d},{:12.8f},{:12.8f},{:12.8f},{:12.8f}\n'
        s = 0
        for c in range(63):
            f.write(fstr.format(
                c+1,
                #wvranges[0][c,s,0], wvranges[0][c,s,1],
                wvranges[1][c,s,0], wvranges[1][c,s,1],
                wvranges[2][c,s,0], wvranges[2][c,s,1]))

    with h5py.File(wvr_file.replace('csv', 'h5'), 'w') as h:
        #h['wavelen1_theo'] = wvranges[0][:,s,0]
        #h['wavelen2_theo'] = wvranges[0][:,s,1]
        h['wavelen1_50'] = wvranges[1][:,s,0]
        h['wavelen2_50'] = wvranges[1][:,s,1]
        h['wavelen1_05'] = wvranges[2][:,s,0]
        h['wavelen2_05'] = wvranges[2][:,s,1]

if __name__ == '__main__':

    srf_file = sys.argv[1]
    wvr_file = srf_file.replace('.nc', '_wave_ranges.csv')
    wvr_file = os.path.split(wvr_file)[1]

    wvranges = get_selected_pct_wvranges(srf_file)
    write_wvranges_to_csv(wvranges, wvr_file)
