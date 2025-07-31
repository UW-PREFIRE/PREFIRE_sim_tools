# script to compute CTP from the GFDL simulation time steps.
# essentially just a copy of compute_GFDL_wps.py with some small changes,
# since I did not feel like spending the time to make a generalized
# script to do both.
#
# this was implemented as script, so that I can call it from the
# command line, ensuring that there is full memory cleanup between
# runs (this appears to consume > 50% of the memory.)

import sys
import os.path
import datetime

import numpy as np
import netCDF4

from PREFIRE_sim_tools.datasim.GFDL_calcs import adjust_q, compute_ctp


# paths
ddir = "/data/users/xc/GFDL_fields_3km"
q_fstr = os.path.join(ddir, "{0:s}_plev_C3072_12288x6144.fre.day1_{1:s}UTC.nc4")
ps_fstr = os.path.join(ddir, "ps_C3072_12288x6144.fre.day1_{0:s}UTC.nc4")

# get CL params
if len(sys.argv) == 1:
    print("usage: python compute_GFDL_ctps.py <type> <utc> <output_dir>")
    print("       <type> is qi or ql, utc is 03, 06, or 09")
    sys.exit()

# assumptions
tau = 1.0
water_cloud_reff = 10.0
ice_cloud_reff = 30.0

qtype = sys.argv[1]
utc = sys.argv[2]
odir= sys.argv[3]
ofile_fstr = os.path.join(odir, "GFDL_{0:s}_{1:s}UTC.nc4")

t0 = datetime.datetime.now()

surfp_file = ps_fstr.format(utc)
q_file = q_fstr.format(qtype, utc)

outvar = 'ctp'

o_file = ofile_fstr.format(qtype[1:]+outvar, utc)

with netCDF4.Dataset(surfp_file, 'r') as n:
    ps = n['ps'][:]
# convert Pa -> hPa
ps *= 0.01

with netCDF4.Dataset(q_file, 'r') as n:
    n.set_auto_mask(False)
    q = n[qtype+'_plev'][:]
    plev = n['plev'][:]

q_adj, plev_adj = adjust_q(q, plev, ps)
if qtype == 'ql':
    ctp = compute_ctp(q_adj, plev_adj, tau=tau, r_eff=water_cloud_reff, phase='water')
else:
    ctp = compute_ctp(q_adj, plev_adj, tau=tau, r_eff=ice_cloud_reff, phase='ice')
# for testing
#ctp = np.zeros(ps.shape)

print('computed CTP, elapsed time: ' + str(datetime.datetime.now()-t0))

t0 = datetime.datetime.now()

with netCDF4.Dataset(o_file, 'w') as n:

    n.createDimension('grid_yt', ctp.shape[0])
    n.createDimension('grid_xt', ctp.shape[1])

    n.createVariable('grid_yt', np.float32, ('grid_yt',))
    n.createVariable('grid_xt', np.float32, ('grid_xt',))
    n.createVariable(outvar, np.float32, ('grid_yt', 'grid_xt'))

    with netCDF4.Dataset(q_file, 'r') as nn:
        for var in ('grid_xt', 'grid_yt'):
            n[var][:] = nn[var][:]
            for attr in ('units', 'long_name', 'cartesian_axis', 'bounds'):
                n[var].setncattr(attr, nn[var].getncattr(attr))

    n[outvar][:] = ctp
    if qtype == 'ql':
        n[outvar].setncattr('long_name','Cloud Top Pressure (Liquid clouds)')
    else:
        n[outvar].setncattr('long_name','Cloud Top Pressure (Ice clouds)')
    n[outvar].setncattr('units', 'hPa')
    n[outvar].setncattr('description', 'Approximate pressure where cloud optical '+
                        'depth = {0:5.1f} assuming simple optical properties'.format(tau))

print('created CTP output file, elapsed time: ' + str(datetime.datetime.now()-t0))
