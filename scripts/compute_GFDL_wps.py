# script to compute IWP or LWP from the GFDL simulation time steps.
# this was implemented as script, so that I can call it from the
# command line, ensuring that there is full memory cleanup between
# runs (this appears to consume > 50% of the memory.)

import sys
import os.path
import datetime

import numpy as np
import netCDF4

from PREFIRE_sim_tools.datasim.GFDL_calcs import adjust_q, compute_wp


# paths
ddir = "/data/users/xc/GFDL_fields_3km"
q_fstr = os.path.join(ddir, "{0:s}_plev_C3072_12288x6144.fre.day1_{1:s}UTC.nc4")
ps_fstr = os.path.join(ddir, "ps_C3072_12288x6144.fre.day1_{0:s}UTC.nc4")

# get CL params
if len(sys.argv) == 1:
    print("usage: python compute_GFDL_wps <type> <utc> <output_dir>")
    print("       <type> is q, qi or ql, utc is 03, 06, or 09")
    sys.exit()

qtype = sys.argv[1]
utc = sys.argv[2]
odir= sys.argv[3]
ofile_fstr = os.path.join(odir, "GFDL_{0:s}_{1:s}UTC.nc4")

t0 = datetime.datetime.now()

surfp_file = ps_fstr.format(utc)
q_file = q_fstr.format(qtype, utc)
if qtype == 'q':
    outvar = 'pwv'
else:
    outvar = qtype[1:]+'wp'

o_file = ofile_fstr.format(outvar, utc)

with netCDF4.Dataset(surfp_file, 'r') as n:
    ps = n['ps'][:]
# convert Pa -> hPa
ps *= 0.01

with netCDF4.Dataset(q_file, 'r') as n:
    n.set_auto_mask(False)
    q = n[qtype+'_plev'][:]
    plev = n['plev'][:]

q_adj, plev_adj = adjust_q(q, plev, ps)
wp = compute_wp(q_adj, plev_adj)
# for testing
#wp = np.zeros(ps.shape)

print('computed WP, elapsed time: ' + str(datetime.datetime.now()-t0))

t0 = datetime.datetime.now()

with netCDF4.Dataset(o_file, 'w') as n:

    n.createDimension('grid_yt', wp.shape[0])
    n.createDimension('grid_xt', wp.shape[1])

    n.createVariable('grid_yt', np.float32, ('grid_yt',))
    n.createVariable('grid_xt', np.float32, ('grid_xt',))
    n.createVariable(outvar, np.float32, ('grid_yt', 'grid_xt'))

    with netCDF4.Dataset(q_file, 'r') as nn:
        for var in ('grid_xt', 'grid_yt'):
            n[var][:] = nn[var][:]
            for attr in ('units', 'long_name', 'cartesian_axis', 'bounds'):
                n[var].setncattr(attr, nn[var].getncattr(attr))

    n[outvar][:] = wp
    if outvar == 'iwp':
        n[outvar].setncattr('long_name','Ice Water Path')
    elif outvar == 'lwp':
        n[outvar].setncattr('long_name','Liquid Water Path')
    elif outvar == 'pwv':
        n[outvar].setncattr('long_name','Precipitable Water Vapor')

    n[outvar].setncattr('units', 'kg/m^2')


print('created WP output file, elapsed time: ' + str(datetime.datetime.now()-t0))
