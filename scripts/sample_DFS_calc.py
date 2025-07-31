import netCDF4
import numpy as np
import matplotlib.pyplot as plt

import pyPCRTM

import OE_prototypes

####
# Example DFS calc for the temperature profile.
#

####
# Get K matrix from PCRTM. 

F = pyPCRTM.PCRTM()
F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True)

nc = netCDF4.Dataset('../../PREFIRE_sim_tools/data/'+
                     'Modtran_standard_profiles_PCRTM_levels.nc','r')
# use tropical for this example
atm_num = 1

F.tskin = nc['temp'][-1,atm_num]

F.tlev = nc['temp'][:,atm_num].astype(np.float32)
# very approx conversion from ppm to q [g/kg]
q = nc['h2o'][:,atm_num] * 0.622 * 1e-6 * 1e3
F.h2o = q.astype(np.float32)
F.co2 = nc['co2'][:,atm_num].astype(np.float32)
F.o3 = nc['o3'][:,atm_num].astype(np.float32)
F.n2o = nc['n2o'][:,atm_num].astype(np.float32)
F.co = nc['co'][:,atm_num].astype(np.float32)
F.ch4 = nc['ch4'][:,atm_num].astype(np.float32)
nc.close()

F.psfc = 1013.0
F.pobs = 0.005
F.sensor_zen = 0.0
F.emis = 1.0 + np.zeros(F.num_monofreq, np.float32)

dat = F.forward_rt()
K = dat['krad_t']

n_channels, n_levels = K.shape

####
# Get Se, Sa, from assumed parameters.
# NEDR in [mW/m2/sr/cm^-1], same as PCRTM; see figure 3, Mer_relli 2011.
NEDR = np.zeros(n_channels) + 0.5
Se = np.diag(NEDR**2)

# the PCRTM fixed pressure levels.
# need this to construct Sa.
P_levels = np.loadtxt('../data/plevs101.txt')

# T_uncertainty, in [K], for each of the 101 levels
T_uncertainty = np.zeros(n_levels) + 5.0

# T_correlation, this is the exponential scale length in [hPa], since
# we are using pressure levels.
T_correlation = np.linspace(10.0, 200.0, n_levels)

Sa, Ra = OE_prototypes.level_exp_cov_matrix(P_levels, T_uncertainty, T_correlation)

####
# do the DFS calc.
A = OE_prototypes.compute_A(K, Se, Sa)
DFS = np.trace(A)

fig, ax = plt.subplots()

ax.pcolormesh(P_levels, P_levels, A)
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_title('A matrix for temperature profile, DFS = {0:7.3f}'.format(DFS))

plt.show()
