import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import copy
import pyPCRTM

import OE_prototypes

####
# Example DFS calc for the temperature profile.
#

####
# Get K matrix from PCRTM. 

F = pyPCRTM.PCRTM()
F.init(2, output_jacob_flag=True, output_ch_flag=True, output_jacob_ch_flag=True,output_bt_flag=True,output_jacob_bt_flag=True)

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
K_zero_test = np.all(K==0,axis=0)==0
Kt=K[:,K_zero_test]

Kt_pcrm_all6 = copy.deepcopy(dat['krad_mol'])
Kt_h2o = Kt_pcrm_all6[:,K_zero_test,0]
K_both = np.hstack([Kt_h2o,Kt])

n_channels, n_levels = Kt.shape

####
# Get Se, Sa, from assumed parameters.
# NEDR in [mW/m2/sr/cm^-1], same as PCRTM; see figure 3, Mer_relli 2011.
NEDR = np.zeros(n_channels) + 0.5
Se = np.diag(NEDR**2)

# the PCRTM fixed pressure levels.
# need this to construct Sa.
P_levels_orig = np.loadtxt('../data/plevs101.txt')
P_levels = P_levels_orig[K_zero_test]

# uncertainty for each of the 101 levels
T_uncertainty = np.zeros(n_levels) + 5.0
Q_uncertainty = np.zeros(n_levels) + q[K_zero_test]*0.3

# T_correlation, this is the exponential scale length in [hPa], since
# we are using pressure levels.
T_correlation = np.linspace(10.0, 200.0, n_levels)
Q_correlation = np.linspace(10.0, 200.0, n_levels)

Sa_t, Ra_t = OE_prototypes.level_exp_cov_matrix(P_levels, T_uncertainty, T_correlation)

Sa_q, Ra_q = OE_prototypes.level_exp_cov_matrix(P_levels, Q_uncertainty, Q_correlation)


fillin = Sa_t*0.0
temp1 = np.hstack([fillin,Sa_t])
temp2 = np.hstack([Sa_q,fillin])
Sa_both = np.vstack([temp2,temp1])



####
# do the DFS calc.
#A = OE_prototypes.compute_A(Kt_h2o, Se, Sa_q)
A = OE_prototypes.compute_A(K_both, Se, Sa_both)
DFS = np.trace(A)

fig, ax = plt.subplots()

P_levels2 = np.hstack([P_levels,P_levels])
##ax.pcolormesh(P_levels, P_levels, A)
ax.pcolormesh(A,vmin=-1.5,vmax=1.5,cmap='bwr')
##ax.set_xscale('log')
##ax.set_yscale('log')

ax.set_title('A matrix for temp and h2o profiles, DFS = {0:7.3f}'.format(DFS))


##winnum = 9
##plt.figure(winnum,figsize=(10,10))
#plt.pcolormesh(P_levels2, P_levels2, A,vmin=-3.5,vmax=3.5,cmap='bwr')

#plt.pcolormesh(A)
##plt.pcolormesh(A,vmin=-3.5,vmax=3.5,cmap='bwr')

##cbar = plt.colorbar()


plt.show()
#plt.savefig('/home/nnn/plots/A_temp_h20.png')
#plt.close()
