import numpy as np
import netCDF4 as nc
import os.path
from PREFIRE_sim_tools import paths

#constants from uwiremis_module:
_numwave = 416
_hngpnts = 10
_numpcs = 6

_pcm_modres = np.array([0.9782182, 0.9759824, 0.9681804, 0.9187206, 0.9265846, 0.9733831, 0.9684200, 0.9459322, 0.8725729, 0.8704424])

# loads static data, right now this is stored in the same
# directory as this code (utils subdirectory, from paths._code_dir.)
# use internal paths module to locate it.
_data_dir = os.path.join(paths._code_dir, 'utils')

_pcm = np.loadtxt(os.path.join(_data_dir, 'pcm.txt')).flatten()

_labeigvects = nc.Dataset(os.path.join(_data_dir, 'UWiremis_labeigvects.nc'))

_pcu = _labeigvects['PC_scores'][:]

_pcu_modres = np.reshape(np.loadtxt(os.path.join(_data_dir, 'pcu_modres.txt')),[10,10])

hsr_wns = np.loadtxt(os.path.join(_data_dir, 'hsr_wavenumbers.txt')).flatten()

_A = np.empty([6,10])
for i in range(_numpcs):
    for j in range(_hngpnts):
        _A[i,j] = _pcu_modres[j,i]
_At = np.transpose(_A)
_B = np.matmul(_A,_At)
_C = np.linalg.inv(_B)
_D = np.matmul(_C,_A)


def get_hsr_emis(bfem):
    """ 
    takes emissivity values at 10 hingepoints and calculates high spectral resolution emissivities

    Parameters
    ----------
    bfem : ndarray
        Array of emissivities at 10 hingepoints used in UW Baseline Fit emissivity database

    Returns
    -------
    hsremis : ndarray
        High spectral resolution emissivities
    hsrwns : ndarray
        Wavenumbers at which the hsr emissivity is calculated
    """


    hsremis = np.zeros(_numwave)
    coef = np.zeros(_numpcs)
    col = np.zeros(_hngpnts)
    if bfem[0] <= 0:
        return
    
    col = bfem - _pcm_modres
        
    coef = np.matmul(col,np.transpose(_D))
        
    emis1 = np.zeros(_numwave)
        
    if coef[0] != -999.:
        for i in range(_numwave):
            for j in range(_numpcs):
                emis1[i] += coef[j]*_pcu[i,j]
            
        emis1 += _pcm
        
        for i in range(_numwave):
            if emis1[i] > 1:
                emis1[i] = 1
            
            hsremis[i] = float(emis1[i])
            
    else:
        coef = coef-999
        hsremis = hsremis-999
    
    return hsremis, hsr_wns



def get_instr_emis(hsremis,instrwn):
    """
    Interpolate high spectral resolution emissivity to specific wavenumbers using bilinear
    interpolation.
    
    Parameters
    ----------
    hsremis : ndarray
        array of high spectral resolution emissivity at the wavenumbers defined above
    isntrwn : ndarray
        array of instruments wavenumbers (in 1/cm) to which emissivity is to be interpolated

    Returns
    -------
    instremis : ndarray
        array of emissivities at the given instrument wavenumbers
    """

    instremis = np.zeros(len(instrwn))
    
    for i in range(len(instrwn)):
        #at wavenumbers outside of the range of hsr_wns, emissivities are set equal to the nearest
        #hsremis value
        if instrwn[i] <= hsr_wns[0]:
            instremis[i] = hsremis[0]
        
        elif instrwn[i] >= hsr_wns[-1]:
            instremis[i] = hsremis[-1]
        
        else:
            dist = np.abs(instrwn[i]-hsr_wns)
            mindex = np.argmin(dist)
            
            k = 1
            
            if(instrwn[i]<=hsr_wns[mindex]):
                k = -1
            
            dwvnum1 = dist[mindex]
            dwvnum2 = dist[mindex+k]
            
            hsremis1 = dwvnum1*hsremis[mindex+k]
            hsremis2 = dwvnum2*hsremis[mindex]
            
            dwvsum = dwvnum1 + dwvnum2
            
            instremis[i] = (hsremis1 + hsremis2)/dwvsum
            
    return instremis
