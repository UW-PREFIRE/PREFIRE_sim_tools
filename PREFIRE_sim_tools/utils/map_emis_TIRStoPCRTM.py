import numpy as np
import netCDF4
import pyPCRTM
import copy


def emis_TIRStoPCRTM(wn_tot,valid_emis,emis0,wn_new):
    """
    Map surface emissivity for given TIRS channels to given wavenumber grid 
    (usually PCRTM input values). This code converted from Yan's Matlab code.

    Parameters
    -----------
    wn_tot : ndarray (63,2) - wavenumber ranges for all TIRS channels
    valid_emis :  ndarray (n) - the integer channel numbers that have retrieved emissivity
        these are assumed to be in the python convention, with the first channel at index 0.
    emis0 : ndarray (n) - the retrieved emissivity for each channel
    wn_new : ndarray (m) - monochromatic wavenumbers (usually the 740 used for the PCRTM input

    Returns
    -------
    emis_new : ndarray (m) - emissivities at the monochromatic wavenumbers (can be used as input to PCRTM if wn_new is on the PCRTM grid)

    """
    emis_new = np.zeros(wn_new.shape) -999.0
    wn_valid = wn_tot[valid_emis,:]
    # select out the index of non-valid PREFIRE channels within the spectral range
    temp = np.array(range(valid_emis[0],valid_emis[-1]+1))
    nonvalid_emis = np.setdiff1d(temp,valid_emis)
    wn_nonvalid = wn_tot[nonvalid_emis,:]


    #if the wavenumber falls in any PREFIRE channel
    for i in range(len(valid_emis)):        
        idw = ((wn_new >= wn_valid[i,0]) & (wn_new <= wn_valid[i,1]))
        emis_new[idw] = emis0[i]
    
    # if the wavenumber falls beyond the PREFIRE spectral range
    id_less = np.nonzero(wn_new < min(wn_valid[:,0]))[0]
    id_larger = np.nonzero(wn_new > max(wn_valid[:,1]))[0]
    emis_new[id_less] = emis_new[id_less[-1]+1]
    emis_new[id_larger] = emis_new[id_larger[0]-1]



    # if the wavenumber does not fall in any valid(i.e. selected) PREFIRE channel 
    # but still within the spectral range
    for j in range(len(nonvalid_emis)):
        idw = ((wn_new >= wn_nonvalid[j,0]) & (wn_new <= wn_nonvalid[j,1]))
        if any(nonvalid_emis[j]-1 == valid_emis):
            if any(nonvalid_emis[j]+1 == valid_emis):
                emis_new[idw] = (emis0[(nonvalid_emis[j]- valid_emis[0]-j)-1] + emis0[(nonvalid_emis[j] - valid_emis[0]+1-j)-1])/2.0
            else:
                emis_new[idw] = emis0[(nonvalid_emis[j]- valid_emis[0]-j)-1]
        else:
            if any(nonvalid_emis[j]+1 == valid_emis):
                emis_new[idw] = emis0[(nonvalid_emis[j] - valid_emis[0]+1-j)-1]
            else:
                emis_new[idw] = emis0[(nonvalid_emis[j] - valid_emis[0]-j)-1]
            
    return emis_new

def emis_PCRTMtoTIRS(wn_tot,emis_in,wn_in):
    """
    Map surface emissivity (usually PCRTM input wn grid) to TIRS channels.

    Parameters
    -----------
    wn_tot : ndarray (63,2) - wavenumber ranges for all TIRS channels
    emis_in : ndarray (n) - the emissivity spectra to be converted to TIRS channels
    wn_in : ndarray (m) - monochromatic wavenumbers (usually the 740 used for the PCRTM input) that define the wavenumbers of emis_in

    Returns
    -------
    tirs_new : ndarray (63) - emissivities at the TIRS wavenumbers

    """
    nchan = 63
    tirs_new = np.zeros(nchan) - 999.0

    emis_wn_tot_0 = np.interp(wn_tot[:,0],wn_in,emis_in)
    emis_wn_tot_1 = np.interp(wn_tot[:,1],wn_in,emis_in)

    for i in range(nchan):
        idw = ((wn_in >= wn_tot[i,0]) & (wn_in <= wn_tot[i,1]))
        if any(idw) == True:
            tirs_new[i] = np.mean(emis_in[idw])    
        else:
            tirs_new[i] = np.mean([emis_wn_tot_0[i],emis_wn_tot_1[i]])

    return tirs_new

