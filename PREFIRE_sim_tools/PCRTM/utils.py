import os

import numpy as np
import struct


def read_PC_file(PCfile):
    """
    reads the stored PC file for PCRTM.
    This is a flat float32 binary file, according to an ad hoc format
    from XC/XH's GFDL-PCRTM forward run.

    Note that the output spectrum from PCRTM consists of several bands,
    so that the final spectrum is a concatenation across the bands.

    Returns a python dictionary with fields described below.

    Parameters
    ----------
    PCfile : str
        file path to the stored PC file.
        See data/PCRTM_Pcnew_id2.dat inside PREFIRE_sim_tools for an
        example file, that contains the 4 x 100 stored PCs used in the
        GFDL-PCRTM forward run.

    Returns
    -------
    numbnd : int
        number of PCRTM bands
    numPC : int
        total number of PCs
    numch : int
        total number of channels
    numPC_perband : ndarray
        integer array with number of PCs in each band. shape (numbnd,)
    numch_perband : ndarray
        integer array with number of channels in each band. shape (numbnd,)
    PCcoef : ndarray
        object array with shape (numbnd,), containing the PC coefficient
        arrays for each band. Each array will be shaped (numPC, numch)
    Pstd : ndarray
        object array with shape (numbnd,), containing the PC coefficient
        stdev arrays for each band. Each array will be shaped (numch,)
    Pstd : ndarray
        object array with shape (numbnd,), containing the PC coefficient
        mean arrays for each band. Each array will be shaped (numch,)
    band_slices : list
        for convenience, this contains slice objects for each band, to extract
        bands from a fully concatenated spectrum.
    ch_idx : ndarray
        object array with shape (numbnd,), containing channel index arrays
        into fully concatenated array. This is similar to the band_slices,
        but instead of slice objects this contains integer indexing ndarrays.
    band_idx : ndarray
        integer ndarray with shape (numch,), with integers labels 1 ... numbnd
        for the concatenated array. This is also similar to the ch_idx and
        band_slices, but labels the bands in the full spectral channel array.
    """
    x = np.fromfile(PCfile, np.float32)

    dat = {}
    dat['numbnd'] = int(x[0])

    dat['numPC_perband'] = np.zeros(dat['numbnd'], int)
    dat['numch_perband'] = np.zeros(dat['numbnd'], int)

    dat['PCcoef'] = np.zeros(dat['numbnd'], object)
    dat['Pstd']   = np.zeros(dat['numbnd'], object)
    dat['Pmean']  = np.zeros(dat['numbnd'], object)

    # read header values first, to get array shapes.
    for b in range(dat['numbnd']):
        dat['numPC_perband'][b] = x[1+b]
        dat['numch_perband'][b] = x[1+b+dat['numbnd']]

    ch_idx = np.zeros(dat['numbnd'], object)
    dat['numPC'] = np.sum(dat['numPC_perband'])
    dat['numch'] = np.sum(dat['numch_perband'])
    dat['band_idx'] = np.zeros(dat['numch'], np.int8)

    bslices = []
    # to keep track of the total number of channels.
    ch_start = 0
    for b in range(dat['numbnd']):
        bslices.append( slice(ch_start, ch_start+dat['numch_perband'][b]) )
        ch_idx[b] = np.arange(ch_start, ch_start+dat['numch_perband'][b])
        ch_start += dat['numch_perband'][b]

    # starting file position, after the integer values for
    # 1) number of bands; and 2) the number of PC and channel
    # in each band.
    ctr = 1 + 2*dat['numbnd']

    for b in range(dat['numbnd']):

        dat['band_idx'][ch_idx[b]] = b+1
        dat['PCcoef'][b] = np.zeros((dat['numPC_perband'][b],dat['numch_perband'][b]))

        for p in range(dat['numPC_perband'][b]):
            dat['PCcoef'][b][p,:] = x[ctr:ctr+dat['numch_perband'][b]]
            ctr += dat['numch_perband'][b]

        dat['Pstd'][b] = x[ctr:ctr+dat['numch_perband'][b]]
        ctr += dat['numch_perband'][b]
        dat['Pmean'][b] = x[ctr:ctr+dat['numch_perband'][b]]
        ctr += dat['numch_perband'][b]

    dat['band_slices'] = bslices
    dat['ch_idx'] = ch_idx

    return dat


def read_score_file(numPC_perband, sfile):
    """
    read the PC score information from a stored file.
    This is another flat float32 binary file, according to an ad hoc format
    from XC/XH's GFDL-PCRTM forward run.

    In this case the binary file contains the scores directly with no
    shape metadata, such as the number of PCs per band, which must be
    read from ancillary information (e.g. the PC file)
    The score files do contain the lat/lon of the sample, and the surface
    type (if this is the newer format).

    Parameters
    ----------
    numPC_perband : list or arraylike
        contains the integer number of PCs for each band
    sfile : str
        file path to the stored score file

    Returns
    -------
    numrec : int
        number of records loaded from the file
    lat : ndarray
        latitudes of the samples in the file, shaped (numrec,)
    lon : ndarray
        longitudes of the samples in the file, shaped (numrec,)
    PCscore : ndarray
        object array, shaped (numbnd,), where numbnd was the number of
        bands specified in the input numPC_perband
        Each element is a (numrec, numPC) array.
    surface_type : ndarray
        If the score file is the newer format, this will be the integer
        surface type for each grid cell. Shaped (numrec,).
        These surface types should be matched with the separate netCDF
        file that contains the emissivity spectra and type descriptions.
    """
    
    file_size = os.stat(sfile).st_size
    num_vals = int(file_size / 4)

    num_band = len(numPC_perband)

    # newer data has different format. The files contain some number
    # of records with either:
    # [lat, lon, numPC_total] or [lat, lon, surf_type, numPC_total]
    # check each possibility, looking for either of those to produce
    # an integer number of records. if neither do, throw Error.

    reclen1 = 2 + np.sum(numPC_perband)
    reclen2 = 3 + np.sum(numPC_perband)
    numrec1_float = num_vals / reclen1
    numrec2_float = num_vals / reclen2
    numrec1 = int(numrec1_float)
    numrec2 = int(numrec2_float)

    if (numrec1 != numrec1_float) and (numrec2 != numrec2_float):
        raise ValueError("score file length does not match number of PCs, "+
                         "for either score file format")

    if numrec1 == numrec1_float:
        # this is the lat,lon,Scores format
        file_format = 1
        numrec = numrec1
    else:
        # this is the lat,lon,type,Scores format
        file_format = 2
        numrec = numrec2

    x = np.fromfile(sfile, np.float32)

    dat = {}
    dat['numrec'] = numrec
    dat['lat'] = np.zeros(numrec)
    dat['lon'] = np.zeros(numrec)
    if file_format == 2:
        dat['surface_type'] = np.zeros(numrec, np.int)
    dat['PCscore'] = np.zeros(num_band, np.object)

    for b in range(num_band):
        dat['PCscore'][b] = np.zeros((numrec, numPC_perband[b]))

    ctr = 0
    for r in range(numrec):
        dat['lat'][r] = x[0+ctr]
        dat['lon'][r] = x[1+ctr]
        if file_format == 1:
            ctr += 2
        else:
            dat['surface_type'][r] = x[2+ctr]
            ctr += 3

        for b in range(num_band):
            dat['PCscore'][b][r,:] = x[ctr:ctr+numPC_perband[b]]
            ctr += numPC_perband[b]

    return dat


def radiance_from_PCscores(pcdat, sfile, recslice=None):
    """
    loads a PC score file, and computes the radiance spectrum.

    Parameters
    ----------
    pcdat : dict
        the stored PC dictionary, loaded from read_PC_file()
    sfile : str
        the file path to the stored score file
    recslice : None or slice object
        optionally, set a slice object to subset the records within the
        stored score file. If this is set, this is directly applied to the
        record index of the scores - the user is responsible for giving
        a valid slice for that index, no checking is done.

    Returns
    -------
    rad : ndarray
        float array with shape (numrec, numch), where numrec is the
        number of samples found in the input score file, and numch
        is the number of spectral channels in the concatenated
        spectrum.

    """

    sdat = read_score_file(pcdat['numPC_perband'], sfile)

    for b in range(pcdat['numbnd']):

        Pscale = pcdat['Pstd'][b][np.newaxis, :]
        PC = pcdat['PCcoef'][b]

        if recslice:
            scores = sdat['PCscore'][b][recslice, :]
        else:
            scores = sdat['PCscore'][b]

        # on first iteration, we now know the required shape for
        # the output - if recslice was input, we don't know the
        # shape until after we applied it.
        if b == 0:
            num_output_rec = scores.shape[0]
            rad = np.zeros((num_output_rec, pcdat['numch']))

        rad_block = (np.dot(scores, PC) + 1) * Pscale
        rad[:, pcdat['band_slices'][b]] = rad_block

    return rad
