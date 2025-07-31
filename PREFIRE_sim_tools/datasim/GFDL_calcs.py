# some misc tools to compute derived quantities from the GFDL fields.
# primarily this is to compute IWP/LWP from the q_i/q_l profiles

import numpy as np

def adjust_q(q, plev, ps):
    """
    adjust GFDL q profile to deal with below-surface data.

    Parameters
    ----------
    q : ndarray
        3D array of q profile (q_i or q_l), shape (k,i,j)
    plev : ndarray
        1D array of pressure levels, shape (k,)
    ps : ndarray
        2D surface pressure, shape (i,j). units must match plev.

    Returns
    -------
    q_adj : ndarray
        3D array of adjusted q values, shaped (k+1,i,j)
    plev_adj : ndarray
        3D array of adjusted plevs, shaped (k+1, i, j).
        Even though the input plev is 1D (the pressure levels have
        no dependence on the lat/lon location), the adj version has
        the ps merged in, which does have a location dependence.
    """


    # deal with near-surface and below-surface data.
    #
    # method: insert a point in the pressure profile at the surface
    # pressure. Use linear interp to get the q value here (this assumes 
    # the first below-surface value is meaningful.)
    # Then, in the 'adjusted' data, set the below surface
    # values to -1, so that later this can be used as a mask to limit the
    # integration.

    q_adj = np.zeros((q.shape[0]+1,) + q.shape[1:]) - 1
    plev_adj = np.zeros((q.shape[0]+1,) + q.shape[1:])

    for i,j in np.ndindex(q.shape[1:]):

        k = plev.searchsorted(ps[i,j], side='right')
        tmp_plev = np.insert(plev, k, ps[i,j])
        plev_adj[:,i,j] = tmp_plev
    
        # use copy() on the profile, otherwise it would change the
        # contents of q.
        q_ij = q[:,i,j].copy()

        q_tmp = np.interp(tmp_plev, plev, q_ij)
        q_tmp[k+1:] = -1
        q_adj[:,i,j] = q_tmp

    return q_adj, plev_adj


def compute_wp(q_adj, plev_adj):
    """
    compute water path, from the adjusted q and plev
    (see outputs from adjust_q()).

    Parameters
    ----------

    q_adj : ndarray
        3D array of q values, assumed units kg / kg (liquid water or ice 
        mass mixing ratios)
    plev_adj : ndarray
        3D array of adjusted pressure levels, units hPa.

    returns
    -------
    wp : ndarray
        2D array of integrated water paths

    """

    # the cloud water/ice contents are in kg/kg,
    # and defined on pressure levels.
    #
    # rearrange hydrostatic equation:
    # atmosphere column mass = integ(- dp / g); then,
    # WP = integ(- q * dp / g) ~ sum(- q * Dp / g)
    # ssume g is constant for simplicity.
    #
    # use MKS units;
    # plev is in hPa, convert to Pa
    g = 9.806

    # L/I WP = Liquid/Ice Water Path, units [kg/m2]
    wp = np.zeros(q_adj.shape[1:])

    for i,j in np.ndindex(wp.shape):
        msk = q_adj[:,i,j] < 0
        if np.any(msk):
            # [0][0] applied to the return from nonzero is the
            # first True in the 1D array
            k = np.nonzero(msk)[0][0]
        else:
            # otherwise, there were no negative values,
            # and the entire set of plev was above the surface
            k = q_adj.shape[0]
        # factor of 100 to convert to MKS (hPa -> Pa)
        wp[i,j] = np.trapz(q_adj[:k,i,j], plev_adj[:k, i, j]*100) / g

    return wp


def compute_ctp(q_adj, plev_adj, tau=1.0, r_eff=10.0, phase='water'):
    """
    compute an approximate cloud top pressure.
    This is the pressure where the cumulative (integrate from TOA downwards)
    optical thickness is some value, default 1.0.
    Assume mono disperse cloud at some r_eff, and use the bulk water/
    ice density for the particle.
    For water clouds, r of 10 or 12 um is probably good;
    for ice clouds, this is much trickier, perhaps 30 is a good choice.

    Parameters
    ----------

    q_adj : ndarray
        3D array of q values, assumed units kg / kg (liquid water or ice 
        mass mixing ratios)
    plev_adj : ndarray
        3D array of adjusted pressure levels, units hPa.
    tau : float
        optical thickness to define cloud "top". default 1.0
    r_eff : float
        particle effective radius, um.
    phase : str
        "water" or "ice", this specifies the particle density.

    returns
    -------
    ctp : ndarray
        2D array of cloud top pressures.
        clear grid cells have a value of -9999 (e.g., those that
        do not integrate to tau.)

    """

    if phase not in ('water', 'ice'):
        raise ValueError('phase must be "water" or "ice"')
    # density, kg/m3
    if phase == 'water':
        rho = 1000.0
    else:
        rho = 917.0
    # particle radius in m
    r = r_eff * 1e-6
    # water path for tau
    L = 2 * rho * r / 3 / tau

    ctp = np.zeros(q_adj.shape[1:]) - 9999

    for i,j in np.ndindex(ctp.shape):
        msk = q_adj[:,i,j] < 0
        if np.any(msk):
            # [0][0] applied to the return from nonzero is the
            # first True in the 1D array
            k = np.nonzero(msk)[0][0]
        else:
            # otherwise, there were no negative values,
            # and the entire set of plev was above the surface
            k = q_adj.shape[0]
        ctp[i,j] = _find_ctp(q_adj[:k,i,j], plev_adj[:k,i,j], L)
    return ctp

def _find_ctp(q, p_hPa, L):

    # this is effectively a cumtrapz.
    g = 9.806
    p = p_hPa * 100.0
    dp = np.diff(p)
    mean_q = (q[1:] + q[:-1])/2
    wp_cumulative = np.r_[0.0, np.cumsum(dp * mean_q)/g]
    #print(wp_cumulative)
    if wp_cumulative[-1] < L:
        return -9999

    ctp = np.interp(L, wp_cumulative, p_hPa)

    return ctp
