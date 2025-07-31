import numpy as np
import scipy.interpolate

def fixed_pressure_interp(vdict_in, p, intp):
    """
    very simple pressure interp for Aux-Met data, assuming that
    p (the input pressure levels) and intp (the new pressured levels,
    to interpolate the other variables to)
    are fixed for all profiles.

    intp would generally be the 101 level set from PCRTM.
    p would generally be NWP pressure levels, not the terrain
    following pressure coords (that change per locations and Psurf)

    This is currently only applicable to the GFDL pressure-level output
    """
    vdict_out = {}

    # we'll use scipy.interpolate.interp1d here, since
    # it has nice optional keywords to control the extrapolation.
    # for now, we'll use it to extrapolate with a fixed value, using
    # the values at the endpoints.
    ikw = dict(axis=2, kind='linear', fill_value='extrapolate',
               assume_sorted=True)

    p_extrap = np.concatenate([[0.0], p, [1200.0]])

    for k in vdict_in:

        if vdict_in[k].shape[-1] != p.shape[0]:
            vdict_out[k] = vdict_in[k]
            continue

        v = vdict_in[k]
        v = np.concatenate([v[:,:,[0]], v, v[:,:,[-1]]], axis=2)
        F = scipy.interpolate.interp1d(p_extrap, v, **ikw)
        
        vdict_out[k] = F(intp)

    return vdict_out


def pressure_interp(vdict_in, intp, surf_extrap_method=3):
    """
    core interp function.

    assumes variables for input dictionary (e.g. that will be used to create
    Aux-Met), including 'pressure_profile' that should be the per-footprint
    pressure profile.

    all data will pressure dependence (for those variables with the same
    shape as pressure_profile: (i,j,k)) is vertically reinterpolated

    Note this function is not general and is intended to be used to
    interpolate NWP analysis to the PCRTM grid. This means a simple
    constant value extrapolation at TOA (because the top level in PCRTM
    is above the NWP levels), and a special calculation is done to resolve
    the near surface levels, as the PCRTM grid is defined all the way to a
    pressure of 1100 hPa (e.g., below the surface in all conditions.)


    Parameters
    ----------

    vdict_in : dict
        dictionary containing 'pressure_profile', a (a,b,k) shaped array,
        where k is the number of levels, and (a,b) is a 2D array, typically
        (n_alongtrack, n_crosstrack)
        This can then contain arbitrary other keys with ndarray values;
        those with shapes that match pressure_profile are reinterpolated.
        Those without the 3D shape are just copied (by reference) into the
        output dictionary

    intp : ndarray
        the pressure array for the vertical interpolation grid.
        this must be 1D and has shape (ki,)

    surf_extrap_method : int
        optional flag to control the surface extrapolation method.
        0 : copy the value from the lowest pressure level in the input.
        1 : linearly extrapolate using a linear fit to the bottom 4 levels.
        2 : compute the value needed to make the layer averages match.
        3 : combine methods 2 and 0, based on the fraction of the bottom layer
        that is contained within the input pressure profile. The fraction is
        used as a linear weight between the two: for the bottom layer fraction
        approaching zero, method 0 is used; if the bottom layer fraction
        is near one, method 3 is used.
        This is the default method.

    Returns
    -------

    vdict_out : dict
        dictionary with the same list of keys as the inputs; non-3D ndarray
        will be by-reference copies of the keys in vdict_in; 3D ndarray will
        be interpolated so that the new shapes are (a,b,ki).
        The output value for 'pressure_profile' will be the 1D ndarray
        given as input for intp (the interpolation grid pressure levels).

    """

    shape3D = vdict_in['pressure_profile'].shape
    n1, n2, nk = shape3D
    shape3D_new = (n1, n2, intp.shape[0])

    vdict_out = {}
    var_list = []
    for v in vdict_in:
        if v == 'pressure_profile':
            continue
        if vdict_in[v].shape == shape3D:
            var_list.append(v)
            vdict_out[v] = np.zeros(shape3D_new)
        else:
            vdict_out[v] = vdict_in[v]
            
    for a, b in np.ndindex((n1, n2)):
        p = vdict_in['pressure_profile'][a,b,:]
        for v in var_list:
            # ToDo: this could be replaced with a weights calculation,
            # which is then applied to each entry of var_list
            # would be more efficient.
            vdict_out[v][a,b,:] = _interp_helper(
                vdict_in[v][a,b,:], p, intp, surf_extrap_method)

    vdict_out['pressure_profile'] = intp

    return vdict_out


def _interp_helper(v, p, intp, surf_extrap_method=3):
    """
    helper to actually do the interpolation on a single profile.
    inputs are similar to np.interp (but in a different order):
    v: profile; p: pressure levels; intp: the interpolation pressure grid.
    """
    below_surf_level = intp.searchsorted(p[-1])
    below_surf_p = intp[below_surf_level]
    if surf_extrap_method == 0:
        below_surf_v = v[-1]
    elif surf_extrap_method == 1:
        near_surf_p = p[-4:]
        near_surf_v = v[-4:]
        near_surf_linfit = np.polyfit(near_surf_p, near_surf_v, 1)
        below_surf_v = np.polyval(near_surf_linfit, intp[below_surf_level])
    elif surf_extrap_method in (2,3):
        intp1 = intp[below_surf_level-1]
        intp2 = intp[below_surf_level]
        full_layer_dp = intp2 - intp1

        # get layer average in the partial bottom layer: use a trapz
        # from the last above-surf layer, through each level in p,
        # to the highest pressure in p.
        k = p.searchsorted(intp1)
        p_tmp = np.concatenate([[intp1], p[k:]])
        dp_tmp = (p_tmp[-1]-p_tmp[0])
        v_tmp = np.interp(p_tmp, p, v)
        layer_avg = np.trapz(v_tmp, p_tmp) / dp_tmp
        # now compute the value at the below surface level that is
        # needed to get the same layer average for v.
        v_slope = (layer_avg - v_tmp[0]) / (0.5*dp_tmp)
        below_surf_v = v_slope * full_layer_dp + v_tmp[0]

        # if this was method 2, we are done. if 3, then
        # average this with method 0.
        # use a linear weighting according to the fraction of
        # the interp layer that exists within p
        if surf_extrap_method == 3:
            f = dp_tmp / full_layer_dp
            below_surf_v = below_surf_v*f + v[-1]*(1-f)
    else:
        raise ValueError('unknown surf extrap method: '+
                         str(surf_extrap_method))

    p_aug = np.concatenate([p, [below_surf_p]])
    v_aug = np.concatenate([v, [below_surf_v]])

    v_intp = np.interp(intp, p_aug, v_aug)

    return v_intp


def compute_layer_avg(v, p, layerp):
    """
    compute layer average values from array v, which is some quantity
    that is a function of p (pressure).
    v and p must be 1D arrays of the same length.

    The layerp is a set of layers, where trapezoidal integration
    will be performed, to get the average value of v.
    Linear interpolation of v(p) is performed to get to the 
    layer boundary values.
    layerp is a 2D array, shaped (n,2) for the level boundaries
    of n layers. Column 0 is assumed to be the upper level (e.g.,
    the lower pressure).

    if a layer is only partially filled, the average value
    is computed for only the valid part. If a layer has no
    values, it will be NaN in the output.
    """

    union_p = np.union1d(p, layerp.flatten())
    union_v = np.interp(union_p, p, v)

    dp = layerp[:,1] - layerp[:,0]
    nlayer = dp.shape[0]

    layer_bdry = union_p.searchsorted(layerp)
    layer_v = np.zeros(nlayer)

    # layer_flag is constructed so that a layer that exists
    # completed outside p has a value of 2; a partial overlap
    # has a value of 1; and layers fully inside p have 0.
    layer_flag = np.zeros(nlayer, np.int)
    layer_flag += (layerp <= p[0]).sum(1)
    layer_flag += (layerp >= p[-1]).sum(1)
    level_msk = np.logical_and(union_p >= p[0], union_p <= p[-1])

    for n in range(nlayer):

        s = slice(layer_bdry[n,0], layer_bdry[n,1]+1)
        if layer_flag[n] == 0:
            # normal interior level
            layer_v[n] = np.trapz(union_v[s], union_p[s]) / dp[n]
        elif layer_flag[n] == 1:
            # partially overlapped layer: take the valid subset.
            msk_part = level_msk[s]
            v_tmp = union_v[s][msk_part]
            p_tmp = union_p[s][msk_part]
            dp_tmp = p_tmp[-1] - p_tmp[0]
            layer_v[n] = np.trapz(v_tmp, p_tmp) / dp_tmp
        else:
            # should be == 2, this means no overlap at all.
            layer_v[n] = np.nan

    return layer_v
