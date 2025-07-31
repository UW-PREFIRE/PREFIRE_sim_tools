import json, warnings, collections

import netCDF4
import numpy as np

def _validate_shapes(dat, dat_descr, verbose=False):
    """ helper to validate the shapes. dat is a nested dictionary with the
    data arrays; dat_descr is the nested dictionary loaded from the json file.

    this makes sure the arrays in dat have consistent shapes, as linked
    by the dimension names in dat_descr.

    If verbose, it prints a summary of the dimen names it found.

    returns a dictionary, with the keys equal to the dimension names
    (strings) and the values equal to the dimension lengths (integers)
    """

    dim_shapes = collections.defaultdict(set)

    for g_name in dat:

        if g_name not in dat_descr:
            if verbose:
                print('Skipping group: '+g_name+
                      ', was not present in the dat_descr')
            continue

        for v_name in dat[g_name]:
            if v_name not in dat_descr[g_name]:
                if verbose:
                    print('Skipping variable: '+v_name+
                          ', was not present in the dat_descr')
                continue
            for d, dim in enumerate(dat_descr[g_name][v_name]['shape']):
                dim_shapes[dim].add(dat[g_name][v_name].shape[d])

    for dim in dim_shapes:
        if len(dim_shapes[dim]) > 1:
            raise ValueError('dimension ' + dim +
                             ' did not have consistent sizes: '+
                             str(dim_shapes[dim]))
        if verbose:
            print('Dimension : '+dim+', lengths = '+str(dim_shapes[dim]))

    # convert the sets to scalar integers
    out_dim_shapes = {}
    for dim in dim_shapes:
        out_dim_shapes[dim] = dim_shapes[dim].pop()

    return out_dim_shapes


def _write_generic(nc, dat, dat_descrs, verbose=False):
    """
    generic writer.

    Parameters
    ----------
    nc : netCDF4.Dataset
        the tCDF.Dataset object, open to the desired file
    dat : dict
        a nested python dictionary with data to write
    dat_descrs : dict
        the loaded json data description information.
    """
    

    # method:
    # first, validate and identify dimension lengths from the input dat,
    #     using helper function.
    # second, find the actual groups defined in dat.
    # for the defined groups, copy all vars from dat_descrs.
    #     for each var, get the data inside dat; otherwise set to None.
    # Then proceed to create NetCDF4 variables:
    # create group;
    #    create each var:
    #        if the var dimens were located in step 1, then:
    #            set dimens in netCDF4 if needed.
    #            write the data, if present (otherwise, remains Fill)
    #        if the var does not have data:
    #            write fill Value, if possible - this can only 
    #            happen if this var does not include a new dimension.
    

    dim_shapes = _validate_shapes(dat, dat_descrs, verbose=verbose)

    groups_to_write = []
    for g_name in dat:
        if g_name not in dat_descrs:
            warnings.warn(
                'group name ' + g_name + ' in dat'
                ' is not defined in the dat_descr dictionary, and will be skipped')
        else:
            groups_to_write.append(g_name)

    vars_to_write = []
    for g_name in groups_to_write:
        for v_name in dat_descrs[g_name]:
            new_var = {}
            # create a new dictionary, so we can add the data key to it
            new_var.update(dat_descrs[g_name][v_name])
            new_var['fullname'] = g_name + '/' + v_name
            new_var['g_name'] = g_name
            new_var['v_name'] = v_name
            if v_name in dat[g_name]:
                new_var['data'] = dat[g_name][v_name]
            vars_to_write.append(new_var)

    for var in vars_to_write:

        if var['g_name'] not in nc.groups:
            nc.createGroup(var['g_name'])
    
        dims_exist = True
        for d, dimen in enumerate(var['shape']):
            if dimen not in nc.dimensions:
                # create the dimension, if it is in dim_shapes (which is
                # what was able to be defined from the actual input data).
                if dimen in dim_shapes:
                    nc.createDimension(dimen, dim_shapes[dimen])
                else:
                    warnings.warn('cannot write variable ' + var['fullname'] +
                                  ' to file, no variable data was present to ' +
                                  'set the dimension size')
                    dims_exist = False
                    break

        if dims_exist:
            if 'fill_value' in var:
                v_obj = nc.createVariable(
                    var['fullname'], var['dtype'], dimensions=var['shape'], 
                    fill_value = var['fill_value'])
            else:
                v_obj = nc.createVariable(
                    var['fullname'], var['dtype'], dimensions=var['shape'])

            for aname in ("units", "long_name", "description"):
                if aname in var:
                    v_obj.setncattr(aname, var[aname])

            if 'data' in var:
                nc[var['fullname']][:] = var['data']
            

def write_data_fromspec(dat, file_specs_json, output_file, verbose=False):
    """
    write data to file with a given specification.

    Parameters
    ----------
    dat : dict
        a nested python dictionary with data fields
    file_specs_json : str
        path to JSON file with the data specification. It is assumed that
        the keys in dat correspond to this spec file; any mismatched keys
        in dat are simply skipped.
    output_file : str
        path to output file that will be created
    verbose : bool
        controls verbose output printed to console.
    """
    with open(file_specs_json, 'r') as f:
        file_specs = json.load(f)

    with netCDF4.Dataset(output_file, 'w') as nc:
        _write_generic(nc, dat, file_specs)
