"""
Add non-missing, non-zero values to the 'spectral_radiance_unc' field in
simulated 1B-NLRAD granule file(s).

This program requires python version 3.6 or later, and is importable as a 
python module.
"""

# It is recommended to run this from a separate directory (perhaps named
#  "reconstitute-old_simL1B"), with the following file/dir/symlink setup:
#
# L1B_product_filespecs.json -> ../PREFIRE_L1/dist/ancillary/L1B_product_filespecs.json
# PREFIRE_L1 -> /home/ttt/projects/current/PREFIRE_L1/source/python/PREFIRE_L1/
# PREFIRE_PRD_GEN -> /home/ttt/projects/current/PREFIRE_Product_Generator/source/PREFIRE_PRD_GEN/
# add_radunc_to_1B-NLRAD.py    (a copy of this file)
#
#--- FURTHER NOTES:
# SRF files (used below) must be located in the PREFIRE_L1/dist/ancillary/ dir
# Some links above refer to cloned copies of the following:
#   PREFIRE_sim_tools
#   PREFIRE_L1 (requires PREFIRE_PRD_GEN/source/PREFIRE_PRD_GEN symlinked
#               in its source/python/ directory)
#   PREFIRE_PRD_GEN

  # From the Python standard library:
import os
import sys
import datetime
import glob
import json
from pathlib import Path

  # From other external Python packages:
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import netCDF4

  # Custom utilities:
import PREFIRE_tools.utils.unit_conv as uc
import PREFIRE_tools.utils.exceptions as exc
from PREFIRE_tools.utils.time import ctime_to_UTC_DT, ctimeN_to_ctime, \
                                     init_leap_s_for_ctimeRefEpoch, \
                                     UTC_DT_to_ctime
from PREFIRE_tools.utils.filesys import mkdir_p
import PREFIRE_L1.filepaths as L1_fpaths

from PREFIRE_PRD_GEN import file_read
from PREFIRE_PRD_GEN.file_creation import write_data_fromspec
from PREFIRE_PRD_GEN.apply_PRD_GEN import apply_PRD_GEN
import PREFIRE_PRD_GEN.bitflags as bitflags
import PREFIRE_PRD_GEN.filepaths as PRD_GEN_fpaths


#input_dir = "/data/datasim/S06_R00-allsky/by_kind/old-1B-NLRAD"
#output_dir = "/data/datasim/S06_R00-allsky/by_kind/1B-NLRAD"
input_dir = "/data/datasim/S07_R00-allsky/by_kind/1B-NLRAD"
output_dir = "/data/datasim/S07_R00-allsky/by_kind/1B-NLRAD"
SRF_disambig_str = "SRF_v12_2023-08-09"


#--------------------------------------------------------------------------
def add_radunc_to_1B_NLRAD(anchor_path, input_dir, output_dir):
    """Add finite 'spectral_radiance_unc' values in 1B-NLRAD granule file(s)."""
 
    additional_f_description0 = ("The contents of this file are the result "
          "of adding non-missing, non-zero values to the "
          "spectral_radiance_unc field in the original version of this "
          "file, which only had missing values for that field.")

    Geo_gn = "Geometry"
    Rad_gn = "Radiance"

    output_path = os.path.abspath(os.path.join(anchor_path, output_dir))

    # Determine some input filepath information:
    input_path = os.path.abspath(os.path.join(anchor_path, input_dir))
    input_fpath_l = glob.glob(os.path.join(input_path, "PREFIRE_SAT*.nc"))

    filespecs_fpath = os.path.join(PRD_GEN_fpaths.package_ancillary_data_dir,
                                   "final_products_filespecs.json")

    # Load NEdR values from SRF file(s):
    SRF_NEdR_l = []   
    for TIRS_num in [1, 2]:
        TIRS_str = f"TIRS{TIRS_num:1d}"

        SRF_fn = f"PREFIRE_{TIRS_str}_{SRF_disambig_str}.nc"
        SRF_fpath = os.path.join(L1_fpaths.package_ancillary_data_dir, "SRF",
                                 SRF_fn)

        with netCDF4.Dataset(SRF_fpath, 'r') as SRF_ds:
            tmp_array = SRF_ds.variables["NEDR"][...]  # (spectral,xtrack)
            SRF_NEdR_l.append(np.transpose(tmp_array))  # => (xtrack,spectral)

    # Process each input file separately:
    for input_fpath in input_fpath_l:
        print("Working on", input_fpath, "...")

        # Instantiate and initialize output-data dictionary:
        output = {}

        with netCDF4.Dataset(input_fpath, 'r') as L1B_ds:
            # Read global attributes:
            global_atts_d = file_read.load_all_nc4_global_atts(L1B_ds)
            
            # Read "Geometry" group and its group attributes:
            o_d_Geo = file_read.load_all_vars_of_nc4group(Geo_gn, L1B_ds)
            Geo_atts_d = file_read.load_all_atts_of_nc4group(Geo_gn, L1B_ds)

            # Read "Radiance" group:
            o_d_Rad = file_read.load_all_vars_of_nc4group(Rad_gn, L1B_ds)

        #-- "Mine" input filename for information:

        input_fn = os.path.basename(input_fpath)
        fn_tokens = input_fn.split('_')
       
        TIRS_idx = int(fn_tokens[1].replace("SAT", ''))-1

        # Modify fields:

        global_atts_d["additional_file_description"] = additional_f_description0

        shp = o_d_Rad["spectral_radiance_unc"].shape
        o_d_Rad["spectral_radiance_unc"][:,:,:] = (np.ones(shp)*
                                          SRF_NEdR_l[TIRS_idx][np.newaxis,:,:])

        now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
        now_UTC_str = now_UTC_DT.strftime("%Y-%m-%dT%H:%M:%S.%f")
        global_atts_d["UTC_of_file_creation"] = now_UTC_str

        #== Write NetCDF-format file:

        output["Global_Attributes"] = global_atts_d
        output["Geometry"] = o_d_Geo
        output["Geometry_Group_Attributes"] = Geo_atts_d
        output["Radiance"] = o_d_Rad

        output_fpath = os.path.join(output_dir, input_fn)

        mkdir_p(os.path.dirname(output_dir))

        write_data_fromspec(output, output_fpath, filespecs_fpath,
                            use_shared_geometry_filespecs=True, verbose=False)


if __name__ == "__main__":
    # Determine fully-qualified filesystem location of this script:
    anchor_path = os.path.abspath(os.path.dirname(sys.argv[0]))

    # Run driver as many times/ways as needed:
    add_radunc_to_1B_NLRAD(anchor_path, input_dir, output_dir)
