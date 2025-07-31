"""
Reconstitute old-style (circa 2022) simulated-L1B granule file(s) into new-style
 (circa 2023) L1B granule file(s).

This program requires python version 3.6 or later, and is importable as a 
python module.
"""

# It is recommended to run this from a separate directory (perhaps named
#  "reconstitute-old_simL1B"), with the following file/dir/symlink setup:
#
# ETOPO5.nc -> /home/ttt/projects/current/PREFIRE_sim_tools/data/ETOPO5.nc
# inputs -> /data/users/ttt/data-reconstitute-old_simL1B/inputs/
# L1B_product_filespecs.json -> ../PREFIRE_L1/dist/ancillary/L1B_product_filespecs.json
# outputs -> /data/users/ttt/data-reconstitute-old_simL1B/outputs/
# PREFIRE_L1 -> /home/ttt/projects/current/PREFIRE_L1/source/python/PREFIRE_L1/
# PREFIRE_PRD_GEN -> /home/ttt/projects/current/PREFIRE_Product_Generator/source/PREFIRE_PRD_GEN/
# reconstitute_simL1B.py    (a copy of this file)
#
#--- FURTHER NOTES:
# inputs/ contains symlinks to each old simulated L1B file
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
import PREFIRE_L1.unit_conv as uc
import PREFIRE_L1.exceptions as exc
from PREFIRE_L1.utils import mkdir_p, ctime_to_UTC_DT, ctimeN_to_ctime, \
                             init_leap_s_for_ctimeRefEpoch, UTC_DT_to_ctime
import PREFIRE_L1.filepaths as L1_fpaths

from PREFIRE_PRD_GEN.file_creation import write_data_fromspec
from PREFIRE_PRD_GEN.apply_PRD_GEN import apply_PRD_GEN
import PREFIRE_PRD_GEN.bitflags as bitflags
import PREFIRE_PRD_GEN.filepaths as PRD_GEN_fpaths


input_dir = "inputs"
output_dir = "outputs"


#--------------------------------------------------------------------------
def reconst_simL1B_granule(anchor_path, leap_s_info, input_dir, output_dir,
                           file_version_str_t):
    """Reconstitute a simulated-L1B granule file."""
   
    additional_f_description0 = ("The contents of this file are the result "
          "of reconstituting an old-style (circa mid-2022) simulated-L1B "
          "granule file into a new-style (circa April 2023) L1B granule "
          "file.  This conversion involves adding global and group attributes, "
          "adding/modifying variable names and attributes, and massaging "
          "field values and filenames as necessary.")

    Geo_gn = "Geometry"
    Rad_gn = "Radiance"

    output_path = os.path.abspath(os.path.join(anchor_path, output_dir))

    # Determine some input filepath information:
    input_path = os.path.abspath(os.path.join(anchor_path, input_dir))
    input_fpath_l = glob.glob(os.path.join(input_path, "PREFIRE_SAT*"))

    filespecs_fpath = os.path.join(L1_fpaths.package_ancillary_data_dir,
                                   "L1B_product_filespecs.json")

    # Prepare lower-resolution global DEM (ETOPO5, 5-arc-minute lat/lon grid
    #  spacing) dat for interpolation:
    with netCDF4.Dataset("ETOPO5.nc") as D_ds:
        nx_D, ny_D = D_ds.dimensions['X'].size, D_ds.dimensions['Y'].size
        lon_DEM = np.empty((nx_D+1,))
        lat_DEM = np.empty((ny_D+1,))
        DEM = np.empty((ny_D+1, nx_D+1))

        tmp_field = D_ds.variables["X"][...]
        lon_DEM[:-1] = tmp_field[:]  # [deg_E]
        lon_DEM[-1] = 360.   #
        tmp_field = D_ds.variables["Y"][...]
        lat_DEM[:-1] = tmp_field[:]  # [deg_N]
        lat_DEM[-1] = -90.  # [deg_N]
        DEM_coord_t = (lat_DEM, lon_DEM)

        tmp_field = D_ds.variables["elev"][...]
        DEM[:-1,:-1] = tmp_field[:,:]         # [m]
        DEM[:-1,-1] = tmp_field[:,0]          #
        DEM[-1,:] = np.mean(tmp_field[-1,:])  #

        DEM_interp = RegularGridInterpolator(DEM_coord_t, DEM,
                                             method="linear", bounds_error=True)

    # Process each input file separately:
    for input_fpath in input_fpath_l:
        # Instantiate and initialize output-data dictionaries:
        output = {}
        global_atts_d = {}
        Geo_atts_d = {}
        o_d_Geo = {}
        o_d_Rad = {}

        #-- "Mine" input filename for information:

        input_fn = os.path.basename(input_fpath)
        fn_tokens = input_fn.split('_')
       
        TIRS_num = int(fn_tokens[1].replace("SAT", ''))
        global_atts_d["spacecraft_ID"] = f"PREFIRE{TIRS_num:02d}"
        global_atts_d["sensor_ID"] = f"TIRS{TIRS_num:02d}"

        tmp_str = fn_tokens[2].split('-')[3]
        additional_f_description = additional_f_description0
        if tmp_str == "allsky":
            additional_f_description += ("  This version includes all sky "
                                   "conditions (i.e., both cloudy and clear).")
            file_version_str = file_version_str_t[1]
        elif tmp_str == "clrsky":
            additional_f_description += ("  This version includes only clear "
                       "sky conditions, with all nominally-cloudy scenes made "
                       "artificially clear.")
            file_version_str = file_version_str_t[0]
        global_atts_d["additional_file_description"] = additional_f_description

        global_atts_d["full_versionID"] = file_version_str
        global_atts_d["archival_versionID"] = (
                               file_version_str.split('_')[1].replace('R', ''))
        global_atts_d["granule_ID"] = fn_tokens[6].replace(".nc", '')

        #-- Read variables' data from input file, and process accordingly:
        with netCDF4.Dataset(input_fpath, 'r') as in_ds:
            n_atrack = in_ds.dimensions["atrack"].size
            n_xtrack = in_ds.dimensions["xtrack"].size
            n_spectral = in_ds.dimensions["spectral"].size
            UTCvals = in_ds.groups[Geo_gn].variables["time_UTC"][...]
            o_d_Geo["time_UTC_values"] = UTCvals
            UTC_DT_l = []
            o_d_Geo["obs_ID"] = np.zeros((n_atrack,n_xtrack), dtype="int64")
            obsID_fmtstr = (
                      "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}{:01d}{:01d}{:01d}")
            for i in range(n_atrack):
                v = UTCvals[i,:]
                UTC_DT_l.append(datetime.datetime(v[0], v[1], v[2], v[3], v[4],
                                                  v[5], v[6]*uc.int_ms_to_us,
                                                 tzinfo=datetime.timezone.utc))
                
                o_d_Geo["obs_ID"][i,:] = np.array(
                       [obsID_fmtstr.format(v[0], v[1], v[2], v[3], v[4], v[5],
                           v[6]//100, TIRS_num, x+1) for x in range(n_xtrack)],
                                                  dtype="int64")
                           
            UTC_DT = np.array(UTC_DT_l)

            global_atts_d["UTC_coverage_start"] = (
                                  UTC_DT_l[0].strftime("%Y-%m-%dT%H:%M:%S.%f"))
            global_atts_d["UTC_coverage_end"] = (
                                 UTC_DT_l[-1].strftime("%Y-%m-%dT%H:%M:%S.%f"))

            ctime, ctime_minus_UTC = UTC_DT_to_ctime(UTC_DT, 's', leap_s_info)
            o_d_Geo["ctime"] = ctime  # [s]
            o_d_Geo["ctime_minus_UTC"] = ctime_minus_UTC  # [s]
            global_atts_d["ctime_coverage_start_s"] = ctime[0]  # [s]
            global_atts_d["ctime_coverage_end_s"] = ctime[-1]  # [s]

            o_d_Geo["latitude"] = (
                    in_ds.groups[Geo_gn].variables["latitude"][...])  # [deg_N]
            o_d_Geo["longitude"] = (
                   in_ds.groups[Geo_gn].variables["longitude"][...])  # [deg_E]
            o_d_Geo["geoid_latitude"] = (in_ds.groups[Geo_gn].
                                   variables["latitude_geoid"][...])  # [deg_N]
            o_d_Geo["geoid_longitude"] = (in_ds.groups[Geo_gn].
                                  variables["longitude_geoid"][...])  # [deg_E]

            tmp = in_ds.groups[Geo_gn].variables["latitude_vertices"][...]
            new = np.empty(tmp.shape+np.array([0, 0, 1]))
            new[:,:,0:4] = tmp
            new[:,:,4] = tmp[:,:,0]  # Complete polygon-defining set of points
            o_d_Geo["vertex_latitude"] = new  # [deg_N]

            tmp = in_ds.groups[Geo_gn].variables["longitude_vertices"][...]
            new = np.empty(tmp.shape+np.array([0, 0, 1]))
            new[:,:,0:4] = tmp
            new[:,:,4] = tmp[:,:,0]  # Complete polygon-defining set of points
            o_d_Geo["vertex_longitude"] = new  # [deg_E]

              # Surface elevation -- for now, interpolate from a
              #  lower-resolution DEM, and clip elevations to be no less than
              #  0. meters.
            tmp_lon0 = o_d_Geo["longitude"].flatten()  # [deg_E]
            tmp_lon = np.where(tmp_lon0 < 0., tmp_lon0+360.,
                               tmp_lon0)  # convert from +/-180 to 0-to-360
            tmp_elev = np.reshape(
                          DEM_interp((o_d_Geo["latitude"].flatten(), tmp_lon)),
                                              o_d_Geo["latitude"].shape)
            o_d_Geo["elevation"] = np.where(tmp_elev < 0., 0., tmp_elev)
#            o_d_Geo["elevation"] = np.zeros_like(o_d_Geo["latitude"])  # [m]

            o_d_Geo["viewing_zenith_angle"] = (
                 in_ds.groups[Geo_gn].variables["sensor_zenith"][...])  # [deg]

            o_d_Geo["solar_distance"] = np.ones((n_atrack,n_xtrack))  # [AU]

            o_d_Geo["subsat_latitude"] = (in_ds.groups[Geo_gn].
                                  variables["subsat_latitude"][...])  # [deg_N]

            tmp = (in_ds.groups[Geo_gn].
                          variables["subsat_longitude"][...])  # [deg_E, 0-360]
            o_d_Geo["subsat_longitude"] = np.where(tmp >= 180.,
                                           tmp-360., tmp)   # [deg_E, -180-180]

            o_d_Geo["sat_altitude"] = (in_ds.groups[Geo_gn].
                                        variables["sat_altitude"][...])  # [km]

            Geo_atts_d["image_integration_duration_ms"] = 700.7  # [ms]
            Geo_atts_d["solar_beta_angle_deg"] = 0.  # [deg]
            Geo_atts_d["TAI_minus_ctime_at_epoch_s"] = (
                           leap_s_info["ref_ctimeOffsetFromUTC_atEp_s"])  # [s]

            o_d_Rad["obs_ID"] = o_d_Geo["obs_ID"]

            o_d_Rad["detector_ID"] = np.zeros((n_xtrack,n_spectral),
                                              dtype="int16")
            for i in range(n_xtrack):
                o_d_Rad["detector_ID"][i,:] = np.array(
                              [f"{i+1:01d}{x:02d}" for x in range(n_spectral)],
                                                       dtype="int16")

            old_flag = (in_ds.groups[Rad_gn].variables["detector_flag"][...].
                                                                     flatten())
            new_bitflags = np.zeros((n_xtrack*n_spectral,), dtype="uint16")
            new_v = bitflags.set_bit(new_bitflags[0], 0)
            o_d_Rad["detector_bitflags"] = np.reshape(
                                  np.where(old_flag == 1, new_v, new_bitflags),
                                                      (n_xtrack,n_spectral))

            o_d_Rad["wavelength"] = (in_ds.groups[Rad_gn].
                                          variables["wavelength"][...])  # [um]

            tmp = (in_ds.groups[Rad_gn].variables["spectral_radiance"][...])
            o_d_Rad["spectral_radiance"] = np.where(np.isnan(tmp), -9.999e3,
                                                        tmp)  # [W/(m^2 sr um)]

            o_d_Rad["spectral_radiance_unc"] = (in_ds.groups[Rad_gn].
                    variables["spectral_radiance_unc"][...])  # [W/(m^2 sr um)]
    
            o_d_Rad["radiance_obsq_bitflags"] = np.zeros((n_atrack,n_xtrack),
                                                         dtype="uint16")

#^ Check whether a given fields' values are all missing:
#            dd = in_ds.groups[Rad_gn].variables["total_radiance"][...]
#            print ("all are masked",np.all(dd.mask))
#            print(dd[0], np.amin(dd),np.amax(dd))
#
            with open(filespecs_fpath, 'r') as fs:
                f_specs = json.load(fs)
            sgn = "Geometry"
            for key in o_d_Geo:
                print(key,np.amin(o_d_Geo[key]),np.amax(o_d_Geo[key]),
                      f_specs[sgn][key]["fill_value"])
            sgn = "Radiance"
            for key in o_d_Rad:
                print(key,np.amin(o_d_Rad[key]),np.amax(o_d_Rad[key]),
                      f_specs[sgn][key]["fill_value"])
#^

#  Geo
#       land_fraction, "land_fraction" all masked
#       elevation, "altitude" all masked  == now interpolated from low-res DEM
#       elevation_stdev, "altitude_stdev" all masked
#       viewing_azimuth_angle, "sensor_azimuth" all masked
#       solar_zenith_angle, "solar_zenith" all masked
#       solar_azimuth_angle, "solar_azimuth" all masked
#       solar_distance, "solar_distance" all masked == now set to one
#       solar_glint_distance, "solar_glint_distance" all masked
#       solar_glint_latitude, "solar_glint_latitude" all masked
#       solar_glint_longitude, "solar_glint_longitude" all masked
#       solar_beta_angle (att), "solar_beta_angle" all masked == now set to zero
#       sat_solar_illumination_flag, "solar_flag" all_masked
#  
#  Rad
#       total_radiance, "total_radiance" all masked
#       total_radiance_unc, "total_radiance_unc" all masked
        
        global_atts_d["processing_algorithmID"] = "L1 1.3.0"
        global_atts_d["provenance"] = ("1B-RAD  "+file_version_str+
                                       " ( L1 8a9351a1 + PRD_GEN f6e9be00 )")

        outp_fn_base = "PREFIRE_SAT{:01d}_1B-RAD_{}_{}_{}.nc".format(
                 TIRS_num, file_version_str, str(o_d_Rad["obs_ID"][0,0]//1000),
                   global_atts_d["granule_ID"])
        output_fn = "raw-"+outp_fn_base
        output_fpath = os.path.join(output_dir, output_fn)

        global_atts_d["file_name"] = output_fn
        global_atts_d["input_product_files"] = outp_fn_base.replace("1B", "1A")

        now_UTC_DT = datetime.datetime.now(datetime.timezone.utc)
        now_UTC_str = now_UTC_DT.strftime("%Y-%m-%dT%H:%M:%S.%f")
        global_atts_d["UTC_of_file_creation"] = now_UTC_str
        global_atts_d["netCDF_lib_version"] = (
                                            netCDF4.getlibversion().split()[0])

        print(global_atts_d)

        #== Write "raw" NetCDF-format file:

        output["Global_Attributes"] = global_atts_d
        output["Geometry"] = o_d_Geo
        output["Geometry_Group_Attributes"] = Geo_atts_d
        output["Radiance"] = o_d_Rad

        mkdir_p(os.path.dirname(output_fpath))

        write_data_fromspec(output, output_fpath, filespecs_fpath,
                            use_shared_geometry_filespecs=False, verbose=True)

        #== Create final NetCDF-format file using the Product Generator:
        anc_Path = Path(PRD_GEN_fpaths.package_ancillary_data_dir)
        output_Path = Path(output_dir)
        apply_PRD_GEN(output_fpath, output_Path, anc_Path)


if __name__ == "__main__":
    # Determine fully-qualified filesystem location of this script:
    anchor_path = os.path.abspath(os.path.dirname(sys.argv[0]))

    # Initialize leap-second info:
    leap_s_info = init_leap_s_for_ctimeRefEpoch([2000, 1, 1, 0, 0 ,0],
                                            epoch_for_ctime_is_actual_UTC=True)

    # Run driver as many times/ways as needed:
    reconst_simL1B_granule(anchor_path, leap_s_info, input_dir, output_dir,
                           ("S01_R00", "S02_R00"))
