export data_dir=~/projects/PREFIRE_L1_data

echo 'making PREFIRE_TIRS1_SRF_v11_2022-12-10.nc'
python process_srf.py \
    ${data_dir}/instrument_1_NETD_NEDR_SRF.mat \
    ${data_dir}/PREFIRE_SRF_v11_inst1_wavelengths_2022-12-11.mat \
    ../data/PREFIRE_TIRS1_SRF_v11_FPA_mask.nc \
    ../data/PREFIRE_TIRS1_SRF_v11_2022-12-10.nc

echo 'making PREFIRE_TIRS2_SRF_v11_2022-12-10.nc'
python process_srf.py \
    ${data_dir}/instrument_2_NETD_NEDR_SRF.mat \
    ${data_dir}/PREFIRE_SRF_v11_inst2_wavelengths_2022-12-11.mat \
    ../data/PREFIRE_TIRS2_SRF_v11_FPA_mask.nc \
    ../data/PREFIRE_TIRS2_SRF_v11_2022-12-10.nc
