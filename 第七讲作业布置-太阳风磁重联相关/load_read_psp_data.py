'''BY ZIQI WU @ 2025/05/22'''
import os
import numpy as np
from datetime import datetime, timedelta
from spacepy import pycdf

os.environ["CDF_LIB"] = "/usr/local/cdf/lib"
# !!!! CHANGE TO YOUR DATA PATH
psp_data_path = '/Users/ephe/PSP_Data_Analysis/Encounter08/'


def load_RTN_1min_data(start_time_str, stop_time_str):
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [psp_data_path + 'psp_fld_l2_mag_rtn_1min_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
                for x in range(start_time, stop_time + 1)]
    print('Brtn 1min Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def load_RTN_4sa_data(start_time_str, stop_time_str):
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [
        psp_data_path + 'psp_fld_l2_mag_rtn_4_sa_per_cyc_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v02.cdf'
        for x in range(start_time, stop_time + 1)]
    print('Brtn 4sa Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def load_RTN_data(start_time_str, stop_time_str):
    '''psp_fld_l2_mag_rtn_2021042800_v02.cdf'''
    cycles = [0, 6, 12, 18]
    start_time = datetime.strptime(start_time_str, '%Y%m%d%H')
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d%H')
    start_file_time = datetime(start_time.year, start_time.month, start_time.day, cycles[divmod(start_time.hour, 6)[0]])
    stop_file_time = datetime(stop_time.year, stop_time.month, stop_time.day, cycles[divmod(stop_time.hour, 6)[0]])

    if divmod(stop_time.hour, 6)[1] == 0:
        if stop_file_time != start_file_time:
            stop_file_time -= timedelta(hours=6)
    filelist = []
    tmp_time = start_file_time

    while tmp_time <= stop_file_time:
        print('hi')
        filelist.append(psp_data_path + 'psp_fld_l2_mag_rtn_' + tmp_time.strftime('%Y%m%d%H') + '_v02.cdf')
        tmp_time += timedelta(hours=6)
    print('Brtn Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def read_mag_data(beg_dt: datetime, end_dt: datetime, mag_type='1min'):
    if mag_type == '1min':
        mag_RTN = load_RTN_1min_data(beg_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))

        epochmag = mag_RTN['epoch_mag_RTN_1min']
        timebinmag = (epochmag > beg_dt) & (epochmag < end_dt)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN_1min'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    elif mag_type == 'rtn':
        mag_RTN = load_RTN_data(beg_dt.strftime('%Y%m%d%H'), end_dt.strftime('%Y%m%d%H'))

        epochmag = mag_RTN['epoch_mag_RTN']
        timebinmag = (epochmag > beg_dt) & (epochmag < end_dt)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    elif mag_type == '4sa':
        mag_RTN = load_RTN_4sa_data(beg_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))

        epochmag = mag_RTN['epoch_mag_RTN_4_Sa_per_Cyc']
        timebinmag = (epochmag > beg_dt) & (epochmag < end_dt)
        epochmag = epochmag[timebinmag]

        Br = mag_RTN['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][timebinmag, 0]
        Bt = mag_RTN['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][timebinmag, 1]
        Bn = mag_RTN['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][timebinmag, 2]
        Babs = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)
    else:
        print('!!!WRONG MAG TYPE!!! USE "1min"/"rtn"/"4sa".')
        return None
    return epochmag, Br, Bt, Bn, Babs


def load_spe_data(start_time_str, stop_time_str, ver='03'):
    # psp/sweap/spe/psp_swp_spa_sf0_L3_pad_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    filelist = [
        psp_data_path + 'psp_swp_spe_sf0_L3_pad_' + datetime.fromordinal(x).strftime('%Y%m%d') + '_v' + ver + '.cdf'
        for x in range(start_time, stop_time + 1)]
    print('SPE Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def read_spe_data(beg_dt: datetime, end_dt: datetime, ver='03'):
    spe_pad = load_spe_data(beg_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'), ver=ver)
    epochpade = spe_pad['Epoch']
    timebinpade = (epochpade > beg_dt) & (epochpade < end_dt)
    epochpade = epochpade[timebinpade]
    EfluxVsPAE = spe_pad['EFLUX_VS_PA_E'][timebinpade, :, :]
    PitchAngle = spe_pad['PITCHANGLE'][timebinpade, :]
    Energy_val = spe_pad['ENERGY_VALS'][timebinpade, :]
    return epochpade, timebinpade, epochpade, EfluxVsPAE, PitchAngle, Energy_val


def load_spi_data(start_time_str, stop_time_str, inst=False, species='0'):
    # psp/sweap/spi/psp_swp_spi_sf00_L3_mom_INST_20210115_v03.cdf
    start_time = datetime.strptime(start_time_str, '%Y%m%d').toordinal()
    stop_time = datetime.strptime(stop_time_str, '%Y%m%d').toordinal()
    if inst:
        filelist = [psp_data_path + 'psp_swp_spi_sf0' + species + '_l3_mom_INST_' + datetime.fromordinal(x).strftime(
            '%Y%m%d') + '_v03.cdf'
                    for x in range(start_time, stop_time + 1)]
    else:
        filelist = [psp_data_path + 'psp_swp_spi_sf0' + species + '_l3_mom_' + datetime.fromordinal(x).strftime(
            '%Y%m%d') + '_v04.cdf'
                    for x in range(start_time, stop_time + 1)]
    print('SPI Files: ', filelist)
    data = pycdf.concatCDF([pycdf.CDF(f) for f in filelist])
    # print(data)
    return data


def read_spi_data(beg_dt: datetime, end_dt: datetime, species='0', is_inst=False, ):
    if is_inst:
        mom_SPI_ = load_spi_data(beg_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'), inst=is_inst, species=species)
        epochmom_ = mom_SPI_['Epoch']
        timebinmom_ = (epochmom_ > beg_dt) & (epochmom_ < end_dt)
        epochmom_ = epochmom_[timebinmom_]
        dens_ = mom_SPI_['DENS'][timebinmom_]
        v_r_ = mom_SPI_['VEL'][timebinmom_, 0]
        v_t_ = mom_SPI_['VEL'][timebinmom_, 1]
        v_n_ = mom_SPI_['VEL'][timebinmom_, 2]
        T_ = mom_SPI_['TEMP'][timebinmom_]

        return epochmom_, dens_, v_r_, v_t_, v_n_, T_
    else:
        # load proton moms
        mom_SPI_ = load_spi_data(beg_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'), inst=is_inst, species=species)
        epochmom_ = mom_SPI_['Epoch']
        timebinmom_ = (epochmom_ > beg_dt) & (epochmom_ < end_dt)
        epochmom_ = epochmom_[timebinmom_]
        dens_ = mom_SPI_['DENS'][timebinmom_]
        v_r_ = mom_SPI_['VEL_RTN_SUN'][timebinmom_, 0]
        v_t_ = mom_SPI_['VEL_RTN_SUN'][timebinmom_, 1]
        v_n_ = mom_SPI_['VEL_RTN_SUN'][timebinmom_, 2]
        T_ = mom_SPI_['TEMP'][timebinmom_]
        EFLUX_VS_PHI_ = mom_SPI_['EFLUX_VS_PHI'][timebinmom_]
        PHI_ = mom_SPI_['PHI_VALS'][timebinmom_]
        T_tensor_ = mom_SPI_['T_TENSOR_INST'][timebinmom_]
        MagF_inst_ = mom_SPI_['MAGF_INST'][timebinmom_]
        Sun_dist_ = mom_SPI_['SUN_DIST'][timebinmom_]

        return epochmom_, dens_, v_r_, v_t_, v_n_, T_, EFLUX_VS_PHI_, PHI_, T_tensor_, MagF_inst_,  # Sun_dist_
