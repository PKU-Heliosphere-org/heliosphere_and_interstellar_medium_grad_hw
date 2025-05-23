'''BY ZIQI WU @ 2025/05/22'''

import numpy as np
import scipy.constants as C
import matplotlib.pyplot as plt
# MY MODULES
from load_read_psp_data import *


def calc_plasma_beta(n_pcm3, T_ev, Babs_nT):
    return 4.03e-11 * n_pcm3 * T_ev / ((Babs_nT * 1e-5) ** 2)


def calc_thermal_and_magnetic_pressure(n_pcm3, T_ev, Babs_nT):
    P_thermal = n_pcm3 * T_ev * 1e6 * C.e * 1e9
    P_magnetic = 3.93 * (Babs_nT * 1e-9) ** 2 * 1.0133e5 * 1e9
    return P_thermal, P_magnetic


def psp_MVA_test(mva_beg_dt, mva_end_dt, mag_type, preview=False):
    mva_epochmag, mva_Br, mva_Bt, mva_Bn, Babs = read_mag_data(mva_beg_dt, mva_end_dt, mag_type=mag_type)
    mva_epochpmom, _, mva_Vr, mva_Vt, mva_Vn, _, _, _, _, _ = read_spi_data(mva_beg_dt, mva_end_dt, species='0')

    w, v, b_L, b_M, b_N, v_L, v_M, v_N = do_mag_MVA(mva_epochmag, np.array(mva_Br), np.array(mva_Bt), np.array(mva_Bn),
                                                    mva_epochpmom, np.array(mva_Vr), np.array(mva_Vt), np.array(mva_Vn),
                                                    preview=preview)
    return w, v, b_L, b_M, b_N, v_L, v_M, v_N


def do_mag_MVA(epochmag, bx, by, bz, epochpmom, vx, vy, vz, preview=False):
    M = np.array([[np.nanmean(bx ** 2) - np.nanmean(bx) ** 2, np.nanmean(bx * by) - np.nanmean(bx) * np.nanmean(by),
                   np.nanmean(bx * bz) - np.nanmean(bx) * np.nanmean(bz)],
                  [np.nanmean(by * bx) - np.nanmean(by) * np.nanmean(bx), np.nanmean(by ** 2) - np.nanmean(by) ** 2,
                   np.nanmean(by * bz) - np.nanmean(by) * np.nanmean(bz)],
                  [np.nanmean(bz * bx) - np.nanmean(bz) * np.nanmean(bx),
                   np.nanmean(bz * by) - np.nanmean(bz) * np.nanmean(by),
                   np.nanmean(bz ** 2) - np.nanmean(bz) ** 2]])

    [w, v] = np.linalg.eig(M)
    arg_w = np.argsort(w)

    e_N = v[arg_w[0]]
    e_M = v[arg_w[1]]
    e_L = v[arg_w[2]]

    print('e_L: ', e_L)
    print('e_M: ', e_M)
    print('e_N: ', e_N)

    b_L = bx * e_L[0] + by * e_L[1] + bz * e_L[2]
    b_M = bx * e_M[0] + by * e_M[1] + bz * e_M[2]
    b_N = bx * e_N[0] + by * e_N[1] + bz * e_N[2]

    v_L = vx * e_L[0] + vy * e_L[1] + vz * e_L[2]
    v_M = vx * e_M[0] + vy * e_M[1] + vz * e_M[2]
    v_N = vx * e_N[0] + vy * e_N[1] + vz * e_N[2]

    return w[arg_w], v[arg_w], b_L, b_M, b_N, v_L, v_M, v_N

