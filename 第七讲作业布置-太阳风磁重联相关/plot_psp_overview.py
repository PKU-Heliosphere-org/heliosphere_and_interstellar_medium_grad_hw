'''BY ZIQI WU @ 2025/05/22'''
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates

# MY MODULES
from load_read_psp_data import *
from calc_psp_data import *
from utils import *

# %%
# ==================USER INPUT BEGIN=========================
# --------------------CHOOSE TIME RANGE----------------------
beg_dt = datetime(2021, 4, 29, 0, 20)
end_dt = datetime(2021, 4, 29, 2, 20)
# ---------CHOOSE PAD ENERGY CHANNEL & SET COLOR LIM---------
pad_energy_ev = 314
pad_clim = [9., 10.5]
# ------------------CHOOSE TYPE OF MAG DATA------------------
mag_type = '4sa'
# ==================USER INPUT END===========================

# %%
# ==================READ PSP DATA============================
beg_dt_str = beg_dt.strftime('%Y%m%dT%H%M%S')
end_dt_str = end_dt.strftime('%Y%m%dT%H%M%S')
print('PLOT Time Range: ' + beg_dt_str + '-' + end_dt_str)

# --------------------READ SPE PAD----------------------------
epochpade, timebinpade, epochpade, EfluxVsPAE, PitchAngle, Energy_val = read_spe_data(beg_dt, end_dt, ver='03')

# Set energy channel
pad_energy_ind = np.argmin(abs(Energy_val[0, :] - pad_energy_ev))
pad_energy_ev_str = '%.2f' % Energy_val[0, pad_energy_ind]
print('PAD Energy Channel: ' + pad_energy_ev_str + 'eV')
# Set clim. zmin/max1 for PAD
pad_clim_min = pad_clim[0]
pad_clim_max = pad_clim[1]

# -------------------------READ MAG---------------------------
epochmag, Br, Bt, Bn, Babs = read_mag_data(beg_dt, end_dt, mag_type=mag_type)

# ----------------------READ SPI data------------------------
epochpmom, densp, vp_r, vp_t, vp_n, Tp, EFLUX_VS_PHI_p, PHI_p, T_tensor_p, MagF_inst_p = read_spi_data(beg_dt,
                                                                                                       end_dt,
                                                                                                       species='0',
                                                                                                       is_inst=False)
# -----------------CALCULATE PARAMETERS----------------------
Babs_epochp = interp_epoch(Babs, epochmag, epochpmom)
plasma_beta = calc_plasma_beta(densp, Tp, Babs_epochp)
P_th, P_mag = calc_thermal_and_magnetic_pressure(densp, Tp, Babs_epochp)

# %%
# =======================PLOT OVERVIEW===========================
fig, axs = plt.subplots(5, 1, sharex=True, figsize=(9, 3.5), dpi=200)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.9, left=0.15, hspace=0.1, wspace=0)

axs[0].plot(epochmag, Br, 'k-', linewidth=1, label='Br', zorder=4)
axs[0].plot(epochmag, Bt, 'r-', linewidth=1, label='Bt', zorder=1)
axs[0].plot(epochmag, Bn, 'b-', linewidth=1, label='Bn', zorder=2)
axs[0].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-', label='|B|', zorder=3, linewidth=1)

axs[0].set_ylabel('B\n[nT]')
axs[0].legend(loc=2, bbox_to_anchor=(1.01, 1.1), borderaxespad=0., frameon=False, fontsize=8)
axs[0].tick_params(which='both', top=True, right=True, bottom=True, left=True, direction='in')
axs[0].minorticks_on()

axs[1].plot(epochpmom, vp_r, 'k-', linewidth=1)
axs[1].set_ylim([160, 300])
axs[1].set_ylabel('$V_r$\n$[km/s]$')
axs[1].tick_params(which='both', top=True, right=True, bottom=True, left=True, direction='in')
axs[1].minorticks_on()
plt.xlim(epochpmom[0], epochpmom[-1])

axs[2].plot(epochpmom, densp, 'k-', linewidth=1, zorder=2)
axs[2].set_ylabel('$N_p$\n$[cm^{-3}]$')
axs[2].xaxis.set_minor_locator(AutoMinorLocator())
axs[2].tick_params(which='both', top=True, right=True, bottom=True, left=True, direction='in')
ax2 = axs[2].twinx()
ax2.plot(epochpmom, Tp, 'r-', linewidth=1, zorder=1)
ax2.set_zorder = 1
ax2.set_ylabel('$T_p$\n$[eV]$', color='r')

axs[3].plot(epochpmom, np.log10(plasma_beta), 'k-', linewidth=1, )
axs[3].set_ylabel(r'$\lg\beta$')
locator = mdates.MinuteLocator(interval=30)
formatter = mdates.DateFormatter('%H:%M')
axs[3].xaxis.set_major_locator(locator)
axs[3].xaxis.set_major_formatter(formatter)
axs[3].tick_params(which='both', top=True, right=True, bottom=True, left=True, direction='in')

axs[4].plot(epochpmom, P_mag + P_th, 'k-', linewidth=1, label='$P_{total}$')
axs[4].set_ylabel('Pressure\n[nPa]')
axs[4].stackplot(epochpmom, [P_th, P_mag], labels=['$P_{p,th}$', '$P_{mag}$'], colors=['orange', 'blue'], alpha=0.7)
axs[4].set_xlabel('Time')
axs[4].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0., frameon=False, fontsize=8)
locator = mdates.MinuteLocator(interval=30)
formatter = mdates.DateFormatter('%H:%M')
axs[4].xaxis.set_major_locator(locator)
axs[4].xaxis.set_major_formatter(formatter)
axs[4].tick_params(which='both', top=True, right=True, bottom=True, left=True, direction='in')

plt.show()

# %%
# ==========================DO MVA TEST==========================

w, v, b_L, b_M, b_N, v_L, v_M, v_N = do_mag_MVA(epochmag, np.array(Br), np.array(Bt), np.array(Bn),
                                                epochpmom, np.array(vp_r), np.array(vp_t), np.array(vp_n),
                                                preview=True)
plt.figure()
plt.subplot(311)

plt.plot(epochmag, np.array(Bt), 'r-', label='By', linewidth=0.5)
plt.plot(epochmag, np.array(Bn), 'b-', label='Bz', linewidth=0.5)
plt.plot(epochmag, np.array(Br), 'k-', label='Bx', linewidth=0.5)
plt.plot(epochmag, np.sqrt(b_M ** 2 + b_N ** 2 + b_L ** 2), 'm-', label='|B|', linewidth=0.5)
plt.legend()
plt.ylabel('Bxyz [nT]')

plt.subplot(312)
plt.plot(epochmag, b_M, 'r-', label='BM', linewidth=0.5)
plt.plot(epochmag, b_N, 'b-', label='BN', linewidth=0.5)
plt.plot(epochmag, b_L, 'k-', label='BL', linewidth=1)
plt.legend()
plt.ylabel('BLMN [nT]')

plt.subplot(313)
# plt.plot(epochpmom,v_L,'k-',label='VL')
plt.plot(epochpmom, v_M, 'r-', label='VM', linewidth=0.5)
plt.plot(epochpmom, v_N, 'b-', label='VN', linewidth=0.5)
plt.legend()
plt.ylabel('VMN [km/s]')
ax = plt.twinx()
ax.plot(epochpmom, v_L, 'k-', label='VL', linewidth=1)
plt.legend()
plt.ylabel('VL [km/s]')
plt.show()
