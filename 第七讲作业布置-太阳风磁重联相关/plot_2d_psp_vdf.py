'''BY ZIQI WU @ 2025/05/22, BASED ON CODES BY DIE DUAN'''
import pyspedas
import cdflib
import os
import astropy.constants as constant
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import scipy.interpolate
import scipy.spatial

# MY MODULES
from load_read_psp_data import *
from utils import *
from calc_psp_data import *

### USER INPUT BEGIN ###
SC_dir = '/Users/ephe/PSP_Data_Analysis/'
pyspedas.psp.config.CONFIG['local_data_dir'] = SC_dir

# load SPI and MAG data for plotting
psp_trange = ['2022-02-25/12:20', '2022-02-25/12:40']
psp_trange = ['2022-02-17/13:48','2022-02-18/01:40']
psp_trange = ['2022-12-12/03:00','2022-12-12/12:00']

### USER INPUT END ###

t_range = psp_trange
B_vars = pyspedas.psp.fields(trange=t_range, datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True)
spi_vars = pyspedas.psp.spi(trange=t_range, datatype='sf00_l3_mom', level='l3',
                            time_clip=True)

# import pytplot
# import astropy.constants as const
# from scipy.interpolate import interp1d


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def B_2_n_interp(Btime, Bx, By, Bz, n_time):
    from scipy.interpolate import interp1d
    Bx_interp = interp1d(Btime, Bx, fill_value='extrapolate')
    By_interp = interp1d(Btime, By, fill_value='extrapolate')
    Bz_interp = interp1d(Btime, Bz, fill_value='extrapolate')
    Bx_new = Bx_interp(n_time)
    By_new = By_interp(n_time)
    Bz_new = Bz_interp(n_time)
    return Bx_new, By_new, Bz_new


def griddata_tri(data, x, y):
    cart_temp = np.array([x, y])
    points = np.stack(cart_temp).T
    delaunay = scipy.spatial.Delaunay(points)
    return scipy.interpolate.LinearNDInterpolator(delaunay, data)



# =====INPUT BEGIN=====
beg_dt = datetime(2022,12,12,3)
end_dt = datetime(2022,12,12,12)
pad_energy_ev = 314
pad_clim = [9., 10.5]
pad_norm_clim = [-1.5, -0.]
i_encounter = 8
mag_type = '4sa'
inst = False
# =====INPUT END=====

beg_dt_str = beg_dt.strftime('%Y%m%dT%H%M%S')
end_dt_str = end_dt.strftime('%Y%m%dT%H%M%S')

print('Time Range: ' + beg_dt_str + '-' + end_dt_str)

## ------READ SPE PAD-----
epochpade, timebinpade, epochpade, EfluxVsPAE, PitchAngle, Energy_val = read_spe_data(beg_dt, end_dt, ver='04')
# Calculate normalized PAD
norm_EfluxVsPAE = EfluxVsPAE * 0
for i in range(12):
    norm_EfluxVsPAE[:, i, :] = EfluxVsPAE[:, i, :] / np.nansum(EfluxVsPAE, 1)
# Choose energy channel
pad_energy_ind = np.argmin(abs(Energy_val[0, :] - pad_energy_ev))
pad_energy_ev_str = '%.2f' % Energy_val[0, pad_energy_ind]
print('PAD Energy Channel: ' + pad_energy_ev_str + 'eV')
# Choose clim. zmin/max1 for PAD; zmin/max2 for norm_PAD
pad_clim_min = pad_clim[0]
pad_clim_max = pad_clim[1]
pad_norm_clim_min = pad_norm_clim[0]
pad_norm_clim_max = pad_norm_clim[1]

##  -----READ MAG-----
epochmag, Br, Bt, Bn, Babs = read_mag_data(beg_dt, end_dt, mag_type=mag_type)

## -----READ SPI data-----
epochpmom, densp, vp_r, vp_t, vp_n, Tp, EFLUX_VS_PHI_p, PHI_p, T_tensor_p, MagF_inst_p = read_spi_data(beg_dt,
                                                                                                       end_dt,
                                                                                                       species='0',
                                                                                                       is_inst=False)
Babs_epochp = interp_epoch(Babs, epochmag, epochpmom)
plasma_beta = calc_plasma_beta(densp, Tp, Babs_epochp)
# %%


# %%
import matplotlib.gridspec as gridspec

# make vdf inside SB
spi_l2_dir = '/Users/ephe/PSP_Data_Analysis/Encounter08/'
spi_name = 'psp_swp_spi_sf00_l2_8dx32ex8a_20221212_v04.cdf'
spi_file = os.path.join(spi_l2_dir, spi_name)

data = cdflib.CDF(spi_file)
epoch = data.varget('Epoch')
vdf_time = cdflib.cdfepoch.to_datetime(epoch, to_np=True)
energybin = data.varget('ENERGY')  # 'UNITS': 'eV'
thetabin = data.varget('THETA')
phibin = data.varget('PHI')
rotmat_sc_inst = data.varget('ROTMAT_SC_INST')
eflux = data.varget('EFLUX')  # Units: 'eV/cm2-s-ster-eV'
speedbin = np.sqrt(2 * energybin * constant.e.value / constant.m_p.value) / 1000  # km/s

nflux = eflux / energybin
vdf = nflux / constant.e.value * constant.m_p.value / speedbin ** 2 / 100  # s^3/m^6
import pytplot


def plot_vdf_frame(vdftime, i_, frame='rtn', pos=[0, 0], color='r', ):
    ftsize = 10
    # 将下面的原始图代码改为subfigure版本
    fig = plt.figure(dpi=100, figsize=(6, 9), constrained_layout=False)
    # fig,axs = plt.subplots(figsize=(12,10))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)
    spec = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0], hspace=0.01)
    spec2 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1], wspace=0.4, width_ratios=[1, 1, 1, 0.05],
                                             hspace=0.08)

    axs = []
    x_label = 0.04
    # i = 0
    # axs.append(fig.add_subplot(spec[i, :]))
    # axs[i].plot(epochpmom, r_psp_carr_pmom, 'k-', linewidth=1)
    # ax2 = axs[i].twinx()
    # ax2.plot(epochpmom, np.rad2deg(lon_psp_carr_pmom), 'r-', linewidth=1)
    # ax2.set_ylabel('Carrington\n Longitude (deg)')
    # axs[i].set_ylabel('Radial\n Distance (Rs)')
    # axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    # plt.text(x_label, 0.1, '(a)', transform=plt.gca().transAxes,
    #          fontdict=dict(fontsize=10, color='k', weight='semibold'))

    i = 0
    axs.append(fig.add_subplot(spec[i, :]))
    pos = axs[i].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, pad_energy_ind])).T,
                            cmap='jet', vmax=pad_norm_clim_max, vmin=pad_norm_clim_min)
    axs[i].set_ylabel('Pitch\n Angle (deg)')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.1, '(b) e-PAD (' + pad_energy_ev_str + ' eV)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=10, color='w', weight='semibold'))

    i = 1
    axs.append(fig.add_subplot(spec[i, :]))
    axs[i].plot(epochmag, Br, 'k-', label='Br', zorder=4)
    # axs[i].plot(epochmag, Bt, 'r-', label='Bt', zorder=1)
    # axs[i].plot(epochmag, Bn, 'b-', label='Bn', zorder=2)
    axs[i].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-', label='|B|', zorder=3)
    axs[i].set_ylabel('B\n(nT)')
    axs[i].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0., fontsize=8)
    # axs[i].plot([dt_tmp, dt_tmp], [-400, 400], 'r--')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.3, '(c)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=10, color='k', weight='semibold'))

    i = 2
    axs.append(fig.add_subplot(spec[i, :]))
    axs[i].plot(epochpmom, vp_r, 'k-')
    # ax5.plot([dt_tmp, dt_tmp], [-3., 3.], 'r--')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].set_ylabel(r'$V_r (km/s)$')
    # ax5.set_xlabel('Time', fontsize=8)
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.7, '(d)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=10, color='k', weight='semibold'))

    i = 3
    axs.append(fig.add_subplot(spec[i, :]))

    ax2 = axs[i].twinx()
    ax2.plot(epochpmom, Tp, 'r-', linewidth=1)
    ax2.set_ylabel('$T_p$ \n $(eV)$', color='r')
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])

    axs[i].plot(epochpmom, densp, 'k-', linewidth=1)
    axs[i].set_ylabel('$N_p$ \n$(cm^{-3})$')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    plt.text(x_label, 0.1, '(e)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=10, color='k', weight='semibold'))

    i = 4
    axs.append(fig.add_subplot(spec[i, :]))
    axs[i].plot(epochpmom, np.log10(plasma_beta), 'k-')
    # axs[i].plot([dt_tmp, dt_tmp], [-3., 3.], 'r--')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())

    axs[i].set_ylabel(r'$\lg\beta$')
    axs[i].set_xlabel('Time (UTC)', fontsize=ftsize)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.7, '(f)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=10, color='k', weight='semibold'))

    tind, ttime = find_nearest(vdf_time, vdftime)
    axs[i].axvline(x=ttime, ymax=6, c="blue", linewidth=2, zorder=0, clip_on=False, linestyle='--')

    B_trange = [np.datetime_as_string((vdf_time[tind] - np.timedelta64(3500000))),
                np.datetime_as_string((vdf_time[tind] + np.timedelta64(3500000)))]
    B_vars = pyspedas.psp.fields(trange=B_trange, datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True,
                                 no_update=True)

    B_vec = pytplot.get(B_vars[0])

    print(ttime)
    B_ave_rtn = B_vec.y.mean(axis=0)
    B_ave_abs = np.linalg.norm(B_ave_rtn)
    print(B_ave_rtn)
    B_para_rtn = B_ave_rtn / B_ave_abs
    print(B_para_rtn)
    B_perp2_rtn = np.cross(B_para_rtn, np.array([-1, 0, 0]))
    B_perp2_rtn = B_perp2_rtn / np.linalg.norm(B_perp2_rtn)
    print(B_perp2_rtn)
    B_perp1_rtn = np.cross(B_perp2_rtn, B_para_rtn)
    B_perp1_rtn = B_perp1_rtn / np.linalg.norm(B_perp1_rtn)
    print(B_perp1_rtn)
    rotmat_mfa_rtn = np.array([B_para_rtn, B_perp1_rtn, B_perp2_rtn])
    np.matmul(rotmat_mfa_rtn, np.array([1, 0, 0]))
    # calc vxyz in inst, sc and rtn frame
    temp_vdf0 = vdf[tind, :]
    good_ind = (~np.isnan(temp_vdf0)) & (temp_vdf0 > 0)

    speed = speedbin[tind, good_ind]
    theta = thetabin[tind, good_ind]
    phi = phibin[tind, good_ind]

    vx_inst = speed * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    vy_inst = speed * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    vz_inst = speed * np.sin(np.deg2rad(theta))
    v_sc = np.matmul(rotmat_sc_inst, np.array([vx_inst, vy_inst, vz_inst]))
    v_rtn = np.array([-v_sc[2, :], v_sc[0, :], -v_sc[1, :]])

    temp_vdf = temp_vdf0[good_ind]

    vdf_max_ind = temp_vdf.argmax()
    vdf_max_vrtn = v_rtn[:, vdf_max_ind]

    v0_rtn = v_rtn - vdf_max_vrtn[:, None]

    temp_vdf[vx_inst > -100] = 0
    temp_vdf[vx_inst > -100] = 0

    # plot in mfa frame
    if frame == 'mfa':
        v0_mfa = np.matmul(rotmat_mfa_rtn, v0_rtn)
        points = v0_mfa.T
        delaunay = scipy.spatial.Delaunay(points)
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
        print(v0_mfa.max(axis=1), v0_mfa.min(axis=1))
        grid_vx, grid_vy, grid_vz = np.meshgrid(np.linspace(-400, 400, 100),
                                                np.linspace(-400, 400, 100), np.linspace(-400, 400, 100), indexing='ij')
        grid_vdf_mfa1 = vdf_interp((grid_vx, grid_vy, grid_vz))
        grid_vdf_mfa_max_ind1 = np.unravel_index(np.nanargmax(grid_vdf_mfa1), grid_vdf_mfa1.shape)

        time2 = ttime

        levels = [-11.5, -11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5]
        time = time2.tolist().strftime('%H:%M:%S')

        grid_vdf = grid_vdf_mfa1
        grid_vdf_max_ind = grid_vdf_mfa_max_ind1
        print(grid_vdf_max_ind)
        axs.append(fig.add_subplot(spec2[:, :]))
        axs[i_].contourf(grid_vx[:, grid_vdf_max_ind[1], :], grid_vz[:, grid_vdf_max_ind[1], :],
                         np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]), levels,
                         cmap='jet')
        axs[i_].set_xlabel('$V_\\parallel$ (km/s)')
        axs[i_].set_ylabel('$V_{\\perp2}$ (km/s)')
        axs[i_].set_aspect('equal', adjustable='box')
        axs[i_].set_xlim((-400, 400))
        axs[i_].set_ylim((-400, 400))
        axs[i_].set_title('@' + time, color=color)
    elif frame == 'rtn':
        temp_vdf = temp_vdf0[good_ind]
        psp_vel = pytplot.get('psp_spi_SC_VEL_RTN_SUN')
        psp_vel_vec = psp_vel.y
        psp_vel = psp_vel_vec.mean(axis=0)
        # make linear delaunay interpolation in rtn frame
        cart_temp = v_rtn
        points = np.stack(cart_temp).T
        delaunay = scipy.spatial.Delaunay(points)
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
        # plot vdf in v-vmax frame (eliminate bulk speed)
        levels = [-11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6., -5.5, -5.]
        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]
        print(vdf_max_vrtn)
        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        # make linear delaunay interpolation in v0_rtn frame
        points = v0_rtn.T
        delaunay = scipy.spatial.Delaunay(points)
        temp_vdf[vx_inst > -100] = 0
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)

        grid_vr, grid_vt, grid_vn = np.meshgrid(np.linspace(-800, 800, 200),
                                                np.linspace(-500, 500, 125),
                                                np.linspace(-500, 500, 125), indexing='ij')

        print(grid_vr.shape)
        grid_vdf0 = vdf_interp((grid_vr, grid_vt, grid_vn))
        grid_vdf_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf0), grid_vdf0.shape)

        grid_vdf = grid_vdf0
        grid_vdf_max_ind = grid_vdf_max_ind0

        axs.append(fig.add_subplot(spec2[:, :]))
        psp_vel = psp_vel.value

        pos = axs[i_].contourf(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                              grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                              np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                              levels, cmap='jet', extend='both')
        axs[i_].contour(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                       grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                       np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                       levels, colors='k',
                       linewidths=.5, linestyles='solid', negative_linestyles='solid')

        axs[i_].arrow(vdf_max_vrtn[0] + psp_vel[0],  # - 110 * rotmat_mfa_rtn[0, 0],
                     vdf_max_vrtn[2] + psp_vel[2],  # - 110 * rotmat_mfa_rtn[0, 2],
                     110 * rotmat_mfa_rtn[0, 0],
                     110 * rotmat_mfa_rtn[0, 2],
                     width=10., head_width=30., color='c')
        axs[i_].text(vdf_max_vrtn[0] + psp_vel[0],
                    vdf_max_vrtn[2] + psp_vel[2] + 50,
                    r'$\vec{B}$', color='c')
        axs[i_].set_xlabel('$V_R$ (km/s)')
        axs[i_].set_ylabel('$V_N$ (km/s)')

        axs[i_].set_aspect('equal', adjustable='box')
        axs[i_].set_xlim((0, 900))
        axs[i_].set_ylim((-450, 450))
        plt.colorbar(pos)
        axs[i_].set_title('$v_R-v_N$ plane', color='b')


# %%
vdftime_full_list = [np.datetime64('2022-12-12T03:00:00') + np.timedelta64(180, 's') * n for n in range(180)]
EXPORT_PATH = 'export/work/tres_hcs_crossings/plot_vdf_movie/E14/'
os.makedirs(EXPORT_PATH, exist_ok=True)
for i in range(180):
    plot_vdf_frame(vdftime_full_list[i], 5, frame='rtn')
    time_str = vdftime_full_list[i].tolist().strftime('%H:%M:%S')
    plt.savefig(EXPORT_PATH + 'HCS1_' + time_str + '_vdf_rtn.png')
    plt.close()
# %%
from utils import folder_to_movie

folder_to_movie(EXPORT_PATH, filename_pattern='HCS1_(.+)_vdf_rtn.png',
                time_format='%H:%M:%S', export_pathname=EXPORT_PATH + 'vdf_overview_rtn_HCS1',
                video_format='.mp4')

