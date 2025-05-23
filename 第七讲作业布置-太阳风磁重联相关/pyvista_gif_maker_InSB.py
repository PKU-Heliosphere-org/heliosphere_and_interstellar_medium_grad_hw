'''BY DIE DUAN'''
import pyspedas
import cdflib
import os
import astropy.constants as constant
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import scipy.interpolate
import scipy.spatial
import pyvista as pv

SC_dir = 'C:/Users/qiyyc/PycharmProjects/sc_data/'
pyspedas.psp.config.CONFIG['local_data_dir'] = SC_dir + 'psp_data/'


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


# make vdf outside SB
spi_l2_dir = r'C:\Users\qiyyc\PycharmProjects\sc_data\psp_data\sweap\spi\l2\spi_sf00_8dx32ex8a'
spi_name = 'psp_swp_spi_sf00_l2_8dx32ex8a_20200128_v04.cdf'
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

# find exact time index and get local B direction
# vdftime = np.datetime64('2019-04-04T19:41:36')
vdftime = np.datetime64('2020-01-28T14:46:10')
tind, ttime = find_nearest(vdf_time, vdftime)

B_trange = [np.datetime_as_string((vdf_time[tind] - np.timedelta64(3500000))),
            np.datetime_as_string((vdf_time[tind] + np.timedelta64(3500000)))]
B_vars = pyspedas.psp.fields(trange=B_trange, datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True,
                             no_update=True)
spi_vars = pyspedas.psp.spi(trange=B_trange, datatype='sf00_l3_mom', level='l3',
                            time_clip=True, no_update=True)
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
temp_vdf = temp_vdf0[good_ind]
speed = speedbin[tind, good_ind]
theta = thetabin[tind, good_ind]
phi = phibin[tind, good_ind]
phi_max = phi.max()
phi_min = phi.min()
theta_max = theta.max()
theta_min = theta.min()

vx_inst = speed * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
vy_inst = speed * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
vz_inst = speed * np.sin(np.deg2rad(theta))
v_sc = np.matmul(rotmat_sc_inst, np.array([vx_inst, vy_inst, vz_inst]))
v_rtn = np.array([-v_sc[2, :], v_sc[0, :], -v_sc[1, :]])

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
levels = [-11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5]
vdf_max_ind = temp_vdf.argmax()
vdf_max_vrtn = v_rtn[:, vdf_max_ind]
print(vdf_max_vrtn)
v0_rtn = v_rtn - vdf_max_vrtn[:, None]
# make linear delaunay interpolation in v0_rtn frame
points = v0_rtn.T
delaunay = scipy.spatial.Delaunay(points)
temp_vdf[vx_inst > -100] = 0
vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)

grid_vr, grid_vt, grid_vn = np.meshgrid(np.linspace(-400, 400, 50),
                                        np.linspace(-400, 400, 50), np.linspace(-400, 400, 50), indexing='ij')

print(grid_vr.shape)
grid_vdf0 = vdf_interp((grid_vr, grid_vt, grid_vn))
grid_vdf_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf0), grid_vdf0.shape)

# make linear delaunay interpolation in v0_mfa frame
points = v0_rtn.T
delaunay = scipy.spatial.Delaunay(points)
temp_vdf[vx_inst > -100] = 0
vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)

grid_vr, grid_vt, grid_vn = np.meshgrid(np.linspace(-400, 400, 50),
                                        np.linspace(-400, 400, 50), np.linspace(-400, 400, 50), indexing='ij')

print(grid_vr.shape)
grid_vdf0 = vdf_interp((grid_vr, grid_vt, grid_vn))
grid_vdf_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf0), grid_vdf0.shape)

# plot in mfa frame
v0_mfa = np.matmul(rotmat_mfa_rtn, v0_rtn)
points = v0_mfa.T
delaunay = scipy.spatial.Delaunay(points)
vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
print(v0_mfa.max(axis=1), v0_mfa.min(axis=1))
grid_vx, grid_vy, grid_vz = np.meshgrid(np.linspace(-400, 400, 100),
                                        np.linspace(-400, 400, 100), np.linspace(-400, 400, 100), indexing='ij')
grid_vdf_mfa0 = vdf_interp((grid_vx, grid_vy, grid_vz))
grid_vdf_mfa_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf_mfa0), grid_vdf_mfa0.shape)

psp_vel0 = psp_vel
rotmat_mfa_rtn0 = rotmat_mfa_rtn
vdf_max_vrtn0 = vdf_max_vrtn
time1 = ttime

# %%
import pyvista as pv

# xv = rv*np.cos(lonv)*np.cos(latv)
# yv = rv*np.sin(lonv)*np.cos(latv)
# zv = rv*np.sin(latv)
#
# Br = (Bx*xv+By*yv+Bz*zv)/np.sqrt(xv**2+yv**2+zv**2)
#
# line = lines_from_points(pos_target)

mesh = pv.StructuredGrid(grid_vr + vdf_max_vrtn[0] + psp_vel[0], grid_vt + vdf_max_vrtn[1] + psp_vel[1],
                         grid_vn + vdf_max_vrtn[2] + psp_vel[2])

vf = np.log10(grid_vdf0)
vf[np.isinf(vf)] = -11.01
vf[vf <= -11] = -11.01
mesh.point_data['values'] = vf.ravel(order='F')  # also the active scalars
pl = pv.Plotter()
pl.open_gif("InSB.gif")
pl.add_title('Inside SB@' + time1.tolist().strftime('%Y-%m-%dT%H:%M:%S'))
surface = mesh.contour(isosurfaces=10, rng=[-11., -7.0])
labels = dict(zlabel='VN (km/s)', xlabel='VR (km/s)', ylabel='VT (km/s)')

v_abs = np.linalg.norm(v_rtn,axis=0)
points = v_rtn[:,v_abs<1000].T


ebin = pv.PolyData(points)
pl.add_mesh(mesh, opacity=0)
labels = dict(zlabel='VN (km/s)', xlabel='VR (km/s)', ylabel='VT (km/s)')#,axes_ranges=[0,800,-400,400,-400,400])
pl.show_grid(**labels)
pl.remove_scalar_bar()
pl.add_mesh(surface, cmap='jet', smooth_shading=True, clim=[-11, -7],
            scalar_bar_args={'title': 'log10[f(v)] (s^3/m^6)'})
pl.add_axes(**labels)
pl.add_points(ebin)

pl.camera_position = 'zx'
pl.camera.roll += 90
di = 2
arrow_cent = vdf_max_vrtn + psp_vel - 165 * rotmat_mfa_rtn[0,:]
arrow_cent[1]+=50
pl.add_arrows(arrow_cent,rotmat_mfa_rtn[0,:],mag=165,color='black')
for i in range(0, int(45 / di)):
    pl.camera.azimuth += di
    pl.write_frame()
for i in range(0, int(45 / di)):
    pl.camera.azimuth -= di
    pl.write_frame()
for i in range(0, int(60 / di)):
    pl.camera.azimuth -= di
    pl.write_frame()
for i in range(0, int(60 / di)):
    pl.camera.azimuth += di
    pl.write_frame()
for i in range(0, 30):
    pl.camera.elevation += di
    pl.write_frame()
for i in range(0, 30):
    pl.camera.elevation -= di
    pl.write_frame()
# pl.camera.azimuth += 50
pl.show()

# %%
# x = np.arange(-10, 10, 0.5)
# y = np.arange(-10, 10, 0.5)
# x, y = np.meshgrid(x, y)
# r = np.sqrt(x**2 + y**2)
# z = np.sin(r)
#
# # Create and structured surface
# grid = pv.StructuredGrid(x, y, z)
#
# # Create a plotter object and set the scalars to the Z height
# plotter = pv.Plotter(notebook=False, off_screen=True)
# plotter.add_mesh(
#     grid,
#     scalars=z.ravel(),
#     lighting=False,
#     show_edges=True,
#     scalar_bar_args={"title": "Height"},
#     clim=[-1, 1],
# )
#
# # Open a gif
# plotter.open_gif("wave.gif")
#
# pts = grid.points.copy()
#
# # Update Z and write a frame for each updated position
# nframe = 15
# for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
#     z = np.sin(r + phase)
#     pts[:, -1] = z.ravel()
#     plotter.update_coordinates(pts, render=False)
#     plotter.update_scalars(z.ravel(), render=False)
#
#     # Write a frame. This triggers a render.
#     plotter.write_frame()
#
# # Closes and finalizes movie
# plotter.close()
