
from .constants import *
from .solar_wind_model import solar_wind_velocity, calculate_solar_wind_density, calculate_magnetic_field, calculate_alfven_speed, calculate_sound_speed
from .wave_dispersion import calculate_wave_omega

import numpy as np
import logging
logger = logging.getLogger(__name__)

class Solver:

    def __init__(self, config, mode='linear', to_be_iterate=False, dt_order:int=3):
        self.config = config
        self.t = 0

        if mode == 'linear' or 'log':
            self.mode = mode
        else:
            raise ValueError("The equation mode should only be \'linear\' or \'log\' !")

        self.setup_grids()
        self.initialize_wave_power()
        self.calculate_solar_wind_velocity()

        self.r.requires_grad = True
        self.theta.requires_grad = True
        self.k.requires_grad = True
        self.theta_k.requires_grad = True
        self.calculate_solar_wind_density()
        self.calculate_magnetic_field()
        self.calculate_alfven_speed()
        self.calculate_sound_speed()
        self.calculate_omega_derivatives()
        self.r.requires_grad = False
        self.theta.requires_grad = False
        self.k.requires_grad = False
        self.theta_k.requires_grad = False

        self.calculate_wave_power_derivatives()
        self.calculate_wave_power_time_variation()
        if to_be_iterate:
            while self.t < self.config['t_max']:
                logger.info(f"Time: {self.t:.2f} / {self.config['t_max']:.2f}")
                self.iterate(order=dt_order)
    
    def setup_grids(self):
        self.r, self.theta, self.k, self.theta_k = \
        torch.meshgrid(torch.linspace(self.config['r_min'], self.config['r_max'], self.config['n_r'], device=DEVICE),
                       torch.arange(0, 2 * np.pi, self.config['dtheta'], device=DEVICE),
                       torch.linspace(self.config['k_min'],
                                      self.config['k_max'],
                                      self.config['n_k'], device=DEVICE),
                       torch.arange(0, 2 * np.pi, self.config['dtheta_k'], device=DEVICE),
                       indexing='ij')
        self.dr = (self.config['r_max'] - self.config['r_min']) / (self.config['n_r'] - 1)
        self.dtheta = self.config['dtheta']
        self.dk = (self.config['k_max'] - self.config['k_min']) / (self.config['n_k'] - 1)
        self.dtheta_k = self.config['dtheta_k']
        
    def initialize_wave_power(self, func=None):
        """
        func: function to initialize the wave power spectrum
        """
        if self.mode == 'linear':  # Linear Equation
            if func is None:
                logger.info("No function provided for wave power initialization. Defaulting to Gaussian.")
                self.wave_power = torch.tensor(self.config['initial_amplitude'], device=DEVICE) \
                    * (self.r / torch.tensor(SOLAR_RADIUS, device=DEVICE))**(-3) \
                    * self.k**(-5/3) \
                    / (np.sqrt(2 * torch.pi) * torch.tensor(self.config['initial_sigma_thetak'], device=DEVICE)) \
                    * torch.exp(-(self.k * self.theta_k.cos())**2 / (2 * torch.tensor(self.config['initial_sigma_thetak'], device=DEVICE)**2))
            else:
                self.wave_power = func(self.r, self.theta, self.k, self.theta_k, mode=self.mode)
        else:  # Log Equation
            if func is None:
                logger.info("No function provided for wave power initialization. Defaulting to Gaussian.")
                self.wave_power = torch.tensor(self.config['initial_amplitude'], device=DEVICE).log() \
                    + (-3) * (self.r.log() - torch.tensor(SOLAR_RADIUS, device=DEVICE).log()) \
                    + (-5/3) * self.k.log() \
                    - (np.sqrt(2 * torch.pi) * torch.tensor(self.config['initial_sigma_thetak'], device=DEVICE)).log() \
                    + (-(self.k * self.theta_k.cos())**2 / (2 * torch.tensor(self.config['initial_sigma_thetak'], device=DEVICE)**2))
                # self.wave_power = torch.tensor(self.config['initial_amplitude'], device=DEVICE) \
                #     * (self.r / torch.tensor(SOLAR_RADIUS, device=DEVICE))**(-3) \
                #     * self.k**(-5/3) \
                #     / (np.sqrt(2 * torch.pi) * torch.tensor(self.config['initial_sigma_thetak'], device=DEVICE)) \
                #     * torch.exp(-(self.k * self.theta_k.cos())**2 / (2 * torch.tensor(self.config['initial_sigma_thetak'], device=DEVICE)**2))
                # self.wave_power = self.wave_power.log()
            else:
                self.wave_power = func(self.r, self.theta, self.k, self.theta_k, mode=self.mode)
    
    def calculate_solar_wind_velocity(self):
        """
        Calculate the solar wind velocity.
        """
        r = np.linspace(self.config['r_min'], self.config['r_max'], self.config['n_r'])
        v_sw, r_crit, v_crit = solar_wind_velocity(r, temperature=self.config['temperature'])
        v_sw_surface, _, _ = solar_wind_velocity(SOLAR_RADIUS, temperature=self.config['temperature'])
        v_sw = np.expand_dims(v_sw, axis=(1, 2, 3))
        for axis in (1, 2, 3):
            v_sw = np.repeat(v_sw, self.r.shape[axis], axis=axis)
        self.v_sw = torch.tensor(v_sw, device=DEVICE)
        self.v_sw_surface = torch.tensor(v_sw_surface, device=DEVICE)
        self.r_crit = torch.tensor(r_crit, device=DEVICE)
        self.v_crit = torch.tensor(v_crit, device=DEVICE)
    
    def calculate_solar_wind_density(self):
        self.n_sw = calculate_solar_wind_density(self.r, self.v_sw, self.v_sw_surface)
    
    def calculate_magnetic_field(self):
        self.B, self.theta_B = calculate_magnetic_field(self.r, self.theta, self.v_sw)

    def calculate_alfven_speed(self):
        self.vA, self.theta_vA = calculate_alfven_speed(self.B, self.theta_B, self.n_sw)

    def calculate_sound_speed(self):
        self.c_s = calculate_sound_speed(self.config['temperature'])

    def set_boundary(self):
        """
        Set the boundary conditions for the wave power.
        """
        # r boundary condition is zero gradient
        self.wave_power[0, :, :, :] = self.wave_power[1, :, :, :]
        self.wave_power[-1, :, :, :] = self.wave_power[-2, :, :, :]
        # theta boundary condition is periodic
        self.wave_power[:, 0, :, :] = self.wave_power[:, -1, :, :]
        # k boundary condition is zero gradient
        self.wave_power[:, :, 0, :] = self.wave_power[:, :, 1, :]
        self.wave_power[:, :, -1, :] = self.wave_power[:, :, -2, :]
        # theta_k boundary condition is periodic
        self.wave_power[:, :, :, 0] = self.wave_power[:, :, :, -1]

    def calculate_omega_derivatives(self):
        """
        Calculate the derivatives of omega with respect to r, theta, k, and thetak.
        """
        wave_omega = calculate_wave_omega(self.k, self.theta_k, self.vA, self.theta_vA, self.v_sw, self.theta, self.c_s, wave_type=self.config['wave_type'])
        wave_omega.sum().backward()
        self.domega_dr = self.r.grad
        self.domega_dtheta = self.theta.grad
        self.domega_dk = self.k.grad
        self.domega_dtheta_k = self.theta_k.grad

    def calculate_wave_power_derivatives(self):
        """
        Calculate the derivatives of the wave power with respect to r, theta, k, and thetak.
        """
        # r boundary condition is zero gradient
        # self.dW_dr = (
        #     self.wave_power.diff(dim=0, prepend=self.wave_power[1:2])
        #     + self.wave_power.diff(dim=0, append=self.wave_power[-2:-1])
        # ) / (2 * self.dr)
        self.dW_dr = (
            self.wave_power.diff(dim=0, prepend=self.wave_power[0:1])
            + self.wave_power.diff(dim=0, append=self.wave_power[-1:])
        ) / (2 * self.dr)

        # theta boundary condition is periodic
        self.dW_dtheta = (
            self.wave_power.diff(dim=1, prepend=self.wave_power[:, -1:])
            + self.wave_power.diff(dim=1, append=self.wave_power[:, 0:1])
        ) / (2 * self.dtheta)

        # k boundary condition is zero gradient
        # self.dW_dk = (
        #     self.wave_power.diff(dim=2, prepend=self.wave_power[:, :, 1:2])
        #     + self.wave_power.diff(dim=2, append=self.wave_power[:, :, -2:-1])
        # ) / (
        #     self.k.diff(dim=2, prepend=self.k[:, :, 0:1]**2 / self.k[:, :, 1:2])
        #     + self.k.diff(dim=2, append=self.k[:, :, -1:]**2 / self.k[:, :, -2:-1])
        # )
        # self.dW_dk = (
        #     self.wave_power.diff(dim=2, prepend=self.wave_power[:, :, 1:2])
        #     + self.wave_power.diff(dim=2, append=self.wave_power[:, :, -2:-1])
        # ) / (2 * self.dk)
        self.dW_dk = (
            self.wave_power.diff(dim=2, prepend=self.wave_power[:, :, 0:1])
            + self.wave_power.diff(dim=2, append=self.wave_power[:, :, -1:])
        ) / (2 * self.dk)

        # theta_k boundary condition is periodic
        self.dW_dtheta_k = (
            self.wave_power.diff(dim=3, prepend=self.wave_power[:, :, :, -1:])
            + self.wave_power.diff(dim=3, append=self.wave_power[:, :, :, 0:1])
        ) / (2 * self.dtheta_k)

    def calculate_wave_power_time_variation(self):
        """
        Calculate the time variation according to the wave kinetic equation.
        """

        term1a = (torch.cos(self.theta_k) * self.domega_dk - torch.sin(self.theta_k) / self.k * self.domega_dtheta_k) * \
                (torch.cos(self.theta) * self.dW_dr - torch.sin(self.theta)/self.r * self.dW_dtheta)
        term1b = (torch.sin(self.theta_k) * self.domega_dk + torch.cos(self.theta_k) / self.k * self.domega_dtheta_k) * \
                (torch.sin(self.theta) * self.dW_dr + torch.cos(self.theta)/self.r * self.dW_dtheta)
        term2a = (torch.cos(self.theta) * self.domega_dr - torch.sin(self.theta)/self.r * self.domega_dtheta) * \
                (torch.cos(self.theta_k) * self.dW_dk - torch.sin(self.theta_k)/self.k * self.dW_dtheta_k)
        term2b = (torch.sin(self.theta) * self.domega_dr + torch.cos(self.theta)/self.r * self.domega_dtheta) * \
                (torch.sin(self.theta_k) * self.dW_dk + torch.cos(self.theta_k)/self.k * self.dW_dtheta_k)
        if self.mode == 'linear':  # Linear Equation
            self.dW_dt = -(term1a + term1b) + (term2a + term2b) + 2 * self.config['gamma_k'] * self.wave_power
        else:  # Log Equation
            self.dW_dt = -(term1a + term1b) + (term2a + term2b) + 2 * self.config['gamma_k']
    
    def iterate(self, order=1):
        """
        Iterate the wave power using the wave kinetic equation.
        """
        dt = self.config['dt']

        # if self.mode == 'linear':
        #     if torch.any(self.wave_power < 0):
        #         raise ValueError("Wave power is iterated to be negative for somewhere!")
        # else:
        #     print(torch.any(self.wave_power.isnan()))
        if order == 1:
            self.wave_power = self.wave_power + self.dW_dt * dt
            self.set_boundary
            
            self.calculate_wave_power_derivatives()
            self.calculate_wave_power_time_variation()
            self.t += dt
        elif order == 3:
            tmp_wave_power_0 = self.wave_power.clone()
            self.wave_power = self.wave_power + self.dW_dt * dt
            self.calculate_wave_power_derivatives()
            self.calculate_wave_power_time_variation()
            self.wave_power = 0.25 * self.wave_power + 0.75 * tmp_wave_power_0 + 0.25 * self.dW_dt * dt
            self.calculate_wave_power_derivatives()
            self.calculate_wave_power_time_variation()
            self.wave_power = 2 / 3 * self.wave_power + 1 / 3 * tmp_wave_power_0 + 2 / 3 * self.dW_dt * dt
            # self.set_boundary()

            self.calculate_wave_power_derivatives()
            self.calculate_wave_power_time_variation()
            self.t += dt
        else:
            raise ValueError("The time stepping order should only be 1 or 3 for now!")