#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Solar wind model module implementing Parker's solution and magnetic field calculations.
"""

import numpy as np
from scipy.optimize import fsolve

class SolarWindModel:
    """
    Class for calculating solar wind properties and magnetic field distribution
    in a 2D polar coordinate system.
    """
    
    # Physical constants
    G = 6.67430e-11    # Gravitational constant in m^3 kg^-1 s^-2
    M_s = 1.989e30     # Solar mass in kg
    R_s = 6.9634e8     # Solar radius in m
    k_B = 1.380649e-23 # Boltzmann constant in J K^-1
    m_p = 1.6726219e-27 # Proton mass in kg
    
    def __init__(self, config):
        """
        Initialize solar wind model with configuration parameters.
        
        Args:
            config (dict): Configuration parameters including temperature,
                         base density, base magnetic field, etc.
        """
        self.T = config['T']
        self.n0 = config['n0']
        self.B0 = config['B0']
        self.omega_sun = config['omega_sun']
        
        # Calculate critical values
        self.v_c = np.sqrt(2 * self.k_B * self.T / self.m_p)
        self.r_c = self.G * self.M_s * self.m_p / (4 * self.k_B * self.T)
        
        # Setup grid
        self.setup_grid(config)
    
    def setup_grid(self, config):
        """
        Set up spatial and wave vector grids.
        
        Args:
            config (dict): Configuration parameters including grid specifications
        """
        # Spatial grid
        self.r = np.linspace(config['r_min'], config['r_max'], config['Nr']) * self.R_s
        self.theta = np.linspace(0, 2*np.pi, config['Ntheta'])
        self.R, self.THETA = np.meshgrid(self.r, self.theta)
        
    def parker_equation(self, v, r):
        """
        Parker solar wind equation.
        
        Args:
            v (float): Solar wind velocity
            r (float): Radial distance
            
        Returns:
            float: Value of Parker equation at given v and r
        """
        return (v/self.v_c)**2 - 2*np.log(v/self.v_c) - 4*np.log(r/self.r_c) \
               - 4*(self.r_c/r) + 3
    
    def get_solar_wind_velocity(self, r):
        """
        Calculate solar wind velocity at given radius.
        
        Args:
            r (float or array): Radial distance(s)
            
        Returns:
            float or array: Solar wind velocity
        """
        def solve_parker(r_val):
            # Initial guess depends on whether r is below or above critical radius
            v_init = 0.5 * self.v_c if r_val < self.r_c else 1.5 * self.v_c
            return fsolve(self.parker_equation, v_init, args=(r_val,))[0]
        
        if isinstance(r, np.ndarray):
            return np.array([solve_parker(r_val) for r_val in r])
        return solve_parker(r)
    
    def get_number_density(self, r, v):
        """
        Calculate number density at given radius and velocity.
        
        Args:
            r (float or array): Radial distance(s)
            v (float or array): Solar wind velocity(ies)
            
        Returns:
            float or array: Number density
        """
        v_surface = self.get_solar_wind_velocity(self.R_s)
        return self.n0 * (self.R_s/r)**2 * v_surface/v
    
    def get_magnetic_field(self, r, theta, v):
        """
        Calculate magnetic field components at given position and velocity.
        
        Args:
            r (float or array): Radial distance(s)
            theta (float or array): Angular position(s)
            v (float or array): Solar wind velocity(ies)
            
        Returns:
            tuple: (B_r, B_theta) magnetic field components
        """
        # Radial component
        B_r = self.B0 * (self.R_s/r)**2
        
        # Azimuthal component
        B_theta = -self.omega_sun * r * B_r / v
        
        return B_r, B_theta
    
    def get_alfven_velocity(self, B_r, B_theta, n):
        """
        Calculate Alfvén velocity components.
        
        Args:
            B_r (float or array): Radial magnetic field
            B_theta (float or array): Azimuthal magnetic field
            n (float or array): Number density
            
        Returns:
            tuple: (v_A_r, v_A_theta) Alfvén velocity components
        """
        mu0 = 4e-7 * np.pi  # Vacuum permeability
        denominator = np.sqrt(mu0 * n * self.m_p)
        
        v_A_r = B_r / denominator
        v_A_theta = B_theta / denominator
        
        return v_A_r, v_A_theta
    
    def get_sound_speed(self):
        """
        Calculate isothermal sound speed.
        
        Returns:
            float: Sound speed
        """
        return np.sqrt(self.k_B * self.T / self.m_p)