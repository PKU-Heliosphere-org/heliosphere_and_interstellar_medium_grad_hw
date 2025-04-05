#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wave kinetic equation solver module implementing numerical methods for
solving the wave kinetic equation in polar coordinates.
"""

import numpy as np
from tqdm import tqdm

class WaveKineticSolver:
    """
    Class for solving the wave kinetic equation in polar coordinates
    using finite difference methods.
    """
    
    def __init__(self, config, solar_wind, dispersion):
        """
        Initialize wave kinetic solver.
        
        Args:
            config (dict): Configuration parameters
            solar_wind (SolarWindModel): Solar wind model instance
            dispersion (WaveDispersionRelation): Wave dispersion relation calculator
        """
        self.config = config
        self.solar_wind = solar_wind
        self.dispersion = dispersion
        
        # Set up grids
        self.setup_grids()
        
        # Time stepping parameters
        self.dt = config['dt']
        self.max_iter = config['max_iter']
        self.convergence_threshold = config['convergence_threshold']
        
        # Physical parameters
        self.wave_mode = config['wave_mode']
        self.sigma_ktheta = config['sigma_ktheta']
        self.A = config['A']
        
        # Wave growth/damping rate (gamma_k)
        # This is a placeholder - in a real model, this would be calculated based on physical processes
        self.gamma_k = np.zeros((self.Nr, self.Ntheta, self.Nkr, self.Nktheta))
    
    def setup_grids(self):
        """
        Set up spatial and wave vector grids.
        """
        # Spatial grid
        self.r_min = self.config['r_min'] * self.solar_wind.R_s
        self.r_max = self.config['r_max'] * self.solar_wind.R_s
        self.Nr = self.config['Nr']
        self.Ntheta = self.config['Ntheta']
        
        self.r = np.linspace(self.r_min, self.r_max, self.Nr)
        self.theta = np.linspace(0, 2*np.pi, self.Ntheta)
        self.dr = (self.r_max - self.r_min) / (self.Nr - 1)
        self.dtheta = 2*np.pi / (self.Ntheta - 1)
        
        # Wave vector grid
        self.kr_min = self.config['kr_min']
        self.kr_max = self.config['kr_max']
        self.Nkr = self.config['Nkr']
        self.Nktheta = self.config['Nktheta']
        
        self.kr = np.logspace(np.log10(self.kr_min), np.log10(self.kr_max), self.Nkr)
        self.ktheta = np.linspace(0, 2*np.pi, self.Nktheta)
        self.dkr = np.diff(self.kr)
        self.dkr = np.append(self.dkr, self.dkr[-1])  # Extend for the last point
        self.dktheta = 2*np.pi / (self.Nktheta - 1)
        
        # Create meshgrids for vectorized operations
        self.R, self.THETA = np.meshgrid(self.r, self.theta, indexing='ij')
        self.KR, self.KTHETA = np.meshgrid(self.kr, self.ktheta, indexing='ij')
    
    def initialize_wave_power(self):
        """
        Initialize wave power spectrum based on the given formula.
        
        Returns:
            array: Initial wave power spectrum W(r, theta, kr, ktheta)
        """
        # Initialize 4D array for wave power
        W = np.zeros((self.Nr, self.Ntheta, self.Nkr, self.Nktheta))
        
        # Calculate initial wave power spectrum
        for i in range(self.Nr):
            for j in range(self.Ntheta):
                for m in range(self.Nkr):
                    for n in range(self.Nktheta):
                        r_val = self.r[i]
                        kr_val = self.kr[m]
                        ktheta_val = self.ktheta[n]
                        
                        # Apply initial condition formula
                        W[i, j, m, n] = self.A * (r_val / self.solar_wind.R_s)**(-3) * kr_val**(-5/3) * \
                                       (1 / (np.sqrt(2*np.pi) * self.sigma_ktheta)) * \
                                       np.exp(-ktheta_val**2 / (2 * self.sigma_ktheta**2))
        
        return W
    
    def apply_boundary_conditions(self, W):
        """
        Apply boundary conditions to the wave power spectrum.
        
        Args:
            W (array): Wave power spectrum
            
        Returns:
            array: Wave power spectrum with boundary conditions applied
        """
        # Radial boundaries (extrapolation)
        W[0, :, :, :] = W[1, :, :, :]
        W[-1, :, :, :] = W[-2, :, :, :]
        
        # Theta boundaries (periodic)
        W[:, 0, :, :] = W[:, -2, :, :]
        W[:, -1, :, :] = W[:, 1, :, :]
        
        # kr boundaries (extrapolation)
        W[:, :, 0, :] = W[:, :, 1, :]
        W[:, :, -1, :] = W[:, :, -2, :]
        
        # ktheta boundaries (periodic)
        W[:, :, :, 0] = W[:, :, :, -2]
        W[:, :, :, -1] = W[:, :, :, 1]
        
        return W
    
    def calculate_derivatives(self, W):
        """
        Calculate spatial and wave vector derivatives using central differences.
        
        Args:
            W (array): Wave power spectrum
            
        Returns:
            tuple: (dW_dr, dW_dtheta, dW_dkr, dW_dktheta) Derivatives of W
        """
        # Spatial derivatives
        dW_dr = np.zeros_like(W)
        dW_dtheta = np.zeros_like(W)
        
        # Wave vector derivatives
        dW_dkr = np.zeros_like(W)
        dW_dktheta = np.zeros_like(W)
        
        # Calculate derivatives using central differences
        for i in range(1, self.Nr - 1):
            dW_dr[i, :, :, :] = (W[i + 1, :, :, :] - W[i - 1, :, :, :]) / (2 * self.dr)
        
        for j in range(1, self.Ntheta - 1):
            dW_dtheta[:, j, :, :] = (W[:, j + 1, :, :] - W[:, j - 1, :, :]) / (2 * self.dtheta)
        
        for m in range(1, self.Nkr - 1):
            dW_dkr[:, :, m, :] = (W[:, :, m + 1, :] - W[:, :, m - 1, :]) / (2 * self.dkr[m])
        
        for n in range(1, self.Nktheta - 1):
            dW_dktheta[:, :, :, n] = (W[:, :, :, n + 1] - W[:, :, :, n - 1]) / (2 * self.dktheta)
        
        return dW_dr, dW_dtheta, dW_dkr, dW_dktheta
    
    def calculate_rhs(self, W, dW_dr, dW_dtheta, dW_dkr, dW_dktheta):
        """
        Calculate the right-hand side of the wave kinetic equation.
        
        This implementation follows the exact mathematical formulation of the wave kinetic equation
        in polar coordinates as described in equations 1.1.6-1.1.8:
        
        [∂ω/∂k · ∇_x - ∂ω/∂x · ∇_k]W = 2γ_k W
        
        Args:
            W (array): Wave power spectrum
            dW_dr, dW_dtheta, dW_dkr, dW_dktheta (array): Derivatives of W
            
        Returns:
            array: Right-hand side of the wave kinetic equation
        """
        # Initialize right-hand side array
        rhs = np.zeros_like(W)
        
        # For each grid point, calculate the right-hand side
        for i in range(1, self.Nr - 1):
            for j in range(1, self.Ntheta - 1):
                for m in range(1, self.Nkr - 1):
                    for n in range(1, self.Nktheta - 1):
                        r_val = self.r[i]
                        theta_val = self.theta[j]
                        kr_val = self.kr[m]
                        ktheta_val = self.ktheta[n]
                        
                        # Get local solar wind properties
                        v_sw = self.solar_wind.get_solar_wind_velocity(r_val)
                        n = self.solar_wind.get_number_density(r_val, v_sw)
                        B_r, B_theta = self.solar_wind.get_magnetic_field(r_val, theta_val, v_sw)
                        v_A_r, v_A_theta = self.solar_wind.get_alfven_velocity(B_r, B_theta, n)
                        c_s = self.solar_wind.get_sound_speed()
                        
                        # Convert to Cartesian for dispersion calculation
                        k_vec = np.array([kr_val * np.cos(ktheta_val), kr_val * np.sin(ktheta_val)])
                        pos = np.array([r_val * np.cos(theta_val), r_val * np.sin(theta_val)])
                        v_A_vec = np.array([v_A_r * np.cos(theta_val) - v_A_theta * np.sin(theta_val),
                                           v_A_r * np.sin(theta_val) + v_A_theta * np.cos(theta_val)])
                        v_sw_vec = np.array([v_sw * np.cos(theta_val), v_sw * np.sin(theta_val)])
                        
                        # Calculate omega derivatives
                        domega_dk, domega_dx = self.dispersion.calc_omega_derivatives(
                            k_vec, pos, v_A_vec, v_sw_vec, c_s
                        )
                        
                        # Convert derivatives back to polar coordinates
                        domega_dkr = domega_dk[0] * np.cos(ktheta_val) + domega_dk[1] * np.sin(ktheta_val)
                        domega_dktheta = -domega_dk[0] * kr_val * np.sin(ktheta_val) + domega_dk[1] * kr_val * np.cos(ktheta_val)
                        
                        domega_dr = domega_dx[0] * np.cos(theta_val) + domega_dx[1] * np.sin(theta_val)
                        domega_dtheta = -domega_dx[0] * r_val * np.sin(theta_val) + domega_dx[1] * r_val * np.cos(theta_val)
                        
                        # Implementation of equation 1.1.6:
                        # ∂ω/∂k · ∇_x term
                        # First part: (cos k_θ ∂ω/∂k_r - sin k_θ/k_r ∂ω/∂k_θ)(cos θ ∂/∂r - sin θ/r ∂/∂θ)W
                        term1a = (np.cos(ktheta_val) * domega_dkr - np.sin(ktheta_val)/kr_val * domega_dktheta) * \
                                (np.cos(theta_val) * dW_dr[i, j, m, n] - np.sin(theta_val)/r_val * dW_dtheta[i, j, m, n])
                        
                        # Second part: (sin k_θ ∂ω/∂k_r + cos k_θ/k_r ∂ω/∂k_θ)(sin θ ∂/∂r + cos θ/r ∂/∂θ)W
                        term1b = (np.sin(ktheta_val) * domega_dkr + np.cos(ktheta_val)/kr_val * domega_dktheta) * \
                                (np.sin(theta_val) * dW_dr[i, j, m, n] + np.cos(theta_val)/r_val * dW_dtheta[i, j, m, n])
                        
                        # Implementation of equation 1.1.7:
                        # ∂ω/∂x · ∇_k term
                        # First part: (cos θ ∂ω/∂r - sin θ/r ∂ω/∂θ)(cos k_θ ∂/∂k_r - sin k_θ/k_r ∂/∂k_θ)W
                        term2a = (np.cos(theta_val) * domega_dr - np.sin(theta_val)/r_val * domega_dtheta) * \
                                (np.cos(ktheta_val) * dW_dkr[i, j, m, n] - np.sin(ktheta_val)/kr_val * dW_dktheta[i, j, m, n])
                        
                        # Second part: (sin θ ∂ω/∂r + cos θ/r ∂ω/∂θ)(sin k_θ ∂/∂k_r + cos k_θ/k_r ∂/∂k_θ)W
                        term2b = (np.sin(theta_val) * domega_dr + np.cos(theta_val)/r_val * domega_dtheta) * \
                                (np.sin(ktheta_val) * dW_dkr[i, j, m, n] + np.cos(ktheta_val)/kr_val * dW_dktheta[i, j, m, n])
                        
                        # Combine terms according to equation 1.1.8:
                        # [∂ω/∂k · ∇_x - ∂ω/∂x · ∇_k]W = 2γ_k W
                        rhs[i, j, m, n] = (term1a + term1b) - (term2a + term2b) + 2 * self.gamma_k[i, j, m, n] * W[i, j, m, n]
        
        return rhs
    
    def solve(self):
        """
        Solve the wave kinetic equation using explicit Euler time stepping.
        
        Returns:
            tuple: (W, convergence_history) Final wave power spectrum and convergence history
        """
        # Initialize wave power spectrum
        W = self.initialize_wave_power()
        W = self.apply_boundary_conditions(W)
        
        # Initialize convergence history
        convergence_history = []
        
        # Time stepping loop
        for step in tqdm(range(self.max_iter)):
            # Calculate derivatives
            dW_dr, dW_dtheta, dW_dkr, dW_dktheta = self.calculate_derivatives(W)
            
            # Calculate right-hand side
            rhs = self.calculate_rhs(W, dW_dr, dW_dtheta, dW_dkr, dW_dktheta)
            
            # Update wave power spectrum
            W_new = W + self.dt * rhs
            
            # Apply boundary conditions
            W_new = self.apply_boundary_conditions(W_new)
            
            # Calculate error
            error = np.max(np.abs(W_new - W))
            convergence_history.append(error)
            
            # Check for convergence
            if error < self.convergence_threshold:
                print(f"Converged after {step+1} iterations")
                break
            
            # Update W for next iteration
            W = W_new
        
        return W, convergence_history