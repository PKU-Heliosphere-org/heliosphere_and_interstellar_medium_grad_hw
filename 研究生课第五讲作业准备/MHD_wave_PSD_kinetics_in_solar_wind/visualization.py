#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for plotting solar wind properties and wave kinetic simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

class Visualizer:
    """
    Class for creating visualizations of solar wind properties and
    wave kinetic simulation results.
    """
    
    def __init__(self, config, solar_wind):
        """
        Initialize visualizer.
        
        Args:
            config (dict): Configuration parameters
            solar_wind (SolarWindModel): Solar wind model instance
        """
        self.config = config
        self.solar_wind = solar_wind
        
        # Set up plotting style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_solar_wind_properties(self):
        """
        Create plots of solar wind properties (velocity, density, magnetic field).
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Plot solar wind velocity
        ax1 = fig.add_subplot(221)
        r_range = np.linspace(self.solar_wind.R_s, 20*self.solar_wind.R_s, 100)
        v_sw = self.solar_wind.get_solar_wind_velocity(r_range)
        ax1.plot(r_range/self.solar_wind.R_s, v_sw/1e3)
        ax1.set_xlabel('r [Rs]')
        ax1.set_ylabel('Solar Wind Velocity [km/s]')
        ax1.grid(True)
        
        # Plot number density
        ax2 = fig.add_subplot(222)
        n = self.solar_wind.get_number_density(r_range, v_sw)
        ax2.plot(r_range/self.solar_wind.R_s, n)
        ax2.set_xlabel('r [Rs]')
        ax2.set_ylabel('Number Density [m^-3]')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        # Plot magnetic field components
        ax3 = fig.add_subplot(223)
        B_r, B_theta = self.solar_wind.get_magnetic_field(r_range, 0, v_sw)
        ax3.plot(r_range/self.solar_wind.R_s, B_r*1e4, label='Br')
        ax3.plot(r_range/self.solar_wind.R_s, B_theta*1e4, label='Bθ')
        ax3.set_xlabel('r [Rs]')
        ax3.set_ylabel('Magnetic Field [G]')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Plot Alfvén velocity
        ax4 = fig.add_subplot(224)
        v_A_r, v_A_theta = self.solar_wind.get_alfven_velocity(B_r, B_theta, n)
        ax4.plot(r_range/self.solar_wind.R_s, v_A_r/1e3, label='vA_r')
        ax4.plot(r_range/self.solar_wind.R_s, v_A_theta/1e3, label='vA_θ')
        ax4.set_xlabel('r [Rs]')
        ax4.set_ylabel('Alfvén Velocity [km/s]')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('solar_wind_properties.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_wave_power_spectrum(self, W):
        """
        Create plots of wave power spectrum in different projections.
        
        Args:
            W (array): Wave power spectrum W(r, theta, kr, ktheta)
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot r-theta projection (integrated over kr, ktheta)
        ax1 = fig.add_subplot(221, projection='polar')
        W_r_theta = np.mean(np.mean(W, axis=3), axis=2)
        im1 = ax1.pcolormesh(self.solar_wind.theta, self.solar_wind.r/self.solar_wind.R_s,
                            W_r_theta, norm=LogNorm())
        plt.colorbar(im1, ax=ax1, label='Wave Power (r-θ)')
        ax1.set_title('Wave Power in r-θ Space')
        
        # Plot kr-ktheta projection (integrated over r, theta)
        ax2 = fig.add_subplot(222, projection='polar')
        W_kr_ktheta = np.mean(np.mean(W, axis=1), axis=0)
        im2 = ax2.pcolormesh(self.solar_wind.ktheta, self.solar_wind.kr,
                            W_kr_ktheta, norm=LogNorm())
        plt.colorbar(im2, ax=ax2, label='Wave Power (kr-kθ)')
        ax2.set_title('Wave Power in kr-kθ Space')
        
        # Plot r-kr projection (integrated over theta, ktheta)
        ax3 = fig.add_subplot(223)
        W_r_kr = np.mean(np.mean(W, axis=3), axis=1)
        im3 = ax3.pcolormesh(self.solar_wind.r/self.solar_wind.R_s,
                            self.solar_wind.kr, W_r_kr.T, norm=LogNorm())
        plt.colorbar(im3, ax=ax3, label='Wave Power (r-kr)')
        ax3.set_xlabel('r [Rs]')
        ax3.set_ylabel('kr [m^-1]')
        ax3.set_yscale('log')
        
        # Plot radial profiles at different kr
        ax4 = fig.add_subplot(224)
        kr_indices = [0, len(self.solar_wind.kr)//2, -1]
        for idx in kr_indices:
            W_profile = np.mean(np.mean(W[:, :, idx, :], axis=2), axis=1)
            ax4.plot(self.solar_wind.r/self.solar_wind.R_s, W_profile,
                    label=f'kr = {self.solar_wind.kr[idx]:.2e} m^-1')
        ax4.set_xlabel('r [Rs]')
        ax4.set_ylabel('Wave Power')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('wave_power_spectrum.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_history(self, convergence_history):
        """
        Plot convergence history of the simulation.
        
        Args:
            convergence_history (list): List of error values during iteration
        """
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_history)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.yscale('log')
        plt.title('Convergence History')
        plt.grid(True)
        plt.savefig('convergence_history.png', dpi=300, bbox_inches='tight')
        plt.close()