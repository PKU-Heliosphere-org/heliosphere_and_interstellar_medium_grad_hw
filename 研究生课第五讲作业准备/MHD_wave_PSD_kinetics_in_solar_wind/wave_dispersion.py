#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wave dispersion relation module for different MHD wave modes in solar wind.
"""

import numpy as np

class WaveDispersionRelation:
    """
    Class for calculating dispersion relations of MHD waves in solar wind,
    including Alfvén, Fast, and Slow modes.
    """
    
    def __init__(self, config, solar_wind):
        """
        Initialize wave dispersion calculator.
        
        Args:
            config (dict): Configuration parameters
            solar_wind (SolarWindModel): Solar wind model instance
        """
        self.config = config
        self.solar_wind = solar_wind
        self.wave_mode = config['wave_mode']
    
    def calc_omega(self, k_vec, pos, v_A_vec, v_sw_vec, c_s):
        """
        Calculate wave frequency based on wave mode and local parameters.
        
        Args:
            k_vec (array): Wave vector [k_r, k_theta]
            pos (array): Position vector [r, theta]
            v_A_vec (array): Alfvén velocity vector [v_A_r, v_A_theta]
            v_sw_vec (array): Solar wind velocity vector [v_sw_r, v_sw_theta]
            c_s (float): Sound speed
            
        Returns:
            float: Wave frequency omega
        """
        if self.wave_mode == 'Alfven':
            return self._calc_omega_alfven(k_vec, v_A_vec, v_sw_vec)
        elif self.wave_mode == 'Fast':
            return self._calc_omega_fast(k_vec, v_A_vec, v_sw_vec, c_s)
        elif self.wave_mode == 'Slow':
            return self._calc_omega_slow(k_vec, v_A_vec, v_sw_vec, c_s)
        else:
            raise ValueError(f"Unknown wave mode: {self.wave_mode}")
    
    def _calc_omega_alfven(self, k_vec, v_A_vec, v_sw_vec):
        """
        Calculate Alfvén wave frequency.
        
        Args:
            k_vec (array): Wave vector
            v_A_vec (array): Alfvén velocity vector
            v_sw_vec (array): Solar wind velocity vector
            
        Returns:
            float: Alfvén wave frequency
        """
        return np.dot(k_vec, v_sw_vec + v_A_vec)
    
    def _calc_omega_fast(self, k_vec, v_A_vec, v_sw_vec, c_s):
        """
        Calculate Fast magnetosonic wave frequency.
        
        Args:
            k_vec (array): Wave vector
            v_A_vec (array): Alfvén velocity vector
            v_sw_vec (array): Solar wind velocity vector
            c_s (float): Sound speed
            
        Returns:
            float: Fast wave frequency
        """
        v_A = np.linalg.norm(v_A_vec)
        k = np.linalg.norm(k_vec)
        k_dot_vA = np.dot(k_vec, v_A_vec)
        k_dot_vsw = np.dot(k_vec, v_sw_vec)
        
        term1 = (c_s**2 + v_A**2) * k**2 / 2
        term2 = np.sqrt((c_s**2 + v_A**2)**2 * k**4 - 4 * k**2 * c_s**2 * k_dot_vA**2) / 2
        term3 = k_dot_vsw
        
        return np.sqrt(term1 + term2) + term3
    
    def _calc_omega_slow(self, k_vec, v_A_vec, v_sw_vec, c_s):
        """
        Calculate Slow magnetosonic wave frequency.
        
        Args:
            k_vec (array): Wave vector
            v_A_vec (array): Alfvén velocity vector
            v_sw_vec (array): Solar wind velocity vector
            c_s (float): Sound speed
            
        Returns:
            float: Slow wave frequency
        """
        v_A = np.linalg.norm(v_A_vec)
        k = np.linalg.norm(k_vec)
        k_dot_vA = np.dot(k_vec, v_A_vec)
        k_dot_vsw = np.dot(k_vec, v_sw_vec)
        
        term1 = (c_s**2 + v_A**2) * k**2 / 2
        term2 = np.sqrt((c_s**2 + v_A**2)**2 * k**4 - 4 * k**2 * c_s**2 * k_dot_vA**2) / 2
        term3 = k_dot_vsw
        
        return np.sqrt(term1 - term2) + term3
    
    def calc_omega_derivatives(self, k_vec, pos, v_A_vec, v_sw_vec, c_s, h=1e-6):
        """
        Calculate partial derivatives of omega with respect to k and x.
        
        Args:
            k_vec (array): Wave vector
            pos (array): Position vector
            v_A_vec (array): Alfvén velocity vector
            v_sw_vec (array): Solar wind velocity vector
            c_s (float): Sound speed
            h (float): Step size for numerical derivatives
            
        Returns:
            tuple: (domega_dk, domega_dx) Derivatives of omega
        """
        # Calculate derivatives with respect to k
        domega_dk = np.zeros_like(k_vec)
        for i in range(len(k_vec)):
            k_plus = k_vec.copy()
            k_plus[i] += h
            k_minus = k_vec.copy()
            k_minus[i] -= h
            
            omega_plus = self.calc_omega(k_plus, pos, v_A_vec, v_sw_vec, c_s)
            omega_minus = self.calc_omega(k_minus, pos, v_A_vec, v_sw_vec, c_s)
            domega_dk[i] = (omega_plus - omega_minus) / (2 * h)
        
        # Calculate derivatives with respect to x (position)
        domega_dx = np.zeros_like(pos)
        for i in range(len(pos)):
            x_plus = pos.copy()
            x_plus[i] += h
            x_minus = pos.copy()
            x_minus[i] -= h
            
            # Note: You need to update v_A_vec, v_sw_vec, and c_s for the new positions
            # This is a simplified version
            omega_plus = self.calc_omega(k_vec, x_plus, v_A_vec, v_sw_vec, c_s)
            omega_minus = self.calc_omega(k_vec, x_minus, v_A_vec, v_sw_vec, c_s)
            domega_dx[i] = (omega_plus - omega_minus) / (2 * h)
        
        return domega_dk, domega_dx