
from .constants import *
from scipy.optimize import fsolve

import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)

def solar_wind_velocity(r, temperature, v_guess=None):
    """
    Calculate the Parker solar wind velocity.

    Args:
        r: float or np.array, NOT torch.tensor

    Returns:
        v_sw: float or np.array, Parker solar wind velocity in m/s
        r_crit: float, critical radius in m
        v_crit: float, critical velocity in m/s
    """
    
    if not isinstance(r, np.ndarray):
        r = np.array(r)

    # Critical radius and velocity
    r_crit = GRAVITATIONAL_CONSTANT * SOLAR_MASS * PROTON_MASS / (4 * K_BOLTZMANN * temperature)
    v_crit = np.sqrt(2 * K_BOLTZMANN * temperature / PROTON_MASS)
    # logger.info(f"Critical radius: {r_crit:.2e} m, Critical velocity: {v_crit:.2e} m/s")

    def parker_equation(v, r):
        return (v / v_crit)**2 - 2 * np.log(v / v_crit) \
            - 4 * np.log(r / r_crit) - 4 * (r_crit / r) + 3
    
    # Initial guess for the velocity
    if v_guess is None:
        v_guess = (r - .9 * SOLAR_RADIUS) / (r_crit - .9 * SOLAR_RADIUS) * v_crit

    # Solve the Parker equation for velocity
    v_sw = fsolve(parker_equation, v_guess, args=(r,))
    
    if v_sw.shape == (1,):
        v_sw = v_sw[0]

    return v_sw, r_crit, v_crit

def calculate_solar_wind_density(r, v_sw, v_sw0):
    """
    Calculate solar wind density.
    """
    return SURFACE_DENSITY * (SOLAR_RADIUS / r)**2 * (v_sw0 / v_sw)

def calculate_magnetic_field(r, theta, v_sw):
    B_r = SURFACE_MAGNETIC_FIELD * (SOLAR_RADIUS / r)**2
    B_theta = -ROTATION_ANGULAR_VELOCITY * r * B_r / v_sw

    theta_B = (torch.arctan2(B_theta, B_r) + theta) % (2 * np.pi)
    B = torch.sqrt(B_r**2 + B_theta**2)
    return B, theta_B

def calculate_alfven_speed(B, theta_B, n_sw):
    """
    Calculate Alfven speed.
    """
    vA = B / torch.sqrt(MU0 * n_sw * PROTON_MASS)
    return vA, theta_B

def calculate_sound_speed(temperature):
    """
    Calculate sound speed.
    """
    return np.sqrt(K_BOLTZMANN * temperature / PROTON_MASS)