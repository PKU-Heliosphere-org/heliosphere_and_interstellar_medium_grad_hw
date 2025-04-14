
from .constants import *

import logging
import numpy as np
logger = logging.getLogger(__name__)

def calculate_wave_omega(k, theta_k, vA, theta_vA, v_sw, theta, c_s, wave_type):
    """
    Calculate wave frequency using dispersion relations.
    TODO: add more wave types
    """

    assert wave_type in ['alfven'], f"Unsupported wave type: {wave_type}"

    if wave_type == 'alfven':
        omega = k * (vA * torch.cos(theta_k - theta_vA) + v_sw * torch.cos(theta_k - theta))

    return omega