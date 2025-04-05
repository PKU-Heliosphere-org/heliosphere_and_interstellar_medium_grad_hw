#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for solving the Wave Kinetic Equation in solar wind.
This program simulates wave propagation in the solar wind using
the wave kinetic equation in polar coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

from solar_wind_model import SolarWindModel
from wave_dispersion import WaveDispersionRelation
from wave_kinetic_solver import WaveKineticSolver
from visualization import Visualizer

def main():
    """
    Main function to execute the wave kinetic equation simulation in solar wind.
    """
    # Print welcome message
    print("=" * 80)
    print("Solar Wind Wave Kinetics Simulation")
    print("=" * 80)
    
    # Configuration parameters
    config = {
        # Spatial grid parameters
        'r_min': 1.0,           # Minimum radius in solar radii
        'r_max': 20.0,          # Maximum radius in solar radii
        'Nr': 100,              # Number of grid points in radial direction
        'Ntheta': 36,           # Number of grid points in theta direction
        
        # Wave vector grid parameters
        'kr_min': 1e-8,         # Minimum kr in m^-1
        'kr_max': 1e-5,         # Maximum kr in m^-1
        'Nkr': 50,              # Number of grid points in kr direction
        'Nktheta': 36,          # Number of grid points in ktheta direction
        
        # Time stepping parameters
        'dt': 0.1,              # Time step in seconds
        'max_iter': 1000,       # Maximum number of iterations
        'convergence_threshold': 1e-6,  # Convergence threshold
        
        # Physical parameters
        'wave_mode': 'Alfven',  # Wave mode: 'Alfven', 'Fast', or 'Slow'
        'sigma_ktheta': 0.5,    # Standard deviation for ktheta distribution
        'A': 1.0,               # Amplitude factor for initial wave power spectrum
        
        # Solar wind parameters
        'T': 1.5e6,             # Temperature in K
        'n0': 1e15,             # Base number density in m^-3
        'B0': 0.01,             # Base magnetic field in T
        'omega_sun': 2.6e-6     # Solar rotation angular velocity in rad/s
    }
    
    # Initialize solar wind model
    print("Initializing solar wind model...")
    solar_wind = SolarWindModel(config)
    
    # Initialize wave dispersion relation
    print("Setting up wave dispersion relation...")
    dispersion = WaveDispersionRelation(config, solar_wind)
    
    # Initialize wave kinetic solver
    print("Setting up wave kinetic solver...")
    solver = WaveKineticSolver(config, solar_wind, dispersion)
    
    # Initialize visualizer
    visualizer = Visualizer(config, solar_wind)
    
    # Solve the wave kinetic equation
    print("Solving wave kinetic equation...")
    start_time = time.time()
    W, convergence_history = solver.solve()
    end_time = time.time()
    print(f"Solution completed in {end_time - start_time:.2f} seconds")
    
    # Visualize results
    print("Generating visualizations...")
    visualizer.plot_solar_wind_properties()
    visualizer.plot_wave_power_spectrum(W)
    visualizer.plot_convergence_history(convergence_history)
    
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()