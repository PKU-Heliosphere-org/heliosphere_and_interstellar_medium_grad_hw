# Solar Wind Wave Kinetics Simulation

This project implements a numerical solver for the Wave Kinetic Equation in solar wind plasma. It simulates the propagation and evolution of MHD waves (Alfvén, Fast, and Slow modes) in the solar wind.

## Overview

The Wave Kinetic Equation describes the evolution of wave energy density in phase space (position and wave vector). This implementation focuses on solving the equation in 2D polar coordinates, which is suitable for studying solar wind waves in the ecliptic plane.

## Features

- Parker solar wind model implementation
- MHD wave dispersion relations for Alfvén, Fast, and Slow modes
- Wave kinetic equation solver using finite difference methods
- Visualization tools for solar wind properties and wave power spectrum

## Project Structure

```
solar_wind_wave_kinetics/
├── main.py                  # Main script to run the simulation
├── solar_wind_model.py      # Parker solar wind model implementation
├── wave_dispersion.py       # MHD wave dispersion relations
├── wave_kinetic_solver.py   # Wave kinetic equation solver
├── visualization.py         # Plotting and visualization tools
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

## Mathematical Background

The Wave Kinetic Equation in polar coordinates is:

$$\left\{\frac{\partial \omega}{\partial \mathbf{k}} \cdot \nabla_{\mathbf{x}} - \frac{\partial \omega}{\partial \mathbf{x}} \cdot \nabla_{\mathbf{k}}\right\} W = 2 \gamma_k W$$

where:
- $W$ is the wave energy density
- $\omega$ is the wave frequency
- $\mathbf{k}$ is the wave vector
- $\mathbf{x}$ is the position vector
- $\gamma_k$ is the wave growth/damping rate

## Installation

1. Clone the repository
2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

Run the simulation:
```
python main.py
```

The simulation will:
1. Initialize the solar wind model
2. Set up the wave dispersion relation
3. Solve the wave kinetic equation
4. Generate visualizations of the results

## Configuration

The simulation parameters can be modified in the `config` dictionary in `main.py`:

- Spatial grid parameters (`r_min`, `r_max`, `Nr`, `Ntheta`)
- Wave vector grid parameters (`kr_min`, `kr_max`, `Nkr`, `Nktheta`)
- Time stepping parameters (`dt`, `max_iter`, `convergence_threshold`)
- Physical parameters (`wave_mode`, `sigma_ktheta`, `A`)
- Solar wind parameters (`T`, `n0`, `B0`, `omega_sun`)

## Output

The simulation generates several visualization files:
- `solar_wind_properties.png`: Plots of solar wind velocity, density, magnetic field, and Alfvén velocity
- `wave_power_spectrum.png`: Plots of wave power spectrum in different projections
- `convergence_history.png`: Plot of convergence history during the simulation

## References

This implementation is based on the theoretical framework described in:
- Tu, C.-Y., & Marsch, E. (1995). MHD structures, waves and turbulence in the solar wind: Observations and theories. Space Science Reviews, 73(1-2), 1-210.
- Velli, M., Grappin, R., & Mangeney, A. (1989). Waves and turbulence in the corona. Physical Review Letters, 63(17), 1807-1810.