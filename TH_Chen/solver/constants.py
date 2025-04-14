
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

GRAVITATIONAL_CONSTANT = 6.67430e-11 # Gravitational constant in m^3 kg^-1 s^-2
MU0 = 4 * torch.pi * 1e-7 # Permeability of free space in T m/A
K_BOLTZMANN = 1.380649e-23 # Boltzmann constant in J/K
PROTON_MASS = 1.6726219e-27 # Proton mass in kg
ROTATION_ANGULAR_VELOCITY = 2.6e-6 # Rotation angular velocity in rad/s
SOLAR_RADIUS = 6.9634e8 # Solar mass in kg
SOLAR_MASS = 1.989e30 # Solar mass in kg
SURFACE_DENSITY = 1e15 # Surface density in m^-3
SURFACE_MAGNETIC_FIELD = 1e-2 # Surface magnetic field in T
