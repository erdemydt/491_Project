import numpy as np
import scipy.linalg as la

def get_sigma_omega_from_T_H_C(T_H, T_C, k_B=1.0, deltaE=1.0):
    """
    Calculate sigma and omega from hot and cold reservoir temperatures.
    
    Parameters:
    T_H (float): Temperature of the hot reservoir.
    T_C (float): Temperature of the cold reservoir.
    k_B (float): Boltzmann constant. Default is 1.0.
    
    Returns:
    tuple: (sigma, omega)
    """
    theta_H = 1 / (k_B * T_H)
    theta_C = 1 / (k_B * T_C)
    sigma = np.tanh(theta_H*deltaE/2) 
    omega = np.tanh(theta_C*deltaE/2)
    return sigma, omega

def get_T_H_C_from_sigma_omega(sigma, omega, k_B=1.0, deltaE=1.0):
    """
    Calculate hot and cold reservoir temperatures from sigma and omega.
    
    Parameters:
    sigma (float): Parameter related to the hot reservoir.
    omega (float): Parameter related to the cold reservoir.
    k_B (float): Boltzmann constant. Default is 1.0.
    
    Returns:
    tuple: (T_H, T_C)
    """
    theta_H = (2 / deltaE) * np.arctanh(sigma)
    theta_C = (2 / deltaE) * np.arctanh(omega)
    T_H = 1 / (k_B * theta_H)
    T_C = 1 / (k_B * theta_C)
    return T_H, T_C

def get_epsilon_from_sigma_omega(sigma, omega):
    """
    Calculate epsilon from sigma and omega.
    
    Parameters:
    sigma (float): Parameter related to the hot reservoir.
    omega (float): Parameter related to the cold reservoir.
    
    Returns:
    float: epsilon
    """
    epsilon = (omega - sigma) / (1 - sigma * omega)
    return epsilon
def get_sigma_from_epsilon_omega(epsilon, omega):
    """
    Calculate sigma from epsilon and omega.
    
    Parameters:
    epsilon (float): Parameter related to the system.
    omega (float): Parameter related to the cold reservoir.
    
    Returns:
    float: sigma
    """
    sigma = (omega - epsilon) / (1 - epsilon * omega)
    return sigma