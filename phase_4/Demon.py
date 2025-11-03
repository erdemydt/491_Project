import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# Constants
kB = 1.0  # Boltzmann constant
T_H = 1.6
T_C = 1.0
DELTAE = 1.0
GAMMA = 1.0



@dataclass
class PhysParams:
    """Class to hold physical parameters for the simulation.
    
    Can be initialized in two ways:
    1. Using temperatures: PhysParams(Th=..., Tc=..., DeltaE=..., gamma=...)
    2. Using rates: PhysParams(sigma=..., omega=..., DeltaE=..., gamma=...)
    
    Attributes:
        Th (float): Temperature of the hot reservoir (computed from sigma if not provided).
        Tc (float): Temperature of the cold reservoir (computed from omega if not provided).
        DeltaE (float): Energy difference between demon states.
        gamma (float): Transition rate with the hot reservoir.
        sigma (float): Intrinsic transition parameter tanh(DeltaE/(2*Th)) (computed from Th if not provided).
        omega (float): Outgoing transition parameter tanh(DeltaE/(2*Tc)) (computed from Tc if not provided).
    """
    DeltaE: float
    gamma: float
    Th: Optional[float] = None
    Tc: Optional[float] = None
    sigma: Optional[float] = None
    omega: Optional[float] = None
    
    def __post_init__(self):
        """Compute derived parameters based on what was provided."""
        # Case 1: Both Th and Tc provided
        if self.Th is not None and self.Tc is not None:
            if self.sigma is not None or self.omega is not None:
                raise ValueError("Provide either (Th, Tc) or (sigma, omega), not both")
            self.sigma = np.tanh(self.DeltaE / (2 * self.Th))
            self.omega = np.tanh(self.DeltaE / (2 * self.Tc))
        
        # Case 2: Both sigma and omega provided
        elif self.sigma is not None and self.omega is not None:
            if self.Th is not None or self.Tc is not None:
                raise ValueError("Provide either (Th, Tc) or (sigma, omega), not both")
            # Compute temperatures from sigma and omega
            self.Th = self.DeltaE / (2 * np.arctanh(self.sigma))
            self.Tc = self.DeltaE / (2 * np.arctanh(self.omega))
        
        # Case 3: Invalid - must provide one complete set
        else:
            raise ValueError("Must provide either (Th, Tc) or (sigma, omega)")
        
        # Validate that all values are now set
        if self.Th <= 0 or self.Tc <= 0:
            raise ValueError("Temperatures must be positive")
        if not (-1 < self.sigma < 1):
            raise ValueError(f"sigma must be in (-1, 1), got {self.sigma}")
        if not (-1 < self.omega < 1):
            raise ValueError(f"omega must be in (-1, 1), got {self.omega}")




class Demon:
    """Class representing the Demon entity.
    Attributes:
        n (int): Number of states the demon can occupy.
        delta_e_values (np.ndarray): Pre-calculated energy differences between consecutive states.
        energy_values (np.ndarray): Pre-calculated total energy for each state.
    """
    def __init__(self, n: int, phys_params: PhysParams = None, init_state: str = 'd0', energy_distribution: str = "uniform"):
        if phys_params is None:
            self.phys_params = PhysParams(sigma=.3, omega=.8, DeltaE=DELTAE, gamma=GAMMA)
        else:
            self.phys_params = phys_params
        self.n = n
        self.current_state = init_state
        self.states = [f'd{i}' for i in range(n)]
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        
        # Pre-calculate all energy differences and total energies
        self.delta_e_values = self._calculate_delta_e_distribution(energy_distribution)
        self.energy_values = np.cumsum(np.concatenate([[0], self.delta_e_values]))
        self.intrinsic_rates = self._compute_intrinsic_rates()
        self.outgoing_rates = self._compute_outgoing_rates()
    
    def _calculate_delta_e_distribution(self, distribution_type: str = "uniform") -> np.ndarray:
        """Calculate energy differences between consecutive states based on distribution type.
        
        Args:
            distribution_type (str): Type of energy distribution ("uniform", "exponential", "quadratic")
            
        Returns:
            np.ndarray: Array of energy differences between consecutive states
        """
        if distribution_type == "uniform":
            # Equal energy steps between all states
            return np.full(self.n - 1, self.phys_params.DeltaE / (self.n - 1))
            
        elif distribution_type == "exponential":
            # Exponentially increasing energy gaps (smaller gaps at lower states)
            exp_values = np.array([np.exp(-i**2 / self.n) for i in range(self.n - 1)])
            normalization = self.phys_params.DeltaE / np.sum(exp_values)
            return exp_values * normalization
            
        elif distribution_type == "quadratic":
            # Quadratically increasing energy gaps
            quad_values = np.array([(i + 1)**2 for i in range(self.n - 1)])
            normalization = self.phys_params.DeltaE / np.sum(quad_values)
            return quad_values * normalization
            
        else:
            raise ValueError(f"Unknown distribution_type '{distribution_type}'. Choose from: 'uniform', 'exponential', 'quadratic'")
    
    def get_delta_e_for_state_n(self, state: int) -> float:
        """Get the energy difference for a specific state transition.
        
        Args:
            state (int): The demon state number (0 to n-2, representing transitions from state to state+1)
            
        Returns:
            float: Energy difference for the state transition
        """
        if state < 0 or state >= self.n :
            raise ValueError(f"State must be between 0 and {self.n-1} for transitions")

        return self.delta_e_values[state]
    
    
    
    def get_total_delta_e(self) -> float:
        """Get the total energy difference across all states.
        
        Returns:
            float: Total energy difference from ground state to highest state
        """
        return np.sum(self.delta_e_values)
    
    def get_energy_of_state(self, state: int) -> float:
        """Get the total energy of a specific demon state.
        
        Args:
            state (int): The demon state number (0 to n-1)
            
        Returns:
            float: Total energy of the state (relative to ground state d0)
        """
        if state < 0 or state >= self.n:
            raise ValueError(f"State must be between 0 and {self.n-1}")
            
        return self.energy_values[state]
    
    def _compute_intrinsic_rates(self) -> Dict[str, float]:
        """Compute intrinsic transition rates for the demon."""
        rates = {}
        for i in range(self.n - 1):
            delta_e = self.delta_e_values[i]
            rates[f'd{i}->d{i+1}'] = self.phys_params.gamma * np.exp(-delta_e / (2 * self.phys_params.Th))
            rates[f'd{i+1}->d{i}'] = self.phys_params.gamma * np.exp(delta_e / (2 * self.phys_params.Th))
        return rates
    def _compute_outgoing_rates(self) -> Dict[str, float]:
        """Compute outgoing transition rates for the demon, interacting with bits and the cold reservoir."""
        rates = {}
        for i in range(self.n - 1):
            delta_e = self.delta_e_values[i]
            omega = np.tanh(delta_e / (2 * self.phys_params.Tc))
            rates[f'1_d{i+1}->0_d{i}'] = 1 + omega
            rates[f'0_d{i}->1_d{i+1}'] = 1 - omega
        return rates
    def get_rates_for_full_state(self, demon_bit_state: str) -> Dict[str, float]:
        """Get transition rates for the full state including the bit state."""
        bit_state, demon_state = demon_bit_state.split('_d')
        rates = {}
        # Intrinsic demon transitions
        if demon_state != '0':
            rates[f'{bit_state}_d{demon_state}->{bit_state}_d{int(demon_state)-1}'] = self.intrinsic_rates[f'd{int(demon_state)}->d{int(demon_state)-1}']
        if demon_state != str(self.n - 1):
            rates[f'{bit_state}_d{demon_state}->{bit_state}_d{int(demon_state)+1}'] = self.intrinsic_rates[f'd{int(demon_state)}->d{int(demon_state)+1}']
        
        # Outgoing transitions involving bit flips
        if bit_state == '0' and demon_state != str(self.n - 1):
            rates[f'0_d{demon_state}->1_d{int(demon_state)+1}'] = self.outgoing_rates[f'0_d{demon_state}->1_d{int(demon_state)+1}']
        elif bit_state == '1' and demon_state != '0':
            rates[f'1_d{demon_state}->0_d{int(demon_state)-1}'] = self.outgoing_rates[f'1_d{int(demon_state)}->0_d{int(demon_state)-1}']
        return rates
    