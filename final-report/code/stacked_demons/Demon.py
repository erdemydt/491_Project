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
    
    Can be initialized in multiple ways:
    1. Using temperatures: PhysParams(Th=..., Tc=..., DeltaE=..., gamma=...)
    2. Using rates: PhysParams(sigma=..., omega=..., DeltaE=..., gamma=...)
    
    New features for stacked demons:
    - delta_e_mode: 'per_state' or 'total' - how DeltaE is interpreted
    - preserve_mode: 'sigma_omega' or 'temperatures' - what to preserve when using total DeltaE
    
    Attributes:
        DeltaE (float): Energy difference (interpretation depends on delta_e_mode)
        gamma (float): Transition rate with the hot reservoir
        delta_e_mode (str): 'per_state' - DeltaE is between consecutive states
                           'total' - DeltaE is total from ground to top
        preserve_mode (str): 'sigma_omega' - preserve sigma/omega when recalculating
                            'temperatures' - preserve Th/Tc when recalculating
        demon_n (int): Number of demon states (required for 'total' mode)
        Th (float): Temperature of hot reservoir
        Tc (float): Temperature of cold reservoir
        sigma (float): Intrinsic transition parameter tanh(DeltaE_per_state/(2*Th))
        omega (float): Outgoing transition parameter tanh(DeltaE_per_state/(2*Tc))
    """
    DeltaE: float
    gamma: float
    delta_e_mode: str = 'per_state'  # 'per_state' or 'total'
    preserve_mode: str = 'sigma_omega'  # 'sigma_omega' or 'temperatures'
    demon_n: Optional[int] = None
    Th: Optional[float] = None
    Tc: Optional[float] = None
    sigma: Optional[float] = None
    omega: Optional[float] = None
    
    def __post_init__(self):
        """Compute derived parameters based on what was provided."""
        
        # Validate delta_e_mode
        if self.delta_e_mode not in ['per_state', 'total']:
            raise ValueError("delta_e_mode must be 'per_state' or 'total'")
        
        # Validate preserve_mode
        if self.preserve_mode not in ['sigma_omega', 'temperatures']:
            raise ValueError("preserve_mode must be 'sigma_omega' or 'temperatures'")
        
        # If total mode, demon_n is required
        if self.delta_e_mode == 'total' and self.demon_n is None:
            raise ValueError("demon_n must be provided when delta_e_mode is 'total'")
        
        # Calculate DeltaE per state if in total mode
        if self.delta_e_mode == 'total':
            if self.demon_n < 2:
                raise ValueError("demon_n must be at least 2")
            delta_e_per_state = self.DeltaE / (self.demon_n - 1)
        else:
            delta_e_per_state = self.DeltaE
        
        # Now compute sigma/omega and temperatures based on what was provided
        # Case 1: Both Th and Tc provided
        if self.Th is not None and self.Tc is not None:
            if self.sigma is not None or self.omega is not None:
                raise ValueError("Provide either (Th, Tc) or (sigma, omega), not both")
            self.sigma = np.tanh(delta_e_per_state / (2 * self.Th))
            self.omega = np.tanh(delta_e_per_state / (2 * self.Tc))
        
        # Case 2: Both sigma and omega provided
        elif self.sigma is not None and self.omega is not None:
            if self.Th is not None or self.Tc is not None:
                raise ValueError("Provide either (Th, Tc) or (sigma, omega), not both")
            # Compute temperatures from sigma and omega using per-state energy
            self.Th = delta_e_per_state / (2 * np.arctanh(self.sigma))
            self.Tc = delta_e_per_state / (2 * np.arctanh(self.omega))
        
        # Case 3: Invalid - must provide one complete set
        else:
            raise ValueError("Must provide either (Th, Tc) or (sigma, omega)")
        
        # Store the per-state energy for reference
        self._delta_e_per_state = delta_e_per_state
        
        # Validate that all values are now set
        if self.Th <= 0 or self.Tc <= 0:
            raise ValueError("Temperatures must be positive")
        if not (-1 < self.sigma < 1):
            raise ValueError(f"sigma must be in (-1, 1), got {self.sigma}")
        if not (-1 < self.omega < 1):
            raise ValueError(f"omega must be in (-1, 1), got {self.omega}")
    def get_delta_e_per_state(self) -> float:
        """Get the energy difference per state transition."""
        return self._delta_e_per_state
    
    def recalculate_for_new_demon_n(self, new_demon_n: int) -> 'PhysParams':
        """Create a new PhysParams object for a different number of demon states.
        
        This is used when we have K stacked demons and need to adjust parameters.
        
        Args:
            new_demon_n (int): New number of demon states
            
        Returns:
            PhysParams: New parameters adjusted for the new demon count
        """
        if self.delta_e_mode == 'per_state':
            # DeltaE is per state, so it stays the same
            return PhysParams(
                DeltaE=self.DeltaE,
                gamma=self.gamma,
                delta_e_mode='per_state',
                preserve_mode=self.preserve_mode,
                demon_n=new_demon_n,
                sigma=self.sigma,
                omega=self.omega
            )
        else:  # delta_e_mode == 'total'
            # DeltaE is total, need to recalculate based on preserve_mode
            new_delta_e_per_state = self.DeltaE / (new_demon_n - 1)
            
            if self.preserve_mode == 'temperatures':
                # Preserve Th and Tc, recalculate sigma and omega
                return PhysParams(
                    DeltaE=self.DeltaE,
                    gamma=self.gamma,
                    delta_e_mode='total',
                    preserve_mode='temperatures',
                    demon_n=new_demon_n,
                    Th=self.Th,
                    Tc=self.Tc
                )
            else:  # preserve_mode == 'sigma_omega'
                # Preserve sigma and omega, recalculate Th and Tc
                return PhysParams(
                    DeltaE=self.DeltaE,
                    gamma=self.gamma,
                    delta_e_mode='total',
                    preserve_mode='sigma_omega',
                    demon_n=new_demon_n,
                    sigma=self.sigma,
                    omega=self.omega
                )


class Demon:
    """Class representing a single Demon entity.
    
    Attributes:
        n (int): Number of states the demon can occupy.
        delta_e_values (np.ndarray): Pre-calculated energy differences between consecutive states.
        energy_values (np.ndarray): Pre-calculated total energy for each state.
    """
    def __init__(self, n: int, phys_params: PhysParams = None, init_state: str = 'd0', 
                 energy_distribution: str = "uniform"):
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
        delta_e_per_state = self.phys_params.get_delta_e_per_state()
        
        if distribution_type == "uniform":
            # Equal energy steps between all states
            return np.full(self.n - 1, delta_e_per_state)
            
        elif distribution_type == "exponential":
            # Exponentially increasing energy gaps (smaller gaps at lower states)
            exp_values = np.array([np.exp(-i**2 / self.n) for i in range(self.n - 1)])
            total_delta_e = delta_e_per_state * (self.n - 1)
            normalization = total_delta_e / np.sum(exp_values)
            return exp_values * normalization
            
        elif distribution_type == "quadratic":
            # Quadratically increasing energy gaps
            quad_values = np.array([(i + 1)**2 for i in range(self.n - 1)])
            total_delta_e = delta_e_per_state * (self.n - 1)
            normalization = total_delta_e / np.sum(quad_values)
            return quad_values * normalization
            
        else:
            raise ValueError(f"Unknown distribution_type '{distribution_type}'. "
                           f"Choose from: 'uniform', 'exponential', 'quadratic'")
    
    def get_delta_e_for_state_n(self, state: int) -> float:
        """Get the energy difference for a specific state transition.
        
        Args:
            state (int): The demon state number (0 to n-2, representing transitions from state to state+1)
            
        Returns:
            float: Energy difference for the state transition
        """
        if state < 0 or state >= self.n:
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
            rates[f'{bit_state}_d{demon_state}->{bit_state}_d{int(demon_state)-1}'] = \
                self.intrinsic_rates[f'd{int(demon_state)}->d{int(demon_state)-1}']
        if demon_state != str(self.n - 1):
            rates[f'{bit_state}_d{demon_state}->{bit_state}_d{int(demon_state)+1}'] = \
                self.intrinsic_rates[f'd{int(demon_state)}->d{int(demon_state)+1}']
        
        # Outgoing transitions involving bit flips
        if bit_state == '0' and demon_state != str(self.n - 1):
            rates[f'0_d{demon_state}->1_d{int(demon_state)+1}'] = \
                self.outgoing_rates[f'0_d{demon_state}->1_d{int(demon_state)+1}']
        elif bit_state == '1' and demon_state != '0':
            rates[f'1_d{demon_state}->0_d{int(demon_state)-1}'] = \
                self.outgoing_rates[f'1_d{int(demon_state)}->0_d{int(demon_state)-1}']
        
        return rates
