import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# Constants
kB = 1.0  # Boltzmann constant


@dataclass
class PhysParams:
    """Physical parameters for the demon simulation.
    
    Attributes:
        DeltaE (float): Energy difference between consecutive demon states
        gamma (float): Transition rate with the hot reservoir
        Th (float): Temperature of the hot reservoir
        Tc (float): Temperature of the cold reservoir
        sigma (float): tanh(DeltaE/(2*Th)) - computed from Th
        omega (float): tanh(DeltaE/(2*Tc)) - computed from Tc
    """
    DeltaE: float
    gamma: float
    Th: float
    Tc: float
    sigma: Optional[float] = None
    omega: Optional[float] = None
    
    def __post_init__(self):
        """Compute derived parameters."""
        if self.Th <= 0 or self.Tc <= 0:
            raise ValueError("Temperatures must be positive")
        
        # Compute sigma and omega
        self.sigma = np.tanh(self.DeltaE / (2 * self.Th))
        self.omega = np.tanh(self.DeltaE / (2 * self.Tc))


class CompetingDemon:
    """Demon that can be in one of n states.
    
    This demon competes with K-1 other identical demons based on which one
    has the shortest time-to-next-event when interacting with a bit.
    
    Attributes:
        n (int): Number of states the demon can occupy (d0, d1, ..., d_{n-1})
        phys_params (PhysParams): Physical parameters
        current_state (str): Current demon state
        demon_id (int): Unique identifier for this demon (0 to K-1)
    """
    
    def __init__(self, n: int, phys_params: PhysParams, init_state: str = 'd0', demon_id: int = 0):
        """Initialize a competing demon.
        
        Args:
            n (int): Number of demon states
            phys_params (PhysParams): Physical parameters
            init_state (str): Initial demon state (default 'd0')
            demon_id (int): Unique identifier for this demon
        """
        self.n = n
        self.phys_params = phys_params
        self.current_state = init_state
        self.demon_id = demon_id
        
        # Generate state names
        self.states = [f'd{i}' for i in range(n)]
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        
        # Pre-calculate energy differences and total energies
        self.delta_e_values = np.full(n - 1, phys_params.DeltaE)
        self.energy_values = np.cumsum(np.concatenate([[0], self.delta_e_values]))
        
        # Precompute transition rates
        self.intrinsic_rates = self._compute_intrinsic_rates()
        self.outgoing_rates = self._compute_outgoing_rates()
    
    def _compute_intrinsic_rates(self) -> Dict[str, float]:
        """Compute intrinsic transition rates (hot reservoir)."""
        rates = {}
        delta_e = self.phys_params.DeltaE
        
        for i in range(self.n - 1):
            rates[f'd{i}->d{i+1}'] = self.phys_params.gamma * np.exp(-delta_e / (2 * self.phys_params.Th))
            rates[f'd{i+1}->d{i}'] = self.phys_params.gamma * np.exp(delta_e / (2 * self.phys_params.Th))
        
        return rates
    
    def _compute_outgoing_rates(self) -> Dict[str, float]:
        """Compute outgoing transition rates (cold reservoir + bit flips)."""
        rates = {}
        omega = self.phys_params.omega
        
        for i in range(self.n - 1):
            # Bit flip 0->1, demon goes up
            rates[f'0_d{i}->1_d{i+1}'] = 1 - omega
            # Bit flip 1->0, demon goes down
            rates[f'1_d{i+1}->0_d{i}'] = 1 + omega
        
        return rates
    
    def get_rates_for_joint_state(self, joint_state: str) -> Dict[str, float]:
        """Get all transition rates from a given joint state.
        
        Args:
            joint_state (str): State like '0_d0', '1_d2', etc.
            
        Returns:
            Dict[str, float]: Dictionary of transitions and their rates
        """
        bit, demon_state = joint_state.split('_')
        demon_idx = self.state_indices[demon_state]
        rates = {}
        
        # Intrinsic transitions (hot reservoir, no bit change)
        if demon_idx > 0:
            key = f'd{demon_idx}->d{demon_idx-1}'
            rates[f'{bit}_{demon_state}->{bit}_d{demon_idx-1}'] = self.intrinsic_rates[key]
        
        if demon_idx < self.n - 1:
            key = f'd{demon_idx}->d{demon_idx+1}'
            rates[f'{bit}_{demon_state}->{bit}_d{demon_idx+1}'] = self.intrinsic_rates[key]
        
        # Outgoing transitions (cold reservoir, with bit flip)
        if bit == '0' and demon_idx < self.n - 1:
            key = f'0_d{demon_idx}->1_d{demon_idx+1}'
            rates[key] = self.outgoing_rates[key]
        elif bit == '1' and demon_idx > 0:
            key = f'1_d{demon_idx}->0_d{demon_idx-1}'
            rates[key] = self.outgoing_rates[key]
        
        return rates
    
    def get_energy_of_state(self, state: str) -> float:
        """Get the total energy of a specific demon state.
        
        Args:
            state (str): The demon state (e.g., 'd0', 'd1')
            
        Returns:
            float: Total energy of the state
        """
        state_idx = self.state_indices[state]
        return self.energy_values[state_idx]
