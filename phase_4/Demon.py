import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la
from types import Dict, Any
from dataclasses import dataclass


# Constants
kB = 1.0  # Boltzmann constant
T_H = 1.6
T_C = 1.0
DELTAE = 1.0
GAMMA = 1.0



@dataclass
class PhysParams:
    """Class to hold physical parameters for the simulation.
    Attributes:
        Th (float): Temperature of the hot reservoir.
        Tc (float): Temperature of the cold reservoir.
        DeltaE (float): Energy difference between demon states.
        gamma_hot (float): Transition rate with the hot reservoir.
    """
    Th: float
    Tc: float
    DeltaE: float
    gamma: float




class Demon:
    """Class representing the Demon entity.
    Attributes:
        n (int): Number of states the demon can occupy.
    """
    def __init__(self, n : int, phys_params: PhysParams= None, init_state: str = 'd0'):
        if phys_params is None:
            self.phys_params = PhysParams(Th=T_H, Tc=T_C, DeltaE=DELTAE, gamma=GAMMA)
        else:
            self.phys_params = phys_params
        self.n = n
        self.current_state = init_state
        self.states = [f'd{i}' for i in range(n)]
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.intrinsic_rates = self._compute_intrinsic_rates()
        self.outgoing_rates = self._compute_outgoing_rates()
    def _compute_intrinsic_rates(self) -> Dict[str, float]:
        """Compute intrinsic transition rates for the demon."""
        rates = {}
        for i in range(self.n - 1):
            rates[f'd{i}->d{i+1}'] = self.phys_params.gamma * np.exp(-self.phys_params.DeltaE/self.n / (2 * self.phys_params.Th))
            rates[f'd{i+1}->d{i}'] = self.phys_params.gamma * np.exp(self.phys_params.DeltaE/self.n / (2 * self.phys_params.Th))
        return rates
    def _compute_outgoing_rates(self) -> Dict[str, float]:
        """Compute outgoing transition rates for the demon, interacting with bits & and the cold reservoir."""
        rates = {}
        self.omega = np.tanh(self.phys_params.DeltaE/self.n / (2 * self.phys_params.Tc))
        for i in range(self.n-1):
            rates[f'1_d{i+1}->0_d{i}'] =  1+self.omega
            rates[f'0_d{i}->1_d{i+1}'] = 1-self.omega
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
    