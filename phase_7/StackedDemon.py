import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# Constants
kB = 1.0  # Boltzmann constant

@dataclass
class StackedPhysParams:
    """Physical parameters for the stacked demon simulation.
    
    The demon has 3 states: low (d0), medium (d1), high (d2)
    Energy gaps: d0->d1 has energy DeltaE_1, d1->d2 has energy DeltaE_2
    
    Attributes:
        DeltaE_1 (float): Energy difference between low and medium states
        DeltaE_2 (float): Energy difference between medium and high states
        gamma (float): Transition rate with the hot reservoir
        Th (float): Temperature of the hot reservoir
        Tc (float): Temperature of the cold reservoir
        sigma_1 (float): tanh(DeltaE_1/(2*Th)) - computed from Th
        sigma_2 (float): tanh(DeltaE_2/(2*Th)) - computed from Th
        omega_1 (float): tanh(DeltaE_1/(2*Tc)) - computed from Tc
        omega_2 (float): tanh(DeltaE_2/(2*Tc)) - computed from Tc
    """
    DeltaE_1: float
    DeltaE_2: float
    gamma: float
    Th: float
    Tc: float
    sigma_1: Optional[float] = None
    sigma_2: Optional[float] = None
    omega_1: Optional[float] = None
    omega_2: Optional[float] = None
    
    def __post_init__(self):
        """Compute derived parameters."""
        if self.Th <= 0 or self.Tc <= 0:
            raise ValueError("Temperatures must be positive")
        
        # Compute sigma values (hot reservoir parameters)
        self.sigma_1 = np.tanh(self.DeltaE_1 / (2 * self.Th))
        self.sigma_2 = np.tanh(self.DeltaE_2 / (2 * self.Th))
        
        # Compute omega values (cold reservoir parameters)
        self.omega_1 = np.tanh(self.DeltaE_1 / (2 * self.Tc))
        self.omega_2 = np.tanh(self.DeltaE_2 / (2 * self.Tc))


class StackedDemon:
    """Demon that interacts with 2 bits at a time.
    
    States: d0 (low), d1 (medium), d2 (high)
    Bit pairs: 00, 01, 10, 11 (representing 0, 1, 2, 3 in decimal)
    
    Transition rules (with cold reservoir):
    - 00_d0 <-> 01_d1 or 10_d1 (1 bit flip, +DeltaE_1)
    - 00_d0 <-> 11_d2 (2 bit flips, +DeltaE_1 + DeltaE_2)
    - 01_d1 or 10_d1 <-> 11_d2 (1 bit flip, +DeltaE_2)
    - Reverse transitions go down in demon energy
    """
    
    def __init__(self, phys_params: StackedPhysParams, init_state: str = 'd0'):
        self.phys_params = phys_params
        self.states = ['d0', 'd1', 'd2']  # low, medium, high
        self.current_state = init_state
        
        # Bit pair states (2 bits)
        self.bit_states = ['00', '01', '10', '11']
        
        # Energy levels
        self.energy_levels = {
            'd0': 0.0,
            'd1': phys_params.DeltaE_1,
            'd2': phys_params.DeltaE_1 + phys_params.DeltaE_2
        }
        
        # Precompute all transition rates
        self.intrinsic_rates = self._compute_intrinsic_rates()
        self.outgoing_rates = self._compute_outgoing_rates()
        
        print(f"Stacked Demon initialized:")
        print(f"  States: {self.states}")
        print(f"  Energy d0->d1: {phys_params.DeltaE_1}")
        print(f"  Energy d1->d2: {phys_params.DeltaE_2}")
        print(f"  Total energy range: {self.energy_levels['d2']}")
    
    def _compute_intrinsic_rates(self) -> Dict[str, float]:
        """Compute intrinsic transition rates (hot reservoir)."""
        rates = {}
        
        # d0 <-> d1 (energy gap DeltaE_1)
        rates['d0->d1'] = self.phys_params.gamma * np.exp(-self.phys_params.DeltaE_1 / (2 * self.phys_params.Th))
        rates['d1->d0'] = self.phys_params.gamma * np.exp(self.phys_params.DeltaE_1 / (2 * self.phys_params.Th))
        
        # d1 <-> d2 (energy gap DeltaE_2)
        rates['d1->d2'] = self.phys_params.gamma * np.exp(-self.phys_params.DeltaE_2 / (2 * self.phys_params.Th))
        rates['d2->d1'] = self.phys_params.gamma * np.exp(self.phys_params.DeltaE_2 / (2 * self.phys_params.Th))
        
        return rates
    
    def _compute_outgoing_rates(self) -> Dict[str, float]:
        """Compute outgoing transition rates (cold reservoir + bit flips).
        
        Transitions follow the pattern:
        - Flipping 1 bit from 0->1 and going up 1 demon level: use omega for that level
        - Flipping 1 bit from 1->0 and going down 1 demon level: use omega for that level
        - Flipping 2 bits involves both energy gaps
        """
        rates = {}
        
        # 00_d0 -> 01_d1 or 10_d1 (one bit 0->1, demon up by DeltaE_1)
        omega_1 = self.phys_params.omega_1
        rates['00_d0->01_d1'] = 1 - omega_1
        rates['00_d0->10_d1'] = 1 - omega_1
        
        # 01_d1 -> 00_d0 or 10_d1 -> 00_d0 (one bit 1->0, demon down by DeltaE_1)
        rates['01_d1->00_d0'] = 1 + omega_1
        rates['10_d1->00_d0'] = 1 + omega_1
        
        # 01_d1 -> 11_d2 or 10_d1 -> 11_d2 (one bit 0->1, demon up by DeltaE_2)
        omega_2 = self.phys_params.omega_2
        rates['01_d1->11_d2'] = 1 - omega_2
        rates['10_d1->11_d2'] = 1 - omega_2
        
        # 11_d2 -> 01_d1 or 11_d2 -> 10_d1 (one bit 1->0, demon down by DeltaE_2)
        rates['11_d2->01_d1'] = 1 + omega_2
        rates['11_d2->10_d1'] = 1 + omega_2
        
        # 00_d0 -> 11_d2 (two bits 0->1, demon up by DeltaE_1 + DeltaE_2)
        # This is a composite transition - rate depends on both energy gaps
        
        rates['00_d0->11_d2'] = (1 - omega_1) * (1 - omega_2)
        
        # 11_d2 -> 00_d0 (two bits 1->0, demon down by DeltaE_1 + DeltaE_2)
        rates['11_d2->00_d0'] = (1 + omega_1) * (1 + omega_2)
        
        return rates
    
    def get_rates_for_joint_state(self, joint_state: str) -> Dict[str, float]:
        """Get all transition rates from a given joint state (bits_demonstate).
        
        Args:
            joint_state (str): State like '00_d0', '01_d1', etc.
            
        Returns:
            Dict[str, float]: Dictionary of transitions and their rates
        """
        bits, demon = joint_state.split('_')
        rates = {}
        
        # Intrinsic transitions (hot reservoir, no bit change)
        if demon == 'd0':
            rates[f'{bits}_d0->{bits}_d1'] = self.intrinsic_rates['d0->d1']
        elif demon == 'd1':
            rates[f'{bits}_d1->{bits}_d0'] = self.intrinsic_rates['d1->d0']
            rates[f'{bits}_d1->{bits}_d2'] = self.intrinsic_rates['d1->d2']
        elif demon == 'd2':
            rates[f'{bits}_d2->{bits}_d1'] = self.intrinsic_rates['d2->d1']
        
        # Outgoing transitions (cold reservoir, with bit flips)
        transition_key = f'{bits}_{demon}'
        
        # Check all possible outgoing transitions from this state
        for key, rate in self.outgoing_rates.items():
            if key.startswith(transition_key + '->'):
                rates[key] = rate
        
        return rates
    
    def get_all_joint_states(self):
        """Get all possible joint states (bit_pair_demon_state)."""
        return [f'{bits}_{demon}' for bits in self.bit_states for demon in self.states]
