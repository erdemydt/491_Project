"""
Phase 10: Two-Bit Demon Implementation

A demon that interacts with TWO bits at a time, with cooperative transitions
that can flip bits based on their joint state.

States: Up (u), Down (d)
Default Cooperative Transitions:
    - d00 <-> u01  (demon goes up, flips second bit)
    - u11 <-> d10  (demon goes down, flips second bit)
"""

import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Set

# Constants
kB = 1.0  # Boltzmann constant
T_H = 1.6
T_C = 1.0
DELTAE = 1.0
GAMMA = 1.0


@dataclass
class PhysParams:
    """Physical parameters for the two-bit demon simulation.
    
    Attributes:
        DeltaE (float): Energy difference between demon states (u and d)
        gamma (float): Intrinsic transition rate
        Th (float): Hot reservoir temperature (for intrinsic transitions)
        Tc (float): Cold reservoir temperature (for cooperative transitions)
        sigma (float): Intrinsic transition parameter tanh(DeltaE/(2*Th))
        omega (float): Cooperative transition parameter tanh(DeltaE/(2*Tc))
    """
    DeltaE: float = DELTAE
    gamma: float = GAMMA
    Th: Optional[float] = None
    Tc: Optional[float] = None
    sigma: Optional[float] = None
    omega: Optional[float] = None
    
    def __post_init__(self):
        """Compute derived parameters."""
        # Case 1: Temperatures provided
        if self.Th is not None and self.Tc is not None:
            if self.sigma is not None or self.omega is not None:
                raise ValueError("Provide either (Th, Tc) or (sigma, omega), not both")
            self.sigma = np.tanh(self.DeltaE / (2 * self.Th))
            self.omega = np.tanh(self.DeltaE / (2 * self.Tc))
        
        # Case 2: sigma/omega provided
        elif self.sigma is not None and self.omega is not None:
            if self.Th is not None or self.Tc is not None:
                raise ValueError("Provide either (Th, Tc) or (sigma, omega), not both")
            self.Th = self.DeltaE / (2 * np.arctanh(self.sigma))
            self.Tc = self.DeltaE / (2 * np.arctanh(self.omega))
        
        else:
            raise ValueError("Must provide either (Th, Tc) or (sigma, omega)")
        
        # Validate
        if self.Th <= 0 or self.Tc <= 0:
            raise ValueError("Temperatures must be positive")
        if not (-1 < self.sigma < 1):
            raise ValueError(f"sigma must be in (-1, 1), got {self.sigma}")
        if not (-1 < self.omega < 1):
            raise ValueError(f"omega must be in (-1, 1), got {self.omega}")


class TwoBitDemon:
    """A demon that interacts with two bits simultaneously.
    
    The demon has two states: 'u' (up) and 'd' (down).
    
    It can undergo:
    1. Intrinsic transitions: u <-> d (thermal contact with hot reservoir)
    2. Cooperative transitions: joint bit-demon transitions with cold reservoir
    
    Default cooperative transitions (can be modified):
        - d00 <-> u01  (going up flips b2: 0->1)
        - u10 <-> d11  (going down flips b2: 0->1)
        
    These form a valid set because:
        - d_b1b2 <-> u_b1'b2' requires exactly one bit to flip
        - The energy exchange is consistent with detailed balance
    
    Attributes:
        phys_params (PhysParams): Physical parameters
        current_state (str): Current demon state ('u' or 'd')
        cooperative_transitions (dict): Maps (demon_state, b1, b2) -> (new_demon, new_b1, new_b2)
    """
    
    def __init__(self, phys_params: PhysParams = None, init_state: str = 'd'):
        """Initialize a two-bit demon.
        
        Args:
            phys_params (PhysParams): Physical parameters
            init_state (str): Initial state, 'u' or 'd'
        """
        if phys_params is None:
            self.phys_params = PhysParams(sigma=0.3, omega=0.8)
        else:
            self.phys_params = phys_params
        
        if init_state not in ['u', 'd']:
            raise ValueError(f"init_state must be 'u' or 'd', got {init_state}")
        
        self.current_state = init_state
        self.states = ['d', 'u']  # d is ground state (lower energy)
        
        # Initialize with default cooperative transitions
        self.cooperative_transitions: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}
        self._initialize_default_transitions()
        
        # Pre-compute rates
        self.intrinsic_rates = self._compute_intrinsic_rates()
        self._update_cooperative_rates()
    
    def _initialize_default_transitions(self):
        """Set up the default cooperative transitions."""
        # d00 <-> u01: demon goes up, second bit flips 0->1
        self.add_cooperative_transition('d', '0', '0', 'u', '0', '1')
        # u11 <-> d10: demon goes down, second bit flips 1->0  
        self.add_cooperative_transition('u', '1', '1', 'd', '1', '0')
    
    def add_cooperative_transition(self, d_from: str, b1_from: str, b2_from: str,
                                   d_to: str, b1_to: str, b2_to: str) -> bool:
        """Add a cooperative transition if it's valid.
        
        A valid cooperative transition must:
        1. Change the demon state (d <-> u)
        2. Change exactly one bit OR keep both bits same (for energy conservation)
        3. Not already exist
        
        Args:
            d_from, b1_from, b2_from: Initial state (demon, bit1, bit2)
            d_to, b1_to, b2_to: Final state (demon, bit1, bit2)
            
        Returns:
            bool: True if transition was added, False if invalid
        """
        # Validate demon states
        if d_from not in ['u', 'd'] or d_to not in ['u', 'd']:
            raise ValueError("Demon states must be 'u' or 'd'")
        if d_from == d_to:
            raise ValueError("Cooperative transitions must change demon state")
        
        # Validate bit states
        for b in [b1_from, b2_from, b1_to, b2_to]:
            if b not in ['0', '1']:
                raise ValueError("Bit states must be '0' or '1'")
        
        # Count bit changes
        bits_changed = (b1_from != b1_to) + (b2_from != b2_to)
        # if bits_changed > 1:
        #     raise ValueError(f"At most one bit can flip in cooperative transition, got {bits_changed}")
        
        # Add forward transition
        key_fwd = (d_from, b1_from, b2_from)
        val_fwd = (d_to, b1_to, b2_to)
        
        # Add reverse transition
        key_rev = (d_to, b1_to, b2_to)
        val_rev = (d_from, b1_from, b2_from)
        
        self.cooperative_transitions[key_fwd] = val_fwd
        self.cooperative_transitions[key_rev] = val_rev
        
        # Update rates if already initialized
        if hasattr(self, 'cooperative_rates'):
            self._update_cooperative_rates()
        
        return True
    
    def remove_cooperative_transition(self, d_state: str, b1: str, b2: str) -> bool:
        """Remove a cooperative transition (and its reverse).
        
        Args:
            d_state, b1, b2: One end of the transition to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        key = (d_state, b1, b2)
        if key not in self.cooperative_transitions:
            return False
        
        # Get the reverse
        reverse_key = self.cooperative_transitions[key]
        
        # Remove both directions
        del self.cooperative_transitions[key]
        if reverse_key in self.cooperative_transitions:
            del self.cooperative_transitions[reverse_key]
        
        self._update_cooperative_rates()
        return True
    
    def clear_cooperative_transitions(self):
        """Remove all cooperative transitions."""
        self.cooperative_transitions.clear()
        self._update_cooperative_rates()
    
    def _compute_intrinsic_rates(self) -> Dict[str, float]:
        """Compute intrinsic transition rates (u <-> d with hot reservoir)."""
        rates = {}
        # d -> u (absorb energy from hot reservoir)
        rates['d->u'] = self.phys_params.gamma * np.exp(-self.phys_params.DeltaE / (2 * self.phys_params.Th))
        # u -> d (release energy to hot reservoir)
        rates['u->d'] = self.phys_params.gamma * np.exp(self.phys_params.DeltaE / (2 * self.phys_params.Th))
        return rates
    
    def _update_cooperative_rates(self):
        """Compute cooperative transition rates based on current transitions."""
        self.cooperative_rates = {}
        omega = self.phys_params.omega
        
        for (d_from, b1_from, b2_from), (d_to, b1_to, b2_to) in self.cooperative_transitions.items():
            key = f"{d_from}{b1_from}{b2_from}->{d_to}{b1_to}{b2_to}"
            
            # Determine direction: going up (d->u) or down (u->d)
            if d_from == 'd' and d_to == 'u':
                # Going up: releasing energy to cold reservoir
                # Rate depends on bit flip direction
                bit_flips_0_to_1 = (b1_from == '0' and b1_to == '1') or (b2_from == '0' and b2_to == '1')
                if bit_flips_0_to_1:
                    self.cooperative_rates[key] = 1 - omega  # 0->1 transition
                else:
                    self.cooperative_rates[key] = 1 - omega  # 1->0 transition or no flip
            else:  # u -> d
                # Going down: absorbing energy from cold reservoir
                bit_flips_0_to_1 = (b1_from == '0' and b1_to == '1') or (b2_from == '0' and b2_to == '1')
                if bit_flips_0_to_1:
                    self.cooperative_rates[key] = 1 + omega  # Matches energy release
                else:
                    self.cooperative_rates[key] = 1 + omega
        print(self.cooperative_rates)
    def get_rates_for_joint_state(self, joint_state: str) -> Dict[str, float]:
        """Get all transition rates from a given joint state.
        
        Args:
            joint_state (str): State in format 'b1b2_d' (e.g., '01_u')
            
        Returns:
            dict: Maps transition strings to rates
        """
        bits, demon_state = joint_state.split('_')
        b1, b2 = bits[0], bits[1]
        
        rates = {}
        
        # Intrinsic transitions (only demon state changes)
        if demon_state == 'd':
            rates[f'{b1}{b2}_d->{b1}{b2}_u'] = self.intrinsic_rates['d->u']
        else:  # demon_state == 'u'
            rates[f'{b1}{b2}_u->{b1}{b2}_d'] = self.intrinsic_rates['u->d']
        
        # Cooperative transitions
        key = (demon_state, b1, b2)
        if key in self.cooperative_transitions:
            d_to, b1_to, b2_to = self.cooperative_transitions[key]
            trans_key = f"{demon_state}{b1}{b2}->{d_to}{b1_to}{b2_to}"
            if trans_key in self.cooperative_rates:
                rates[f'{b1}{b2}_{demon_state}->{b1_to}{b2_to}_{d_to}'] = self.cooperative_rates[trans_key]
        
        return rates
    
    def get_all_joint_states(self) -> List[str]:
        """Get all possible joint states."""
        states = []
        for b1 in ['0', '1']:
            for b2 in ['0', '1']:
                for d in ['d', 'u']:
                    states.append(f'{b1}{b2}_{d}')
        return states
    
    def print_transition_table(self):
        """Print a formatted table of all transitions and their rates."""
        print("\n" + "="*60)
        print("TWO-BIT DEMON TRANSITION TABLE")
        print("="*60)
        
        print("\n--- Intrinsic Transitions (Hot Reservoir) ---")
        for trans, rate in self.intrinsic_rates.items():
            print(f"  {trans}: {rate:.4f}")
        
        print("\n--- Cooperative Transitions (Cold Reservoir) ---")
        if not self.cooperative_transitions:
            print("  (none defined)")
        else:
            seen = set()
            for (d_from, b1_from, b2_from), (d_to, b1_to, b2_to) in self.cooperative_transitions.items():
                pair = tuple(sorted([(d_from, b1_from, b2_from), (d_to, b1_to, b2_to)]))
                if pair not in seen:
                    seen.add(pair)
                    key_fwd = f"{d_from}{b1_from}{b2_from}->{d_to}{b1_to}{b2_to}"
                    key_rev = f"{d_to}{b1_to}{b2_to}->{d_from}{b1_from}{b2_from}"
                    rate_fwd = self.cooperative_rates.get(key_fwd, 0)
                    rate_rev = self.cooperative_rates.get(key_rev, 0)
                    print(f"  {d_from}_{b1_from}{b2_from} <-> {d_to}_{b1_to}{b2_to}")
                    print(f"    Forward ({key_fwd}): {rate_fwd:.4f}")
                    print(f"    Reverse ({key_rev}): {rate_rev:.4f}")
        
        print("\n" + "="*60)


class SingleBitDemon:
    """A standard single-bit demon for comparison.
    
    This is the classic demon that interacts with one bit at a time.
    States: 'd' (down/ground) and 'u' (up/excited)
    
    Transitions:
        - Intrinsic: u <-> d (hot reservoir)
        - Cooperative: 0_d <-> 1_u (cold reservoir, bit flip + demon flip)
    """
    
    def __init__(self, phys_params: PhysParams = None, init_state: str = 'd'):
        """Initialize a single-bit demon.
        
        Args:
            phys_params (PhysParams): Physical parameters
            init_state (str): Initial state, 'u' or 'd'
        """
        if phys_params is None:
            self.phys_params = PhysParams(sigma=0.3, omega=0.8)
        else:
            self.phys_params = phys_params
        
        if init_state not in ['u', 'd']:
            raise ValueError(f"init_state must be 'u' or 'd', got {init_state}")
        
        self.current_state = init_state
        self.states = ['d', 'u']
        
        self.intrinsic_rates = self._compute_intrinsic_rates()
        self.cooperative_rates = self._compute_cooperative_rates()
    
    def _compute_intrinsic_rates(self) -> Dict[str, float]:
        """Compute intrinsic transition rates."""
        rates = {}
        rates['d->u'] = self.phys_params.gamma * np.exp(-self.phys_params.DeltaE / (2 * self.phys_params.Th))
        rates['u->d'] = self.phys_params.gamma * np.exp(self.phys_params.DeltaE / (2 * self.phys_params.Th))
        return rates
    
    def _compute_cooperative_rates(self) -> Dict[str, float]:
        """Compute cooperative transition rates."""
        omega = self.phys_params.omega
        rates = {}
        # 0_d -> 1_u: bit goes 0->1, demon goes d->u
        rates['0_d->1_u'] = 1 - omega
        # 1_u -> 0_d: bit goes 1->0, demon goes u->d
        rates['1_u->0_d'] = 1 + omega
        return rates
    
    def get_rates_for_joint_state(self, joint_state: str) -> Dict[str, float]:
        """Get all transition rates from a given joint state.
        
        Args:
            joint_state (str): State in format 'b_d' (e.g., '0_d')
            
        Returns:
            dict: Maps transition strings to rates
        """
        bit, demon_state = joint_state.split('_')
        rates = {}
        
        # Intrinsic transitions
        if demon_state == 'd':
            rates[f'{bit}_d->{bit}_u'] = self.intrinsic_rates['d->u']
        else:
            rates[f'{bit}_u->{bit}_d'] = self.intrinsic_rates['u->d']
        
        # Cooperative transitions
        if bit == '0' and demon_state == 'd':
            rates['0_d->1_u'] = self.cooperative_rates['0_d->1_u']
        elif bit == '1' and demon_state == 'u':
            rates['1_u->0_d'] = self.cooperative_rates['1_u->0_d']
        
        return rates


if __name__ == "__main__":
    # Demo: Create and inspect a two-bit demon
    print("Creating Two-Bit Demon with default parameters...")
    params = PhysParams(sigma=0.3, omega=0.8)
    demon = TwoBitDemon(phys_params=params)
    
    demon.print_transition_table()
    
    # Test getting rates for different states
    print("\nRates for joint state '00_d':")
    rates = demon.get_rates_for_joint_state('00_d')
    for trans, rate in rates.items():
        print(f"  {trans}: {rate:.4f}")
    
    print("\nRates for joint state '10_u':")
    rates = demon.get_rates_for_joint_state('10_u')
    for trans, rate in rates.items():
        print(f"  {trans}: {rate:.4f}")
    
    # Demo: Add a custom transition
    print("\n--- Adding custom transition: d01 <-> u11 ---")
    demon.add_cooperative_transition('d', '0', '1', 'u', '1', '1')
    demon.print_transition_table()
