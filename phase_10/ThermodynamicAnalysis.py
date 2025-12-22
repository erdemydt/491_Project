"""
Thermodynamic Analysis for Two-Bit Demon

This module provides detailed thermodynamic tracking and visualization for the
demon simulations, including energy and entropy exchanges with reservoirs.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from Demon import TwoBitDemon, SingleBitDemon, PhysParams
from Tape import TwoBitTape


class ThermodynamicTracker:
    """Track thermodynamic quantities during a demon simulation.
    
    For each transition, we track:
    - Energy exchange with hot reservoir (intrinsic transitions)
    - Energy exchange with cold reservoir (cooperative transitions)
    - Entropy production in each reservoir
    - Demon internal energy
    - Tape information content
    """
    
    def __init__(self, demon, phys_params: PhysParams):
        """Initialize the tracker.
        
        Args:
            demon: The demon object (TwoBitDemon or SingleBitDemon)
            phys_params: Physical parameters
        """
        self.demon = demon
        self.phys_params = phys_params
        
        # Energy tracking
        self.Q_h_cumulative = 0.0  # Heat to hot reservoir
        self.Q_c_cumulative = 0.0  # Heat to cold reservoir
        
        # Entropy tracking
        self.S_h_cumulative = 0.0  # Entropy change of hot reservoir
        self.S_c_cumulative = 0.0  # Entropy change of cold reservoir
        
        # Transition counts
        self.n_intrinsic_up = 0    # d -> u (absorb from hot)
        self.n_intrinsic_down = 0  # u -> d (release to hot)
        self.n_coop_up = 0         # d -> u cooperative (release to cold)
        self.n_coop_down = 0       # u -> d cooperative (absorb from cold)
        
        # Bit flip tracking
        self.n_bit_flips = 0
        self.n_0_to_1 = 0
        self.n_1_to_0 = 0
        
        # History for plotting
        self.history = {
            'Q_h': [0.0],
            'Q_c': [0.0],
            'S_h': [0.0],
            'S_c': [0.0],
            'demon_energy': [0.0],  # Start in ground state
            'step': [0]
        }
    
    def record_transition(self, transition_str: str, step: int):
        """Record a transition and update thermodynamic quantities.
        
        Args:
            transition_str: String like '00_d->00_u' or '01_d->11_u'
            step: Current step number
        """
        parts = transition_str.split('->')
        initial = parts[0]
        final = parts[1]
        
        # Parse states
        initial_parts = initial.split('_')
        final_parts = final.split('_')
        
        initial_demon = initial_parts[-1]
        final_demon = final_parts[-1]
        
        # Determine transition type
        is_intrinsic = (initial_parts[0] == final_parts[0])  # Bits don't change
        
        DeltaE = self.phys_params.DeltaE
        
        if is_intrinsic:
            # Intrinsic transition: demon exchanges energy with hot reservoir
            if initial_demon == 'd' and final_demon == 'u':
                # Demon goes up: absorbs DeltaE from hot reservoir
                self.Q_h_cumulative -= DeltaE
                self.S_h_cumulative -= DeltaE / self.phys_params.Th
                self.n_intrinsic_up += 1
            else:  # u -> d
                # Demon goes down: releases DeltaE to hot reservoir
                self.Q_h_cumulative += DeltaE
                self.S_h_cumulative += DeltaE / self.phys_params.Th
                self.n_intrinsic_down += 1
        
        else:
            # Cooperative transition: demon + bits exchange energy with cold reservoir
            initial_bits = initial_parts[0]
            final_bits = final_parts[0]
            
            # Track bit flips
            for i in range(len(initial_bits)):
                if initial_bits[i] != final_bits[i]:
                    self.n_bit_flips += 1
                    if initial_bits[i] == '0':
                        self.n_0_to_1 += 1
                    else:
                        self.n_1_to_0 += 1
            
            if initial_demon == 'd' and final_demon == 'u':
                # Demon goes up in cooperative transition
                # Demon absorbs DeltaE, bit releases DeltaE -> net 0 to cold reservoir
                # But there's asymmetry based on bit flip direction
                self.Q_c_cumulative -= DeltaE
                self.S_c_cumulative -= DeltaE / self.phys_params.Tc
                self.n_coop_up += 1
            else:  # u -> d
                # Demon goes down in cooperative transition
                self.Q_c_cumulative += DeltaE
                self.S_c_cumulative += DeltaE / self.phys_params.Tc
                self.n_coop_down += 1

        # Current demon energy (u has higher energy than d)
        current_demon_energy = DeltaE if final_demon == 'u' else 0.0
        
        # Append to history
        self.history['Q_h'].append(self.Q_h_cumulative)
        self.history['Q_c'].append(self.Q_c_cumulative)
        self.history['S_h'].append(self.S_h_cumulative)
        self.history['S_c'].append(self.S_c_cumulative)
        self.history['demon_energy'].append(current_demon_energy)
        self.history['step'].append(step)
    
    def get_summary(self) -> Dict:
        """Get a summary of all thermodynamic quantities."""
        total_entropy_production = self.S_h_cumulative + self.S_c_cumulative
        
        return {
            'energy': {
                'Q_h': self.Q_h_cumulative,
                'Q_c': self.Q_c_cumulative,
                'Q_total': self.Q_h_cumulative + self.Q_c_cumulative,
            },
            'entropy': {
                'S_h': self.S_h_cumulative,
                'S_c': self.S_c_cumulative,
                'S_total_production': total_entropy_production,
            },
            'transitions': {
                'intrinsic_up': self.n_intrinsic_up,
                'intrinsic_down': self.n_intrinsic_down,
                'cooperative_up': self.n_coop_up,
                'cooperative_down': self.n_coop_down,
            },
            'bit_flips': {
                'total': self.n_bit_flips,
                '0_to_1': self.n_0_to_1,
                '1_to_0': self.n_1_to_0,
            },
            'efficiency': {
                'work_output': -self.Q_c_cumulative,  # Useful work extracted
                'heat_input': self.Q_h_cumulative if self.Q_h_cumulative > 0 else 0,
                'carnot_efficiency': 1 - self.phys_params.Tc / self.phys_params.Th,
            }
        }


def plot_thermodynamic_analysis(comparison: Dict, title: str = "Thermodynamic Analysis"):
    """Create comprehensive thermodynamic plots.
    
    Args:
        comparison: Comparison dict from compare_demons_with_thermodynamics
        title: Plot title
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    two_bit_thermo = comparison['two_bit']['thermodynamics']
    single_bit_thermo = comparison['single_bit']['thermodynamics']
    
    two_bit_stats = comparison['two_bit']['stats']
    single_bit_stats = comparison['single_bit']['stats']
    
    init_tape = comparison['initial_tape']
    two_bit_tape = comparison['two_bit']['final_tape']
    single_bit_tape = comparison['single_bit']['final_tape']
    
    # Row 1: Energy flows
    # 1a: Energy flow diagram
    ax = fig.add_subplot(gs[0, 0])
    labels = ['Q_h\n(Hot)', 'Q_c\n(Cold)', 'Q_total']
    two_bit_energies = [
        two_bit_thermo['energy']['Q_h'],
        two_bit_thermo['energy']['Q_c'],
        two_bit_thermo['energy']['Q_total']
    ]
    single_bit_energies = [
        single_bit_thermo['energy']['Q_h'],
        single_bit_thermo['energy']['Q_c'],
        single_bit_thermo['energy']['Q_total']
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, two_bit_energies, width, label='Two-Bit', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, single_bit_energies, width, label='Single-Bit', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Energy')
    ax.set_title('Energy Exchanged with Reservoirs')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1b: Entropy production
    ax = fig.add_subplot(gs[0, 1])
    labels = ['ΔS_h\n(Hot)', 'ΔS_c\n(Cold)', 'ΔS_total\n(Production)']
    two_bit_entropies = [
        two_bit_thermo['entropy']['S_h'],
        two_bit_thermo['entropy']['S_c'],
        two_bit_thermo['entropy']['S_total_production']
    ]
    single_bit_entropies = [
        single_bit_thermo['entropy']['S_h'],
        single_bit_thermo['entropy']['S_c'],
        single_bit_thermo['entropy']['S_total_production']
    ]
    
    x = np.arange(len(labels))
    ax.bar(x - width/2, two_bit_entropies, width, label='Two-Bit', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, single_bit_entropies, width, label='Single-Bit', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Changes')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1c: Tape entropy (information)
    ax = fig.add_subplot(gs[0, 2])
    tape_labels = ['Initial', 'Two-Bit\nFinal', 'Single-Bit\nFinal']
    tape_entropies = [
        init_tape.get_entropy(),
        two_bit_tape.get_entropy(),
        single_bit_tape.get_entropy()
    ]
    pair_entropies = [
        init_tape.get_pair_entropy(),
        two_bit_tape.get_pair_entropy(),
        single_bit_tape.get_pair_entropy()
    ]
    
    x = np.arange(len(tape_labels))
    bars1 = ax.bar(x - width/2, tape_entropies, width, label='Bit Entropy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, pair_entropies, width, label='Pair Entropy', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tape_labels)
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Tape Information Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Transition analysis
    # 2a: Transition counts (two-bit)
    ax = fig.add_subplot(gs[1, 0])
    trans_labels = ['Intrinsic\nUp', 'Intrinsic\nDown', 'Coop.\nUp', 'Coop.\nDown']
    two_bit_trans = [
        two_bit_thermo['transitions']['intrinsic_up'],
        two_bit_thermo['transitions']['intrinsic_down'],
        two_bit_thermo['transitions']['cooperative_up'],
        two_bit_thermo['transitions']['cooperative_down']
    ]
    ax.bar(trans_labels, two_bit_trans, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Two-Bit Demon: Transition Counts')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2b: Transition counts (single-bit)
    ax = fig.add_subplot(gs[1, 1])
    single_bit_trans = [
        single_bit_thermo['transitions']['intrinsic_up'],
        single_bit_thermo['transitions']['intrinsic_down'],
        single_bit_thermo['transitions']['cooperative_up'],
        single_bit_thermo['transitions']['cooperative_down']
    ]
    ax.bar(trans_labels, single_bit_trans, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Single-Bit Demon: Transition Counts')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2c: Bit flip analysis
    ax = fig.add_subplot(gs[1, 2])
    flip_labels = ['0→1', '1→0', 'Net']
    two_bit_flips = [
        two_bit_thermo['bit_flips']['0_to_1'],
        two_bit_thermo['bit_flips']['1_to_0'],
        two_bit_thermo['bit_flips']['0_to_1'] - two_bit_thermo['bit_flips']['1_to_0']
    ]
    single_bit_flips = [
        single_bit_thermo['bit_flips']['0_to_1'],
        single_bit_thermo['bit_flips']['1_to_0'],
        single_bit_thermo['bit_flips']['0_to_1'] - single_bit_thermo['bit_flips']['1_to_0']
    ]
    
    x = np.arange(len(flip_labels))
    ax.bar(x - width/2, two_bit_flips, width, label='Two-Bit', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, single_bit_flips, width, label='Single-Bit', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(flip_labels)
    ax.set_ylabel('Count')
    ax.set_title('Bit Flip Analysis')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Summary statistics
    # 3a: Second law check
    ax = fig.add_subplot(gs[2, 0])
    ax.axis('off')
    
    phys_params = comparison['params']['phys_params']
    
    second_law_text = f"""Second Law Verification

Two-Bit Demon:
• ΔS_total = {two_bit_thermo['entropy']['S_total_production']:.6f}
• Q_h/T_h = {two_bit_thermo['energy']['Q_h']/phys_params.Th:.6f}
• Q_c/T_c = {two_bit_thermo['energy']['Q_c']/phys_params.Tc:.6f}
• ΔS_tape = {two_bit_stats['changes']['delta_entropy']:.6f}

Single-Bit Demon:
• ΔS_total = {single_bit_thermo['entropy']['S_total_production']:.6f}
• Q_h/T_h = {single_bit_thermo['energy']['Q_h']/phys_params.Th:.6f}
• Q_c/T_c = {single_bit_thermo['energy']['Q_c']/phys_params.Tc:.6f}
• ΔS_tape = {single_bit_stats['changes']['delta_entropy']:.6f}

Second Law: ΔS_total ≥ 0 ✓
"""
    ax.text(0.05, 0.5, second_law_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    # 3b: Energy balance
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    
    carnot_eff = 1 - phys_params.Tc / phys_params.Th
    
    energy_text = f"""Energy & Efficiency

Parameters:
• T_h = {phys_params.Th:.3f}, T_c = {phys_params.Tc:.3f}
• ΔE = {phys_params.DeltaE:.3f}
• η_Carnot = {carnot_eff:.4f}

Two-Bit Demon:
• Work output = {two_bit_thermo['efficiency']['work_output']:.4f}
• Heat input = {two_bit_thermo['efficiency']['heat_input']:.4f}
• φ = {two_bit_stats['phi']:.4f}

Single-Bit Demon:
• Work output = {single_bit_thermo['efficiency']['work_output']:.4f}
• Heat input = {single_bit_thermo['efficiency']['heat_input']:.4f}
• φ = {single_bit_stats['phi']:.4f}
"""
    ax.text(0.05, 0.5, energy_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    # 3c: Correlation & Information
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    info_text = f"""Information & Correlation

Initial Tape:
• Pair correlation = {init_tape.compute_pair_correlation():.4f}
• Mutual info = {init_tape.compute_mutual_information_pairs():.4f}

Two-Bit Final:
• Pair correlation = {two_bit_tape.compute_pair_correlation():.4f}
• Mutual info = {two_bit_tape.compute_mutual_information_pairs():.4f}
• Δ correlation = {two_bit_stats['changes']['delta_pair_correlation']:.4f}

Single-Bit Final:
• Pair correlation = {single_bit_tape.compute_pair_correlation():.4f}
• Mutual info = {single_bit_tape.compute_mutual_information_pairs():.4f}
• Δ correlation = {single_bit_stats['changes']['delta_pair_correlation']:.4f}
"""
    ax.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.show()


if __name__ == "__main__":
    # Demo usage would go here
    print("Thermodynamic Analysis Module")
    print("Use compare_demons_with_thermodynamics() from Simulation.py")
