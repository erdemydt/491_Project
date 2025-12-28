"""
Phase 10: Two-Bit Demon Simulation

Simulation framework for the two-bit demon that:
- Processes tape in pairs
- Compares 2-bit demon vs standard 1-bit demon
- Analyzes correlation changes in the tape
- Computes phi, entropy changes, and energy transfer
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from Demon import TwoBitDemon, SingleBitDemon, PhysParams
from Tape import TwoBitTape, compare_tapes
from ThermodynamicAnalysis import ThermodynamicTracker, plot_thermodynamic_analysis


class TwoBitDemonSimulation:
    """Simulation for a two-bit demon interacting with a tape.
    
    The demon processes the tape in pairs: (b0, b1), (b2, b3), etc.
    Each pair interacts with the demon for a time window τ using Gillespie algorithm.
    
    Attributes:
        demon (TwoBitDemon): The two-bit demon
        tape (TwoBitTape): The tape to process
        tau (float): Interaction time per pair
        N (int): Number of bits
        n_pairs (int): Number of pairs
    """
    
    def __init__(self, demon: TwoBitDemon, tape: TwoBitTape, tau: float):
        """Initialize the simulation.
        
        Args:
            demon (TwoBitDemon): The demon to use
            tape (TwoBitTape): The tape to process
            tau (float): Interaction time per pair
        """
        self.demon = demon
        self.init_tape = tape.copy()
        self.tau = tau
        self.N = tape.N
        self.n_pairs = tape.n_pairs
    
    def run_gillespie_for_pair(self, pair_value: str, demon_state: str) -> Tuple[str, str, float]:
        """Run Gillespie algorithm for one pair-demon interaction.
        
        Args:
            pair_value (str): Initial pair value ('00', '01', '10', or '11')
            demon_state (str): Initial demon state ('u' or 'd')
            
        Returns:
            final_pair (str): Final pair value
            final_demon_state (str): Final demon state
            time_elapsed (float): Total time of the interaction
        """
        time_elapsed = 0.0
        current_pair = pair_value
        current_demon = demon_state
        
        while time_elapsed < self.tau:
            # Form joint state
            joint_state = f'{current_pair}_{current_demon}'
            
            # Get rates
            rates = self.demon.get_rates_for_joint_state(joint_state)
            total_rate = sum(rates.values())
            
            if total_rate == 0:
                break  # No transitions possible
            
            # Time to next event
            dt = np.random.exponential(1 / total_rate)
            
            if time_elapsed + dt > self.tau:
                break  # Next event exceeds interaction window
            
            time_elapsed += dt
            
            # Choose which event occurs
            rand = np.random.uniform(0, total_rate)
            cumulative_rate = 0.0
            
            for transition, rate in rates.items():
                cumulative_rate += rate
                if rand < cumulative_rate:
                    # Parse transition: 'b1b2_d->b1'b2'_d''
                    final_state = transition.split('->')[1]
                    current_pair = final_state[:2]
                    current_demon = final_state.split('_')[1]
                    break
        
        return current_pair, current_demon, time_elapsed
    
    def run_simulation(self) -> Tuple[TwoBitTape, List[str], Dict]:
        """Run the full simulation over the tape.
        
        Returns:
            final_tape (TwoBitTape): The processed tape
            demon_history (List[str]): History of demon states
            stats (dict): Simulation statistics
        """
        # Create output tape
        final_tape = self.init_tape.copy()
        
        # Track demon state
        current_demon_state = self.demon.current_state
        demon_history = [current_demon_state]
        
        # Track transitions
        n_intrinsic = 0
        n_cooperative = 0
        
        # Process each pair
        for pair_idx in range(self.n_pairs):
            # Get current pair
            current_pair = final_tape.get_pair_at(pair_idx)
            
            # Run Gillespie
            final_pair, current_demon_state, _ = self.run_gillespie_for_pair(
                current_pair, current_demon_state
            )
            
            # Update tape
            final_tape.set_pair_at(pair_idx, final_pair)
            demon_history.append(current_demon_state)
        
        # Compute statistics
        stats = self._compute_statistics(final_tape, demon_history)
        
        return final_tape, demon_history, stats
    
    def _compute_statistics(self, final_tape: TwoBitTape, demon_history: List[str]) -> Dict:
        """Compute statistics from the simulation.
        
        Args:
            final_tape: The processed tape
            demon_history: History of demon states
            
        Returns:
            dict: Comprehensive statistics
        """
        # Bit statistics
        init_p0 = self.init_tape.probabilities[0]
        final_p0 = final_tape._compute_bit_probabilities()[0]
        init_p1 = 1 - init_p0
        final_p1 = 1 - final_p0
        # Count flipped bits
        bits_flipped = np.sum(self.init_tape.tape_arr != final_tape.tape_arr)
        phi = final_p1 - init_p1
        
        # Entropy changes
        init_entropy = self.init_tape.get_entropy()
        final_entropy = final_tape.get_entropy()
        delta_s_b = final_entropy - init_entropy
        
        # Pair distribution changes
        init_pair_dist = self.init_tape.get_pair_distribution()
        final_pair_dist = final_tape.get_pair_distribution()
        
        # Correlation changes
        init_corr = self.init_tape.compute_pair_correlation()
        final_corr = final_tape.compute_pair_correlation()
        
        init_mi = self.init_tape.compute_mutual_information_pairs()
        final_mi = final_tape.compute_mutual_information_pairs()
        
        # Demon state distribution
        demon_up_frac = demon_history.count('u') / len(demon_history)
        demon_down_frac = 1 - demon_up_frac
        
        # Energy transferred (simplified model)
        delta_e = self.demon.phys_params.DeltaE
        q_c = phi * delta_e  # Energy to cold reservoir
        
        return {
            'incoming': {
                'p0': init_p0,
                'p1': 1 - init_p0,
                'entropy': init_entropy,
                'pair_distribution': init_pair_dist,
                'pair_correlation': init_corr,
                'mutual_information': init_mi
            },
            'outgoing': {
                'p0': final_p0,
                'p1': 1 - final_p0,
                'entropy': final_entropy,
                'pair_distribution': final_pair_dist,
                'pair_correlation': final_corr,
                'mutual_information': final_mi
            },
            'changes': {
                'delta_p0': final_p0 - init_p0,
                'delta_entropy': delta_s_b,
                'delta_pair_correlation': final_corr - init_corr,
                'delta_mutual_information': final_mi - init_mi
            },
            'phi': phi,
            'bits_flipped': bits_flipped,
            'Q_c': q_c,
            'demon': {
                'up_fraction': demon_up_frac,
                'down_fraction': demon_down_frac
            },
            'N': self.N,
            'n_pairs': self.n_pairs,
            'tau': self.tau
        }


class SingleBitDemonSimulation:
    """Simulation for a standard single-bit demon for comparison.
    
    Processes the tape one bit at a time.
    """
    
    def __init__(self, demon: SingleBitDemon, tape: TwoBitTape, tau: float):
        """Initialize the simulation.
        
        Args:
            demon (SingleBitDemon): The demon to use
            tape (TwoBitTape): The tape to process (uses same tape type for comparison)
            tau (float): Interaction time per bit
        """
        self.demon = demon
        self.init_tape = tape.copy()
        self.tau = tau
        self.N = tape.N
    
    def run_gillespie_for_bit(self, bit_value: str, demon_state: str) -> Tuple[str, str, float]:
        """Run Gillespie algorithm for one bit-demon interaction.
        
        Args:
            bit_value (str): Initial bit value ('0' or '1')
            demon_state (str): Initial demon state ('u' or 'd')
            
        Returns:
            final_bit (str): Final bit value
            final_demon_state (str): Final demon state
            time_elapsed (float): Total time of the interaction
        """
        time_elapsed = 0.0
        current_bit = bit_value
        current_demon = demon_state
        
        while time_elapsed < self.tau:
            joint_state = f'{current_bit}_{current_demon}'
            rates = self.demon.get_rates_for_joint_state(joint_state)
            total_rate = sum(rates.values())
            
            if total_rate == 0:
                break
            
            dt = np.random.exponential(1 / total_rate)
            
            if time_elapsed + dt > self.tau:
                break
            
            time_elapsed += dt
            
            rand = np.random.uniform(0, total_rate)
            cumulative_rate = 0.0
            
            for transition, rate in rates.items():
                cumulative_rate += rate
                if rand < cumulative_rate:
                    final_state = transition.split('->')[1]
                    current_bit = final_state.split('_')[0]
                    current_demon = final_state.split('_')[1]
                    break
        
        return current_bit, current_demon, time_elapsed
    
    def run_simulation(self) -> Tuple[TwoBitTape, List[str], Dict]:
        """Run the full simulation over the tape.
        
        Returns:
            final_tape (TwoBitTape): The processed tape
            demon_history (List[str]): History of demon states
            stats (dict): Simulation statistics
        """
        final_tape = self.init_tape.copy()
        
        current_demon_state = self.demon.current_state
        demon_history = [current_demon_state]
        
        for bit_idx in range(self.N):
            current_bit = final_tape.tape_arr[bit_idx]
            final_bit, current_demon_state, _ = self.run_gillespie_for_bit(
                current_bit, current_demon_state
            )
            final_tape.tape_arr[bit_idx] = final_bit
            demon_history.append(current_demon_state)
        
        # Update tape probabilities
        final_tape.probabilities = final_tape._compute_bit_probabilities()
        
        stats = self._compute_statistics(final_tape, demon_history)
        return final_tape, demon_history, stats
    
    def _compute_statistics(self, final_tape: TwoBitTape, demon_history: List[str]) -> Dict:
        """Compute statistics from the simulation."""
        init_p0 = self.init_tape.probabilities[0]
        final_p0 = final_tape.probabilities[0]
        init_p1 = 1 - init_p0
        final_p1 = 1 - final_p0
        bits_flipped = np.sum(self.init_tape.tape_arr != final_tape.tape_arr)
        phi = final_p1 - init_p1
        
        init_entropy = self.init_tape.get_entropy()
        final_entropy = final_tape.get_entropy()
        delta_s_b = final_entropy - init_entropy
        
        init_pair_dist = self.init_tape.get_pair_distribution()
        final_pair_dist = final_tape.get_pair_distribution()
        
        init_corr = self.init_tape.compute_pair_correlation()
        final_corr = final_tape.compute_pair_correlation()
        
        init_mi = self.init_tape.compute_mutual_information_pairs()
        final_mi = final_tape.compute_mutual_information_pairs()
        
        demon_up_frac = demon_history.count('u') / len(demon_history)
        
        delta_e = self.demon.phys_params.DeltaE
        q_c = phi * delta_e
        
        return {
            'incoming': {
                'p0': init_p0,
                'p1': 1 - init_p0,
                'entropy': init_entropy,
                'pair_distribution': init_pair_dist,
                'pair_correlation': init_corr,
                'mutual_information': init_mi
            },
            'outgoing': {
                'p0': final_p0,
                'p1': 1 - final_p0,
                'entropy': final_entropy,
                'pair_distribution': final_pair_dist,
                'pair_correlation': final_corr,
                'mutual_information': final_mi
            },
            'changes': {
                'delta_p0': final_p0 - init_p0,
                'delta_entropy': delta_s_b,
                'delta_pair_correlation': final_corr - init_corr,
                'delta_mutual_information': final_mi - init_mi
            },
            'phi': phi,
            'bits_flipped': bits_flipped,
            'Q_c': q_c,
            'demon': {
                'up_fraction': demon_up_frac,
                'down_fraction': 1 - demon_up_frac
            },
            'N': self.N,
            'tau': self.tau
        }


def compare_demons(tape_params: Dict, phys_params: PhysParams = None, tau: float = 1.0,
                   seed: int = 42, plot: bool = True, compare_tapes_plot: bool = False,
                   two_bit_demon: TwoBitDemon = None, title: str = None) -> Dict:
    """Compare the two-bit demon with the single-bit demon on the same initial tape.
    
    Args:
        tape_params: Dictionary with tape parameters (N, p0, init_mode, etc.)
        phys_params: Physical parameters (shared by both demons)
        tau: Interaction time
        seed: Random seed for reproducibility
        plot: Whether to generate comparison plots
        compare_tapes_plot: Whether to also generate detailed tape comparison plots
        two_bit_demon: Optional pre-configured TwoBitDemon instance (if None, creates default)
        title: Optional custom title/subtitle for the plot
        
    Returns:
        dict: Comparison results including stats for both demons
    """
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8)
    
    # Create initial tape
    np.random.seed(seed)
    init_tape = TwoBitTape(**tape_params, seed=seed)
    
    # Create demons (use provided two_bit_demon or create default)
    if two_bit_demon is None:
        two_bit_demon = TwoBitDemon(phys_params=phys_params, init_state='d')
    single_bit_demon = SingleBitDemon(phys_params=phys_params, init_state='d')
    
    # Run two-bit simulation
    np.random.seed(seed + 1)  # Different seed for simulation randomness
    two_bit_sim = TwoBitDemonSimulation(demon=two_bit_demon, tape=init_tape, tau=tau)
    two_bit_final_tape, two_bit_history, two_bit_stats = two_bit_sim.run_simulation()
    
    # Run single-bit simulation (on a fresh copy of the same initial tape)
    np.random.seed(seed + 2)
    single_bit_tape = init_tape.copy()
    single_bit_sim = SingleBitDemonSimulation(demon=single_bit_demon, tape=single_bit_tape, tau=tau)
    single_bit_final_tape, single_bit_history, single_bit_stats = single_bit_sim.run_simulation()
    
    # Compile comparison
    comparison = {
        'two_bit': {
            'stats': two_bit_stats,
            'final_tape': two_bit_final_tape,
            'demon_history': two_bit_history
        },
        'single_bit': {
            'stats': single_bit_stats,
            'final_tape': single_bit_final_tape,
            'demon_history': single_bit_history
        },
        'initial_tape': init_tape,
        'params': {
            'tape_params': tape_params,
            'phys_params': phys_params,
            'tau': tau
        },
        'two_bit_demon': two_bit_demon,
        'title': title
    }
    
    if plot:
        _plot_demon_comparison(comparison)
    
    if compare_tapes_plot:
        # Generate detailed tape comparison for two-bit demon
        compare_tapes(init_tape, two_bit_final_tape, title="Two-Bit Demon: Tape Comparison")
        # Generate detailed tape comparison for single-bit demon
        compare_tapes(init_tape, single_bit_final_tape, title="Single-Bit Demon: Tape Comparison")
    
    return comparison


def _plot_demon_comparison(comparison: Dict):
    """Plot comparison between two-bit and single-bit demons."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    
    init_tape = comparison['initial_tape']
    two_bit_tape = comparison['two_bit']['final_tape']
    single_bit_tape = comparison['single_bit']['final_tape']
    
    two_bit_stats = comparison['two_bit']['stats']
    single_bit_stats = comparison['single_bit']['stats']
    
    # Row 1: Pair distributions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (tape, title) in enumerate([
        (init_tape, 'Initial'),
        (two_bit_tape, 'Two-Bit Demon'),
        (single_bit_tape, 'Single-Bit Demon')
    ]):
        ax = axes[0, idx]
        pair_dist = tape.get_pair_distribution()
        bars = ax.bar(pair_dist.keys(), pair_dist.values(), color=colors)
        ax.set_xlabel('Pair Type')
        ax.set_ylabel('Fraction')
        ax.set_title(f'{title} Pair Distribution')
        max_val = max(pair_dist.values()) if max(pair_dist.values()) > 0 else 0.5
        ax.set_ylim(0, max_val * 1.3)
        for bar, val in zip(bars, pair_dist.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Row 2: Comparison metrics
    # Left: Key metrics comparison
    ax = axes[1, 0]
    metrics = ['φ (flip frac)', 'Final corr', 'Δ corr', 'Δ MI']
    two_bit_vals = [
        two_bit_stats['phi'],
        two_bit_stats['outgoing']['pair_correlation'],
        two_bit_stats['changes']['delta_pair_correlation'],
        two_bit_stats['changes']['delta_mutual_information']
    ]
    single_bit_vals = [
        single_bit_stats['phi'],
        single_bit_stats['outgoing']['pair_correlation'],
        single_bit_stats['changes']['delta_pair_correlation'],
        single_bit_stats['changes']['delta_mutual_information']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, two_bit_vals, width, label='Two-Bit Demon', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, single_bit_vals, width, label='Single-Bit Demon', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('Key Metrics Comparison')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Middle: p0/p1 comparison (Initial vs Final for both demons)
    ax = axes[1, 1]
    labels = ['Initial', 'Two-Bit\nFinal', 'Single-Bit\nFinal']
    p0_vals = [
        two_bit_stats['incoming']['p0'],
        two_bit_stats['outgoing']['p0'],
        single_bit_stats['outgoing']['p0']
    ]
    p1_vals = [
        two_bit_stats['incoming']['p1'],
        two_bit_stats['outgoing']['p1'],
        single_bit_stats['outgoing']['p1']
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, p0_vals, width, label='p₀', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, p1_vals, width, label='p₁', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Probability')
    ax.set_title('Bit Probabilities: Initial vs Final')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    # Right: Summary text with transitions
    ax = axes[1, 2]
    ax.axis('off')
    
    # Get cooperative transitions from the demon
    two_bit_demon = comparison.get('two_bit_demon')
    transitions_text = ""
    if two_bit_demon is not None:
        seen = set()
        trans_list = []
        for (d_from, b1_from, b2_from), (d_to, b1_to, b2_to) in two_bit_demon.cooperative_transitions.items():
            pair = tuple(sorted([(d_from, b1_from, b2_from), (d_to, b1_to, b2_to)]))
            if pair not in seen:
                seen.add(pair)
                trans_list.append(f"{d_from}_{b1_from}{b2_from} ↔ {d_to}_{b1_to}{b2_to}")
        if trans_list:
            transitions_text = "\nTwo-Bit Coop. Transitions:\n" + "\n".join(f"  • {t}" for t in trans_list)
    
    summary_text = f"""Parameters:
• N = {init_tape.N}, τ = {comparison['params']['tau']}
• σ = {comparison['params']['phys_params'].sigma:.3f}
• ω = {comparison['params']['phys_params'].omega:.3f}
{transitions_text}

Two-Bit Demon:
• φ = {two_bit_stats['phi']:.4f}
• ΔS = {two_bit_stats['changes']['delta_entropy']:.4f}
• Δ corr = {two_bit_stats['changes']['delta_pair_correlation']:.4f}

Single-Bit Demon:
• φ = {single_bit_stats['phi']:.4f}
• ΔS = {single_bit_stats['changes']['delta_entropy']:.4f}
• Δ corr = {single_bit_stats['changes']['delta_pair_correlation']:.4f}

Differences (2-Bit − 1-Bit):
• Δφ = {two_bit_stats['phi'] - single_bit_stats['phi']:.4f}
• ΔΔS = {two_bit_stats['changes']['delta_entropy'] - single_bit_stats['changes']['delta_entropy']:.4f}
"""
    ax.text(0.02, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    # Build title
    main_title = f'Two-Bit vs Single-Bit Demon Comparison, τ = {comparison["params"]["tau"]}'
    custom_title = comparison.get('title')
    if custom_title:
        main_title = f'{main_title}\n{custom_title}'
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    # We want very high resolution for detailed text
    plt.savefig('final-report/demonn/two-bit-comp-plot.png', dpi=300)
    print("Saved figure: final-report/demonn/two-bit-comp-plot.png")


def sweep_tau(tau_values: List[float], tape_params: Dict, phys_params: PhysParams = None,
              n_runs: int = 5, seed_base: int = 42) -> Dict:
    """Sweep over tau values and compare both demon types.
    
    Args:
        tau_values: List of tau values to test
        tape_params: Tape parameters
        phys_params: Physical parameters
        n_runs: Number of runs per tau value (for averaging)
        seed_base: Base seed for reproducibility
        
    Returns:
        dict: Results for each tau value
    """
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8)
    
    results = {
        'tau_values': tau_values,
        'two_bit': {'phi': [], 'delta_entropy': [], 'delta_corr': []},
        'single_bit': {'phi': [], 'delta_entropy': [], 'delta_corr': []}
    }
    
    for i, tau in enumerate(tau_values):
        print(f"Processing tau = {tau} ({i+1}/{len(tau_values)})", end='\r')
        
        two_bit_phi = []
        two_bit_ds = []
        two_bit_dc = []
        single_bit_phi = []
        single_bit_ds = []
        single_bit_dc = []
        
        for run in range(n_runs):
            seed = seed_base + i * n_runs + run
            comparison = compare_demons(tape_params, phys_params, tau, seed, plot=False)
            
            two_bit_phi.append(comparison['two_bit']['stats']['phi'])
            two_bit_ds.append(comparison['two_bit']['stats']['changes']['delta_entropy'])
            two_bit_dc.append(comparison['two_bit']['stats']['changes']['delta_pair_correlation'])
            
            single_bit_phi.append(comparison['single_bit']['stats']['phi'])
            single_bit_ds.append(comparison['single_bit']['stats']['changes']['delta_entropy'])
            single_bit_dc.append(comparison['single_bit']['stats']['changes']['delta_pair_correlation'])
        
        results['two_bit']['phi'].append(np.mean(two_bit_phi))
        results['two_bit']['delta_entropy'].append(np.mean(two_bit_ds))
        results['two_bit']['delta_corr'].append(np.mean(two_bit_dc))
        
        results['single_bit']['phi'].append(np.mean(single_bit_phi))
        results['single_bit']['delta_entropy'].append(np.mean(single_bit_ds))
        results['single_bit']['delta_corr'].append(np.mean(single_bit_dc))
    
    print()
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, label) in enumerate([
        ('phi', 'Flip Fraction (φ)'),
        ('delta_entropy', 'Entropy Change (ΔS)'),
        ('delta_corr', 'Pair Correlation Change')
    ]):
        ax = axes[idx]
        ax.plot(tau_values, results['two_bit'][metric], 'o-', label='Two-Bit Demon', 
                color='steelblue', linewidth=2, markersize=6)
        ax.plot(tau_values, results['single_bit'][metric], 's--', label='Single-Bit Demon',
                color='coral', linewidth=2, markersize=6)
        ax.set_xlabel('Interaction Time (τ)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} vs τ', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric in ['phi', 'delta_entropy']:
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.suptitle('Demon Comparison: Tau Sweep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return results


def compare_demons_with_thermodynamics(tape_params: Dict, phys_params: PhysParams = None, 
                                      tau: float = 1.0, seed: int = 42, 
                                      two_bit_demon: TwoBitDemon = None, 
                                      title: str = None, plot: bool = True) -> Dict:
    """Compare demons WITH detailed thermodynamic tracking.
    
    This function runs the same comparison as compare_demons, but also tracks
    all thermodynamic quantities (energy and entropy exchanges with reservoirs).
    
    Args:
        tape_params: Dictionary with tape parameters
        phys_params: Physical parameters
        tau: Interaction time
        seed: Random seed
        two_bit_demon: Optional pre-configured demon
        title: Optional plot title
        plot: Whether to show thermodynamic plots
        
    Returns:
        dict: Comparison results with thermodynamics
    """
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8)
    
    # Create initial tape
    np.random.seed(seed)
    init_tape = TwoBitTape(**tape_params)
    
    # Create demons
    if two_bit_demon is None:
        two_bit_demon = TwoBitDemon(phys_params=phys_params, init_state='d')
    single_bit_demon = SingleBitDemon(phys_params=phys_params, init_state='d')
    
    # Run two-bit simulation WITH thermodynamic tracking
    np.random.seed(seed + 1)
    two_bit_tape, two_bit_history, two_bit_stats, two_bit_thermo = run_simulation_with_thermodynamics(
        two_bit_demon, init_tape.copy(), tau, phys_params, is_two_bit=True
    )
    
    # Run single-bit simulation WITH thermodynamic tracking
    np.random.seed(seed + 2)
    single_bit_tape, single_bit_history, single_bit_stats, single_bit_thermo = run_simulation_with_thermodynamics(
        single_bit_demon, init_tape.copy(), tau, phys_params, is_two_bit=False
    )
    
    # Compile comparison
    comparison = {
        'two_bit': {
            'stats': two_bit_stats,
            'final_tape': two_bit_tape,
            'demon_history': two_bit_history,
            'thermodynamics': two_bit_thermo
        },
        'single_bit': {
            'stats': single_bit_stats,
            'final_tape': single_bit_tape,
            'demon_history': single_bit_history,
            'thermodynamics': single_bit_thermo
        },
        'initial_tape': init_tape,
        'params': {
            'tape_params': tape_params,
            'phys_params': phys_params,
            'tau': tau
        },
        'two_bit_demon': two_bit_demon,
        'title': title
    }
    
    if plot:
        plot_title = f"Thermodynamic Analysis, τ = {tau}"
        if title:
            plot_title = f"{plot_title}\n{title}"
        plot_thermodynamic_analysis(comparison, title=plot_title)
    
    return comparison


def run_simulation_with_thermodynamics(demon, init_tape: TwoBitTape, tau: float,
                                      phys_params: PhysParams, is_two_bit: bool = True):
    """Run simulation with full thermodynamic tracking.
    
    Args:
        demon: Demon object
        init_tape: Initial tape
        tau: Interaction time
        phys_params: Physical parameters
        is_two_bit: Whether this is a two-bit demon
        
    Returns:
        final_tape, demon_history, stats, thermodynamics
    """
    # Create thermodynamic tracker
    tracker = ThermodynamicTracker(demon, phys_params)
    
    # Run simulation
    final_tape = init_tape.copy()
    current_demon_state = demon.current_state
    demon_history = [current_demon_state]
    
    step = 0
    
    if is_two_bit:
        # Two-bit demon: process pairs
        for pair_idx in range(init_tape.n_pairs):
            current_pair = final_tape.get_pair_at(pair_idx)
            
            # Run Gillespie for this pair
            time_elapsed = 0.0
            current_pair_val = current_pair
            current_demon_val = current_demon_state
            
            while time_elapsed < tau:
                joint_state = f'{current_pair_val}_{current_demon_val}'
                rates = demon.get_rates_for_joint_state(joint_state)
                total_rate = sum(rates.values())
                
                if total_rate == 0:
                    break
                
                dt = np.random.exponential(1 / total_rate)
                if time_elapsed + dt > tau:
                    break
                
                time_elapsed += dt
                
                # Choose transition
                rand = np.random.uniform(0, total_rate)
                cumulative_rate = 0.0
                
                for transition, rate in rates.items():
                    cumulative_rate += rate
                    if rand < cumulative_rate:
                        # Record transition for thermodynamics
                        tracker.record_transition(transition, step)
                        step += 1
                        
                        # Update state
                        final_state = transition.split('->')[1]
                        current_pair_val = final_state[:2]
                        current_demon_val = final_state.split('_')[1]
                        break
            
            # Update tape and demon
            final_tape.set_pair_at(pair_idx, current_pair_val)
            current_demon_state = current_demon_val
            demon_history.append(current_demon_state)
    
    else:
        # Single-bit demon: process bits
        for bit_idx in range(init_tape.N):
            current_bit = final_tape.tape_arr[bit_idx]
            
            # Run Gillespie for this bit
            time_elapsed = 0.0
            current_bit_val = current_bit
            current_demon_val = current_demon_state
            
            while time_elapsed < tau:
                joint_state = f'{current_bit_val}_{current_demon_val}'
                rates = demon.get_rates_for_joint_state(joint_state)
                total_rate = sum(rates.values())
                
                if total_rate == 0:
                    break
                
                dt = np.random.exponential(1 / total_rate)
                if time_elapsed + dt > tau:
                    break
                
                time_elapsed += dt
                
                # Choose transition
                rand = np.random.uniform(0, total_rate)
                cumulative_rate = 0.0
                
                for transition, rate in rates.items():
                    cumulative_rate += rate
                    if rand < cumulative_rate:
                        # Record transition for thermodynamics
                        tracker.record_transition(transition, step)
                        step += 1
                        
                        # Update state
                        final_state = transition.split('->')[1]
                        current_bit_val = final_state.split('_')[0]
                        current_demon_val = final_state.split('_')[1]
                        break
            
            # Update tape and demon
            final_tape.tape_arr[bit_idx] = current_bit_val
            current_demon_state = current_demon_val
            demon_history.append(current_demon_state)
    
    # Update tape probabilities
    final_tape.probabilities = final_tape._compute_bit_probabilities()
    
    # Compute statistics (same as before)
    init_p0 = init_tape.probabilities[0]
    final_p0 = final_tape.probabilities[0]
    init_p1 = 1 - init_p0
    final_p1 = 1 - final_p0
    bits_flipped = np.sum(init_tape.tape_arr != final_tape.tape_arr)
    phi = final_p1 - init_p1
    
    init_entropy = init_tape.get_entropy()
    final_entropy = final_tape.get_entropy()
    delta_s_b = final_entropy - init_entropy
    
    init_pair_dist = init_tape.get_pair_distribution()
    final_pair_dist = final_tape.get_pair_distribution()
    
    init_corr = init_tape.compute_pair_correlation()
    final_corr = final_tape.compute_pair_correlation()
    
    init_mi = init_tape.compute_mutual_information_pairs()
    final_mi = final_tape.compute_mutual_information_pairs()
    
    demon_up_frac = demon_history.count('u') / len(demon_history)
    
    delta_e = phys_params.DeltaE
    q_c = phi * delta_e
    
    stats = {
        'incoming': {
            'p0': init_p0,
            'p1': 1 - init_p0,
            'entropy': init_entropy,
            'pair_distribution': init_pair_dist,
            'pair_correlation': init_corr,
            'mutual_information': init_mi
        },
        'outgoing': {
            'p0': final_p0,
            'p1': 1 - final_p0,
            'entropy': final_entropy,
            'pair_distribution': final_pair_dist,
            'pair_correlation': final_corr,
            'mutual_information': final_mi
        },
        'changes': {
            'delta_p0': final_p0 - init_p0,
            'delta_entropy': delta_s_b,
            'delta_pair_correlation': final_corr - init_corr,
            'delta_mutual_information': final_mi - init_mi
        },
        'phi': phi,
        'bits_flipped': bits_flipped,
        'Q_c': q_c,
        'demon': {
            'up_fraction': demon_up_frac,
            'down_fraction': 1 - demon_up_frac
        },
        'N': init_tape.N,
        'tau': tau
    }
    
    # Get thermodynamic summary
    thermodynamics = tracker.get_summary()
    return final_tape, demon_history, stats, thermodynamics


def sweep_p0_entropy_analysis_v2(p0_values: List[float], tape_size: int = 10000, tau: float = 50.0,
                              phys_params: PhysParams = None, seed_base: int = 42,
                              two_bit_demon: TwoBitDemon = None) -> Dict:
    """Sweep over p0 values and plot input tape entropy vs entropy change and Q_c.
    
    Uses full tape entropy (not per-bit) consistently throughout.
    Generates separate plots for one-bit and two-bit demons.
    
    Args:
        p0_values: List of p0 values to sweep (0 to 1)
        tape_size: Size of tape (N)
        tau: Interaction time
        phys_params: Physical parameters
        seed_base: Base seed for reproducibility
        two_bit_demon: Optional pre-configured TwoBitDemon
        
    Returns:
        dict: Results for each p0 value
    """
    if phys_params is None:
        phys_params = PhysParams(sigma=0.4, omega=0.6)
    
    # Create demons if not provided
    if two_bit_demon is None:
        two_bit_demon_inst = TwoBitDemon(phys_params=phys_params, init_state='d')
    else:
        two_bit_demon_inst = two_bit_demon
    
    single_bit_demon_inst = SingleBitDemon(phys_params=phys_params, init_state='d')
    
    # Storage for results
    results = {
        'p0_values': p0_values,
        'two_bit': {
            'input_entropy': [],           # S_in * N
            'output_total_entropy': [],    # S_out * N + S_H + S_C
            'Q_c': [],                      # Heat to cold reservoir
            'output_bias': [],              # Final p0
            'S_h_plus_S_c': [],             # S_H + S_C only
        },
        'single_bit': {
            'input_entropy': [],           # S_in * N
            'output_total_entropy': [],    # S_out * N + S_H + S_C
            'Q_c': [],                      # Heat to cold reservoir
            'output_bias': [],              # Final p0
            'S_h_plus_S_c': [],             # S_H + S_C only
        }
    }
    
    print(f"\nSweeping p0 from {p0_values[0]} to {p0_values[-1]} ({len(p0_values)} points)...")
    print(f"Tape size: N={tape_size}, τ={tau}")
    
    for i, p0 in enumerate(p0_values):
        print(f"  [{i+1}/{len(p0_values)}] p0 = {p0:.3f}...", end=" ", flush=True)
        
        # Create tape with this p0
        tape_params = {'N': tape_size, 'p0': p0, 'init_mode': 'random', 'seed': seed_base + i}
        init_tape = TwoBitTape(**tape_params)
        
        # Get input entropy (multiply by N for full tape entropy)
        input_entropy_per_bit = init_tape.get_entropy()
        input_entropy_full = input_entropy_per_bit * tape_size
        
        # Run two-bit simulation with thermodynamics
        np.random.seed(seed_base + 1000 + i)
        two_bit_final_tape, two_bit_history, two_bit_stats, two_bit_thermo = run_simulation_with_thermodynamics(
            two_bit_demon_inst, init_tape.copy(), tau, phys_params, is_two_bit=True
        )
        
        # Two-bit results
        output_tape_entropy = two_bit_final_tape.get_entropy() * tape_size
        S_h = two_bit_thermo['entropy']['S_h']
        S_c = two_bit_thermo['entropy']['S_c']
        
        results['two_bit']['input_entropy'].append(input_entropy_full)
        results['two_bit']['output_total_entropy'].append(output_tape_entropy + S_h + S_c)
        results['two_bit']['Q_c'].append(two_bit_thermo['energy']['Q_c'])
        results['two_bit']['output_bias'].append(two_bit_stats['outgoing']['p0'])
        results['two_bit']['S_h_plus_S_c'].append(S_h + S_c)
        
        # Run single-bit simulation with thermodynamics (on fresh tape copy)
        np.random.seed(seed_base + 2000 + i)
        single_bit_tape = TwoBitTape(**tape_params)  # Fresh tape with same p0
        print(single_bit_tape.get_pair_distribution())
        single_bit_final_tape, single_bit_history, single_bit_stats, single_bit_thermo = run_simulation_with_thermodynamics(
            single_bit_demon_inst, single_bit_tape, tau, phys_params, is_two_bit=False
        )
        
        # Single-bit results
        output_tape_entropy = single_bit_final_tape.get_entropy() * tape_size
        S_h = single_bit_thermo['entropy']['S_h']
        S_c = single_bit_thermo['entropy']['S_c']
        
        results['single_bit']['input_entropy'].append(input_entropy_full)
        results['single_bit']['output_total_entropy'].append(output_tape_entropy + S_h + S_c)
        results['single_bit']['Q_c'].append(single_bit_thermo['energy']['Q_c'])
        results['single_bit']['output_bias'].append(single_bit_stats['outgoing']['p0'])
        results['single_bit']['S_h_plus_S_c'].append(S_h + S_c)
        
        print(f"✓")
    
    # Create plots
    plot_p0_entropy_analysis_v2(results, tape_size, tau, phys_params)
    
    return results


def plot_p0_entropy_analysis_v2(results: Dict, tape_size: int, tau: float, phys_params: PhysParams):
    """Create 4 plots for single-bit and two-bit demons vs input tape entropy."""
    p0_values = results['p0_values']
    
    # Create 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Output total entropy (tape + S_H + S_C)
    axes[0, 0].plot(results['two_bit']['input_entropy'], results['two_bit']['output_total_entropy'],
                   marker='o', linewidth=2, label='Two-Bit', color='steelblue')
    axes[0, 0].plot(results['single_bit']['input_entropy'], results['single_bit']['output_total_entropy'],
                   marker='s', linewidth=2, label='Single-Bit', color='coral')
    axes[0, 0].set_xlabel('Input Tape Entropy (S_in × N)')
    axes[0, 0].set_ylabel('Output Total Entropy (Tape + S_H + S_C)')
    axes[0, 0].set_title('Output Total Entropy vs Input Tape Entropy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q_c (heat to cold reservoir)
    axes[0, 1].plot(results['two_bit']['input_entropy'], results['two_bit']['Q_c'],
                   marker='o', linewidth=2, label='Two-Bit', color='steelblue')
    axes[0, 1].plot(results['single_bit']['input_entropy'], results['single_bit']['Q_c'],
                   marker='s', linewidth=2, label='Single-Bit', color='coral')
    axes[0, 1].set_xlabel('Input Tape Entropy (S_in × N)')
    axes[0, 1].set_ylabel('Q_c (Heat to Cold Reservoir)')
    axes[0, 1].set_title('Q_c vs Input Tape Entropy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Output bias (final p0)
    axes[1, 0].plot(results['two_bit']['input_entropy'], results['two_bit']['output_bias'],
                   marker='o', linewidth=2, label='Two-Bit', color='steelblue')
    axes[1, 0].plot(results['single_bit']['input_entropy'], results['single_bit']['output_bias'],
                   marker='s', linewidth=2, label='Single-Bit', color='coral')
    axes[1, 0].set_xlabel('Input Tape Entropy (S_in × N)')
    axes[1, 0].set_ylabel('Output Bias (Final p₀)')
    axes[1, 0].set_title('Output Bias vs Input Tape Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. S_H + S_C only
    axes[1, 1].plot(results['two_bit']['input_entropy'], results['two_bit']['S_h_plus_S_c'],
                   marker='o', linewidth=2, label='Two-Bit', color='steelblue')
    axes[1, 1].plot(results['single_bit']['input_entropy'], results['single_bit']['S_h_plus_S_c'],
                   marker='s', linewidth=2, label='Single-Bit', color='coral')
    axes[1, 1].set_xlabel('Input Tape Entropy (S_in × N)')
    axes[1, 1].set_ylabel('Reservoir Entropy (S_H + S_C)')
    axes[1, 1].set_title('Reservoir Entropy vs Input Tape Entropy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Demon Entropy & Q_c Analysis (N={tape_size}, τ={tau})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def sweep_p0_input_entropy_analysis(tape_size: int = 10000, tau: float = 50.0,
                                    phys_params: PhysParams = None, seed_base: int = 42,
                                    two_bit_demon: TwoBitDemon = None, n_points: int = 11,
                                    plot: bool = True) -> Dict:
    """Sweep over p0 from 0 to 0.5 and plot input entropy vs output entropy and Q_total.
    
    Uses input entropy as x-axis for better physical insight.
    Generates separate plots for one-bit and two-bit demons.
    
    Args:
        tape_size: Size of tape (N)
        tau: Interaction time
        phys_params: Physical parameters
        seed_base: Base seed for reproducibility
        two_bit_demon: Optional pre-configured TwoBitDemon
        n_points: Number of points from p0=0 to p0=0.5
        
    Returns:
        dict: Results for each p0 value
    """
    if phys_params is None:
        phys_params = PhysParams(sigma=0.4, omega=0.6)
    
    # Create demons if not provided
    if two_bit_demon is None:
        two_bit_demon_inst = TwoBitDemon(phys_params=phys_params, init_state='d')
    else:
        two_bit_demon_inst = two_bit_demon
    
    single_bit_demon_inst = SingleBitDemon(phys_params=phys_params, init_state='d')
    
    # p0 values from 0 to 0.5
    p0_values = np.linspace(0.0, 0.5, n_points)
    
    # Storage for results
    results = {
        'p0_values': p0_values,
        'two_bit': {
            'input_entropy': [],       # Full tape entropy (x-axis)
            'output_entropy': [],      # Output entropy
            'Q_total': [],             # Total heat transfer
            'Q_h': [],                 # Heat from hot
            'Q_c': [],                 # Heat to cold
            'S_total_production': [],  # Total entropy production
            'phi': []
        },
        'single_bit': {
            'input_entropy': [],       # Full tape entropy (x-axis)
            'output_entropy': [],      # Output entropy
            'Q_total': [],             # Total heat transfer
            'Q_h': [],                 # Heat from hot
            'Q_c': [],                 # Heat to cold
            'S_total_production': [],  # Total entropy production
            'phi': []
        }
    }
    
    print(f"\nSweeping p0 from 0.0 to 0.5 ({n_points} points)...")
    print(f"Tape size: N={tape_size}, τ={tau}")
    
    for i, p0 in enumerate(p0_values):
        print(f"  [{i+1}/{len(p0_values)}] p0 = {p0:.3f}...", end=" ", flush=True)
        
        # Create tape with this p0
        tape_params = {'N': tape_size, 'p0': p0, 'init_mode': 'random', 'seed': seed_base + i}
        init_tape = TwoBitTape(**tape_params)
        
        # Get input entropy (multiply by N for full tape entropy)
        input_entropy_per_bit = init_tape.get_entropy()
        input_entropy_full = input_entropy_per_bit * tape_size
        
        # Run two-bit simulation with thermodynamics
        np.random.seed(seed_base + 1000 + i)
        two_bit_final_tape, two_bit_history, two_bit_stats, two_bit_thermo = run_simulation_with_thermodynamics(
            two_bit_demon_inst, init_tape.copy(), tau, phys_params, is_two_bit=True
        )
        
        # Get output entropy (multiply by N for full tape entropy)
        two_bit_output_entropy_per_bit = two_bit_final_tape.get_entropy()
        two_bit_output_entropy_full = two_bit_output_entropy_per_bit * tape_size
        
        # Get thermodynamic data
        two_bit_qh = two_bit_thermo['energy']['Q_h']
        two_bit_qc = two_bit_thermo['energy']['Q_c']
        two_bit_qtotal = two_bit_qh #+ two_bit_qh #two_bit_thermo['energy']['Q_total']
        two_bit_s_prod = two_bit_thermo['entropy']['S_total_production'] + (
            (two_bit_stats['outgoing']['entropy'] - two_bit_stats['incoming']['entropy']) * tape_size
        )
        two_bit_phi = two_bit_stats['phi']
        
        results['two_bit']['input_entropy'].append(input_entropy_full)
        results['two_bit']['output_entropy'].append(two_bit_output_entropy_full)
        results['two_bit']['Q_total'].append(two_bit_qtotal)
        results['two_bit']['Q_h'].append(two_bit_qh)
        results['two_bit']['Q_c'].append(two_bit_qc)
        results['two_bit']['S_total_production'].append(two_bit_s_prod)
        results['two_bit']['phi'].append(two_bit_phi)
        
        # Run single-bit simulation with thermodynamics (on fresh tape copy)
        np.random.seed(seed_base + 2000 + i)
        single_bit_tape = TwoBitTape(**tape_params)  # Fresh tape with same p0
        single_bit_final_tape, single_bit_history, single_bit_stats, single_bit_thermo = run_simulation_with_thermodynamics(
            single_bit_demon_inst, single_bit_tape, tau, phys_params, is_two_bit=False
        )
        
        # Get output entropy (multiply by N for full tape entropy)
        single_bit_output_entropy_per_bit = single_bit_final_tape.get_entropy()
        single_bit_output_entropy_full = single_bit_output_entropy_per_bit * tape_size
        
        # Get thermodynamic data
        single_bit_qtotal = single_bit_thermo['energy']['Q_total']
        single_bit_qh = single_bit_thermo['energy']['Q_h']
        single_bit_qc = single_bit_thermo['energy']['Q_c']
        single_bit_s_prod = single_bit_thermo['entropy']['S_total_production'] + (
            (single_bit_stats['outgoing']['entropy'] - single_bit_stats['incoming']['entropy']) * tape_size
        )
        single_bit_phi = single_bit_stats['phi']
        
        results['single_bit']['input_entropy'].append(input_entropy_full)
        results['single_bit']['output_entropy'].append(single_bit_output_entropy_full)
        results['single_bit']['Q_total'].append(single_bit_qtotal)
        results['single_bit']['Q_h'].append(single_bit_qh)
        results['single_bit']['Q_c'].append(single_bit_qc)
        results['single_bit']['S_total_production'].append(single_bit_s_prod)
        results['single_bit']['phi'].append(single_bit_phi)
        
        print(f"✓")
    
    # Create plots
    if plot:
        _plot_input_entropy_analysis(results, tape_size, tau, phys_params)
    
    return results


def _plot_input_entropy_analysis(results: Dict, tape_size: int, tau: float, phys_params: PhysParams):
    """Create separate plots for single-bit and two-bit demons using input entropy as x-axis."""
    
    input_entropy_2bit = results['two_bit']['input_entropy']
    input_entropy_1bit = results['single_bit']['input_entropy']
    
    # Two-Bit Demon Plot
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Input Entropy vs Output Entropy
    ax1 = axes1[0]
    ax1.plot(input_entropy_2bit, results['two_bit']['output_entropy'], 
             marker='o', linewidth=2, markersize=8, label='Output Entropy', color='steelblue')
    ax1.plot(input_entropy_2bit, input_entropy_2bit, 
             linestyle='--', linewidth=1.5, label='Input = Output', color='gray', alpha=0.7)
    
    ax1.set_xlabel('Input Entropy (S_in × N)', fontsize=12)
    ax1.set_ylabel('Output Entropy (S_out × N)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title('Two-Bit Demon: Input vs Output Entropy', fontsize=12, fontweight='bold')
    
    # Right plot: Input Entropy vs Q_total
    ax2 = axes1[1]
    ax2.plot(input_entropy_2bit, results['two_bit']['Q_total'], 
             marker='^', linewidth=2, markersize=8, label='Q_total', color='darkgreen')
    
    ax2.set_xlabel('Input Entropy (S_in × N)', fontsize=12)
    ax2.set_ylabel('Q_total (Heat Exchange)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title('Two-Bit Demon: Input Entropy vs Q_total', fontsize=12, fontweight='bold')
    
    fig1.suptitle(f'Two-Bit Demon - Input Entropy Analysis (N={tape_size}, τ={tau})', 
                  fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Single-Bit Demon Plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Input Entropy vs Output Entropy
    ax1 = axes2[0]
    ax1.plot(input_entropy_1bit, results['single_bit']['output_entropy'], 
             marker='o', linewidth=2, markersize=8, label='Output Entropy', color='steelblue')
    ax1.plot(input_entropy_1bit, input_entropy_1bit, 
             linestyle='--', linewidth=1.5, label='Input = Output', color='gray', alpha=0.7)
    
    ax1.set_xlabel('Input Entropy (S_in × N)', fontsize=12)
    ax1.set_ylabel('Output Entropy (S_out × N)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title('Single-Bit Demon: Input vs Output Entropy', fontsize=12, fontweight='bold')
    
    # Right plot: Input Entropy vs Q_total
    ax2 = axes2[1]
    ax2.plot(input_entropy_1bit, results['single_bit']['Q_total'], 
             marker='^', linewidth=2, markersize=8, label='Q_total', color='darkgreen')
    
    ax2.set_xlabel('Input Entropy (S_in × N)', fontsize=12)
    ax2.set_ylabel('Q_total (Heat Exchange)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title('Single-Bit Demon: Input Entropy vs Q_total', fontsize=12, fontweight='bold')
    
    fig2.suptitle(f'Single-Bit Demon - Input Entropy Analysis (N={tape_size}, τ={tau})', 
                  fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 10: Two-Bit Demon Simulation Demo")
    print("=" * 60)
    
    # Set up parameters
    phys_params = PhysParams(sigma=0.4, omega=0.6)
    
    # Demo 1: Basic comparison with random tape
    print("\n1. Basic comparison with random tape (p0=1.0)...")
    tape_params = {'N': 20000, 'p0': 1.0, 'init_mode': 'random'}
   # comparison = compare_demons(tape_params, phys_params, tau=50.0, seed=123, plot=True, title="Demon 2")
    tape_params = {'N': 20000, 'p0': 0.2, 'init_mode': 'random'}
    #comparison = compare_demons(tape_params, phys_params, tau=50.0, seed=123, plot=True, title="Demon 2")
    

    
    # Demo 2: Comparison with pair-controlled tape
    print("\n2. Comparison with controlled pair distribution (60% '00', 20% '11')...")
    tape_params = {
        'N': 20000, 
        'p0': 0.5,  # This won't be used much since we control pairs
        'init_mode': 'pair_distribution',
        'pair_00_frac': 0.5,
        'pair_11_frac': 0.5
    }
   # comparison2 = compare_demons(tape_params, phys_params, tau=50.0, seed=123, plot=True, title="Demon 2")
    # Demo 2: Comparison with pair-controlled tape
    print("\n2. Comparison with controlled pair distribution (60% '00', 20% '11')...")
    tape_params = {
        'N': 20000, 
        'p0': 0.5,  # This won't be used much since we control pairs
        'init_mode': 'pair_distribution',
        'pair_00_frac': 0.3,
        'pair_11_frac': 0.3
    }
    #comparison2 = compare_demons(tape_params, phys_params, tau=50.0, seed=123, plot=True,title="Demon 2")
    

    
    # Demo 3: Custom Two-Bit Demon
    print("\n3. Custom Two-Bit Demon with modified transitions...")
    custom_two_bit_demon = TwoBitDemon(phys_params=phys_params, init_state='d')
    custom_two_bit_demon.remove_cooperative_transition("u", "1","1")
    custom_two_bit_demon.add_cooperative_transition(d_from="d", b1_from="1", b2_from="1",
                                                    d_to="u", b1_to="1", b2_to="0")
    
    
    tape_params = {'N': 20000, 'p0': 1.0, 'init_mode': 'random'}
    #comparison = compare_demons(tape_params, phys_params, tau=50.0, seed=123, plot=True, two_bit_demon=custom_two_bit_demon, title="Demon 1")

    
    # Demo 2: Comparison with pair-controlled tape

    tape_params = {
        'N': 20000, 
        'p0': 0.5,  # This won't be used much since we control pairs
        'init_mode': 'pair_distribution',
        'pair_00_frac': 0.5,
        'pair_11_frac': 0.5
    }
    #comparison2 = compare_demons(tape_params, phys_params, tau=50.0, seed=123, plot=True, two_bit_demon=custom_two_bit_demon,title="Demon 1")
    # Demo 2: Comparison with pair-controlled tape

    tape_params = {
        'N': 20000, 
        'p0': 0.5,  # This won't be used much since we control pairs
        'init_mode': 'pair_distribution',
        'pair_00_frac': 0.3,
        'pair_11_frac': 0.3
    }
    comparison2 = compare_demons(tape_params, phys_params, tau=5.0, seed=123, plot=True, two_bit_demon=custom_two_bit_demon,title="Demon 1")
    tape_params = {
        'N': 20000, 
        'p0': 0.2,  # This won't be used much since we control pairs
        'init_mode': 'random',
    }
    #comparison2 = compare_demons(tape_params, phys_params, tau=50.0, seed=123, plot=True, two_bit_demon=custom_two_bit_demon,title="Demon 1")
    
    # Demo 4: Entropy analysis sweep over p0
    print("\n4. Entropy analysis: sweep over p0 range...")
    p0_values = np.linspace(0.0, 1.0, 11)  # 11 points from 0 to 1
    # results = sweep_p0_entropy_analysis(
    #     p0_values=p0_values,
    #     tape_size=10000,
    #     tau=50.0,
    #     phys_params=phys_params,
    #     seed_base=42,
    #     two_bit_demon=custom_two_bit_demon
    # )
    
    # Demo 5: Input entropy analysis (p0 from 0 to 0.5)
    print("\n5. Input entropy analysis: p0 from 0 to 0.5...")
    p0_values = np.linspace(0.5, 1.0, 20)  # 11 points from 0 to 0.5
    # results_entropy = sweep_p0_entropy_analysis_v2(
    #     tape_size=4000,
    #     tau=1.0,
    #     phys_params=phys_params,
    #     seed_base=100,
    #     two_bit_demon=custom_two_bit_demon,
    #     p0_values=p0_values
    # )
