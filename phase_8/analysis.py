"""
Comparison analysis between competitive and sequential demon models.

This script helps visualize the differences between:
- Phase 8: Competitive demons (shortest dt wins)
- Phase 7: Sequential demons (all interact in order)
"""

import numpy as np
import matplotlib.pyplot as plt
from CompetingDemon import CompetingDemon, PhysParams
from Tape import Tape
from CompetingSimulation import CompetingDemonSimulation
from typing import List, Dict


def compare_phi_vs_K(K_values: List[int], tape_params: Dict = None,
                     demon_n: int = 3, tau: float = 1.0, 
                     phys_params: PhysParams = None,
                     n_trials: int = 5):
    """Compare phi vs K for competitive model with error bars.
    
    Args:
        K_values (List[int]): List of K values to test
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of states per demon
        tau (float): Interaction time per bit
        phys_params (PhysParams): Physical parameters
        n_trials (int): Number of trials for error estimation
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    
    phi_means = []
    phi_stds = []
    
    for i, K in enumerate(K_values):
        print(f"Progress: {i+1}/{len(K_values)} - K={K}", end='\r')
        
        phi_trials = []
        
        for trial in range(n_trials):
            # Create K demons
            demons = [CompetingDemon(n=demon_n, phys_params=phys_params, 
                                    init_state='d0', demon_id=k) for k in range(K)]
            
            # Create tape with different seed for each trial
            tape = Tape(N=tape_params["N"], p0=tape_params["p0"], seed=trial*100 + K)
            
            # Run simulation
            sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=tau)
            final_tape, _, _, _ = sim.run_full_simulation()
            
            # Compute statistics
            stats = sim.compute_statistics(final_tape)
            phi_trials.append(stats['phi'])
        
        phi_means.append(np.mean(phi_trials))
        phi_stds.append(np.std(phi_trials))
    
    print()  # New line after progress
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    phi_means = np.array(phi_means)
    phi_stds = np.array(phi_stds)
    
    plt.errorbar(K_values, phi_means, yerr=phi_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5,
                color='steelblue', label='Competitive Model (Phase 8)')
    
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='φ = 0.5')
    
    plt.xlabel('Number of Competing Demons (K)', fontsize=13)
    plt.ylabel('Bit Flip Fraction (φ)', fontsize=13)
    
    title = f'Competitive Demons: φ vs K (with {n_trials} trials per K)\n'
    title += f'n={demon_n}, τ={tau}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}, '
    title += f'N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}'
    
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    return phi_means, phi_stds


def analyze_interaction_distribution_vs_K(K_values: List[int], 
                                         tape_params: Dict = None,
                                         demon_n: int = 3, 
                                         tau: float = 1.0,
                                         phys_params: PhysParams = None):
    """Analyze how interaction distribution changes with K.
    
    Shows which demons win more often as K increases.
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    
    # Store interaction fractions for each K
    all_fractions = []
    
    for i, K in enumerate(K_values):
        print(f"Progress: {i+1}/{len(K_values)} - K={K}", end='\r')
        
        # Create K demons
        demons = [CompetingDemon(n=demon_n, phys_params=phys_params, 
                                init_state='d0', demon_id=k) for k in range(K)]
        
        # Create tape
        tape = Tape(N=tape_params["N"], p0=tape_params["p0"])
        
        # Run simulation
        sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=tau)
        final_tape, _, _, interaction_counts = sim.run_full_simulation()
        
        # Get interaction fractions
        fractions = interaction_counts / tape_params["N"]
        all_fractions.append(fractions)
    
    print()
    
    # Plot as heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap showing distribution
    max_K = max(K_values)
    heatmap_data = np.zeros((len(K_values), max_K))
    
    for i, (K, fractions) in enumerate(zip(K_values, all_fractions)):
        heatmap_data[i, :K] = fractions
    
    im = ax1.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel('Demon Index', fontsize=12)
    ax1.set_ylabel('K (Number of Demons)', fontsize=12)
    ax1.set_yticks(range(len(K_values)))
    ax1.set_yticklabels(K_values)
    ax1.set_title('Interaction Fraction Heatmap', fontsize=13)
    plt.colorbar(im, ax=ax1, label='Interaction Fraction')
    
    # Line plot showing distributions
    colors = plt.cm.plasma(np.linspace(0, 1, len(K_values)))
    for i, (K, fractions) in enumerate(zip(K_values, all_fractions)):
        ax2.plot(range(K), fractions, marker='o', linewidth=2, 
                color=colors[i], label=f'K={K}', alpha=0.7)
    
    ax2.set_xlabel('Demon Index', fontsize=12)
    ax2.set_ylabel('Interaction Fraction', fontsize=12)
    ax2.set_title('Interaction Distribution by K', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)
    
    fig.suptitle(f'Interaction Distribution Analysis\n' + 
                f'n={demon_n}, τ={tau}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}',
                fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return all_fractions


def analyze_demon_state_evolution(K: int = 5, demon_n: int = 3,
                                  tape_params: Dict = None,
                                  tau: float = 1.0,
                                  phys_params: PhysParams = None):
    """Analyze how demon states evolve throughout the simulation.
    
    Shows the state trajectory for each demon over the tape.
    """
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    
    # Create K demons
    demons = [CompetingDemon(n=demon_n, phys_params=phys_params, 
                            init_state='d0', demon_id=k) for k in range(K)]
    
    # Create tape
    tape = Tape(N=tape_params["N"], p0=tape_params["p0"])
    
    # Run simulation
    sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=tau)
    final_tape, _, demon_states_history, interaction_counts = sim.run_full_simulation()
    
    # Convert state strings to indices for plotting
    state_trajectories = []
    for k in range(K):
        states = demon_states_history[k]
        indices = [int(s[1:]) for s in states]  # Extract number from 'dX'
        state_trajectories.append(indices)
    
    # Create figure with subplots
    fig, axes = plt.subplots(K, 1, figsize=(14, 2*K), sharex=True)
    if K == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for k in range(K):
        axes[k].plot(state_trajectories[k], linewidth=1.5, color=colors[k], alpha=0.7)
        axes[k].set_ylabel(f'Demon {k}\nState', fontsize=11)
        axes[k].set_yticks(range(demon_n))
        axes[k].set_yticklabels([f'd{i}' for i in range(demon_n)])
        axes[k].grid(True, alpha=0.3)
        axes[k].set_ylim(-0.5, demon_n - 0.5)
        
        # Add interaction count in corner
        axes[k].text(0.98, 0.95, f'Wins: {interaction_counts[k]} ({interaction_counts[k]/tape_params["N"]:.1%})',
                    transform=axes[k].transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Bit Position', fontsize=12)
    
    fig.suptitle(f'Demon State Evolution Over Tape\n' + 
                f'K={K}, n={demon_n}, τ={tau}, N={tape_params["N"]}',
                fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    """Run comparison analyses."""
    print("=" * 70)
    print("COMPETITIVE DEMONS: ANALYSIS SUITE")
    print("=" * 70)
    
    # Common parameters
    phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    tape_params = {"N": 5000, "p0": 1.0}
    demon_n = 3
    tau = 1.0
    
    print("\nPhysical Parameters:")
    print(f"  ΔE: {phys_params.DeltaE}")
    print(f"  Th: {phys_params.Th}, Tc: {phys_params.Tc}")
    print(f"  σ: {phys_params.sigma:.4f}, ω: {phys_params.omega:.4f}")
    
    print("\nSimulation Parameters:")
    print(f"  Demon states (n): {demon_n}")
    print(f"  Interaction time (τ): {tau}")
    print(f"  Tape length (N): {tape_params['N']}")
    
    # Analysis 1: φ vs K with error bars
    print("\n" + "=" * 70)
    print("Analysis 1: φ vs K with statistical error")
    print("=" * 70)
    K_values = list(range(1, 11))
    compare_phi_vs_K(K_values, tape_params, demon_n, tau, phys_params, n_trials=3)
    
    # Analysis 2: Interaction distribution vs K
    print("\n" + "=" * 70)
    print("Analysis 2: Interaction distribution vs K")
    print("=" * 70)
    K_values = [2, 3, 5, 7, 10]
    analyze_interaction_distribution_vs_K(K_values, tape_params, demon_n, tau, phys_params)
    
    # Analysis 3: Demon state evolution
    print("\n" + "=" * 70)
    print("Analysis 3: Demon state evolution over tape")
    print("=" * 70)
    analyze_demon_state_evolution(K=5, demon_n=3, 
                                  tape_params={"N": 1000, "p0": 1.0},
                                  tau=tau, phys_params=phys_params)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
