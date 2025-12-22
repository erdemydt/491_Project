"""
Direct comparison between Phase 7 (Sequential) and Phase 8 (Competitive) models.

This requires both phase_7 and phase_8 to be available.
"""

import numpy as np
import matplotlib.pyplot as plt
from CompetingDemon import CompetingDemon, PhysParams
from Tape import Tape
from CompetingSimulation import CompetingDemonSimulation


def compare_models_single_run(K: int = 5, demon_n: int = 2, 
                              tape_params: dict = None,
                              tau: float = 1.0,
                              phys_params: PhysParams = None):
    """Run both models side-by-side and compare results.
    
    Note: This compares:
    - Phase 8 (Competitive): K demons compete, winner interacts for time τ
    - Phase 7 would be: K demons interact sequentially, each for time τ
    
    For fair comparison, we use same random seed for initial tape.
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    
    print("=" * 70)
    print(f"COMPARING MODELS: K={K} demons")
    print("=" * 70)
    
    # Phase 8: Competitive model
    print("\nRunning Phase 8 (Competitive)...")
    demons_competitive = [CompetingDemon(n=demon_n, phys_params=phys_params, 
                                        init_state='d0', demon_id=k) for k in range(K)]
    tape_competitive = Tape(N=tape_params["N"], p0=tape_params["p0"], seed=42)
    sim_competitive = CompetingDemonSimulation(demons=demons_competitive, 
                                              tape=tape_competitive, tau=tau)
    final_tape_comp, _, _, interaction_counts = sim_competitive.run_full_simulation()
    stats_comp = sim_competitive.compute_statistics(final_tape_comp)
    
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\nPhase 8 (Competitive - shortest dt wins):")
    print(f"  φ (bit flip fraction): {stats_comp['phi']:.4f}")
    print(f"  Bias out: {stats_comp['outgoing']['bias']:.4f}")
    print(f"  ΔS_B: {stats_comp['outgoing']['DeltaS_B']:+.4f}")
    print(f"  Q_c: {stats_comp['Q_c']:.4f}")
    print(f"  Interaction distribution:")
    for k in range(K):
        print(f"    Demon {k}: {interaction_counts[k]/tape_params['N']:.2%}")
    
    print("\n" + "=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Interaction distribution
    ax1 = axes[0]
    demon_labels = [f'D{k}' for k in range(K)]
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    
    interaction_fractions = interaction_counts / tape_params["N"]
    ax1.bar(demon_labels, interaction_fractions, color=colors, alpha=0.7)
    ax1.set_xlabel('Demon ID', fontsize=12)
    ax1.set_ylabel('Interaction Fraction', fontsize=12)
    ax1.set_title('Phase 8: Competitive Interaction Distribution', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1/K, color='red', linestyle='--', linewidth=1, 
               alpha=0.7, label=f'Uniform (1/{K})')
    ax1.legend()
    
    # Add value labels
    for i, frac in enumerate(interaction_fractions):
        ax1.text(i, frac, f'{frac:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Output statistics comparison
    ax2 = axes[1]
    metrics = ['φ', 'Bias\nOut', 'ΔS_B', 'Q_c']
    values_comp = [
        stats_comp['phi'],
        stats_comp['outgoing']['bias'],
        stats_comp['outgoing']['DeltaS_B'],
        stats_comp['Q_c']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x, values_comp, width, label='Phase 8 (Competitive)', 
           color='steelblue', alpha=0.7)
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Output Metrics Comparison', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(values_comp):
        ax2.text(i, v, f'{v:.3f}', ha='center', 
                va='bottom' if v > 0 else 'top', fontsize=9)
    
    fig.suptitle(f'Model Comparison: K={K}, n={demon_n}, τ={tau}, ' + 
                f'σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}',
                fontsize=13)
    plt.tight_layout()
    plt.show()


def compare_phi_vs_K_both_models(K_values: list, demon_n: int = 2,
                                 tape_params: dict = None,
                                 tau: float = 1.0,
                                 phys_params: PhysParams = None):
    """Compare φ vs K for both models.
    
    Note: For fair comparison with sequential model:
    - Phase 8: τ is interaction time per bit (winning demon gets τ)
    - Phase 7: τ is interaction time per demon (total time is K*τ)
    
    To make them comparable, we might want to use τ_phase7 = τ_phase8/K
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    
    phi_competitive = []
    
    for i, K in enumerate(K_values):
        print(f"Progress: {i+1}/{len(K_values)} - K={K}", end='\r')
        
        # Phase 8: Competitive
        demons = [CompetingDemon(n=demon_n, phys_params=phys_params, 
                                init_state='d0', demon_id=k) for k in range(K)]
        tape = Tape(N=tape_params["N"], p0=tape_params["p0"])
        sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=tau)
        final_tape, _, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        phi_competitive.append(stats['phi'])
    
    print()
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    plt.plot(K_values, phi_competitive, marker='o', linewidth=2.5, markersize=8,
            color='steelblue', label='Phase 8: Competitive (shortest dt wins)')
    
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, 
               label='φ = 0.5')
    
    plt.xlabel('Number of Demons (K)', fontsize=13)
    plt.ylabel('Bit Flip Fraction (φ)', fontsize=13)
    plt.title(f'φ vs K Comparison\n' + 
             f'n={demon_n}, τ={tau}, σ={phys_params.sigma:.3f}, ' + 
             f'ω={phys_params.omega:.3f}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}',
             fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    return phi_competitive


if __name__ == "__main__":
    # Single run comparison
    print("\nRunning single comparison (K=5)...")
    compare_models_single_run(
        K=5,
        demon_n=3,
        tape_params={"N": 5000, "p0": 1.0},
        tau=1.0,
        phys_params=PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    )
    
    # φ vs K comparison
    print("\nRunning φ vs K comparison...")
    compare_phi_vs_K_both_models(
        K_values=list(range(1, 16)),
        demon_n=3,
        tape_params={"N": 5000, "p0": 1.0},
        tau=1.0,
        phys_params=PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    )
