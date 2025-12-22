"""
Sweep over different input tape correlations and analyze:
1. Bit flip fraction (phi)
2. Output tape correlations
3. Energy transfer (Q_c)
4. How correlations change through the simulation
"""

from Tape import SmartTape
from Demon import Demon, PhysParams
from Simulation import StackedDemonSimulation
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def run_correlation_sweep(
    correlation_type: str = 'markov',
    correlation_strengths: List[float] = None,
    N: int = 5000,
    p0: float = 1.0,
    demon_n: int = 3,
    K: int = 1,
    tau: float = 1.0,
    phys_params: PhysParams = None,
    seed: int = 42
) -> List[Dict]:
    """
    Run simulations with different input tape correlations.
    
    Args:
        correlation_type: Type of correlation ('markov', 'block', 'periodic')
        correlation_strengths: List of correlation strengths to test
        N: Number of bits on tape
        p0: Initial probability of bit being 0
        demon_n: Number of demon states
        K: Number of stacked demons
        tau: Interaction time per demon
        phys_params: Physical parameters
        seed: Random seed
    
    Returns:
        List of result dictionaries containing all metrics
    """
    if correlation_strengths is None:
        correlation_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8, DeltaE=1.0, gamma=1.0)
    
    results = []
    
    print("=" * 70)
    print(f"Correlation Sweep: {correlation_type}")
    print("=" * 70)
    print(f"Tape: N={N}, p0={p0}")
    print(f"Demon: n={demon_n}, K={K}, tau={tau}")
    print(f"Physics: σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}")
    print("=" * 70)
    
    for i, strength in enumerate(correlation_strengths):
        print(f"\n[{i+1}/{len(correlation_strengths)}] Correlation strength: {strength:.2f}")
        
        # Create correlated input tape
        input_tape = SmartTape(
            N=N, 
            p0=p0, 
            correlation_type=correlation_type,
            correlation_strength=strength,
            seed=seed
        )
        
        # Analyze input correlations
        input_summary = input_tape.get_correlation_summary()
        
        print(f"  Input tape:")
        print(f"    NN correlation: {input_summary['nearest_neighbor_correlation']:.4f}")
        print(f"    Mean block length: {input_summary['mean_block_length']:.2f}")
        print(f"    Mutual information: {input_summary['mutual_information_lag1']:.4f}")
        
        # Create demons
        demons = [Demon(n=demon_n, phys_params=phys_params, init_state='d0') 
                 for _ in range(K)]
        
        # Run simulation
        sim = StackedDemonSimulation(demons=demons, tape=input_tape, tau=tau)
        final_tape_obj, _, _ = sim.run_full_simulation()
        
        # Convert output to SmartTape for correlation analysis
        output_tape = SmartTape(N=N, p0=p0, tape_arr=final_tape_obj.tape_arr)
        
        # Analyze output correlations
        output_summary = output_tape.get_correlation_summary()
        
        print(f"  Output tape:")
        print(f"    NN correlation: {output_summary['nearest_neighbor_correlation']:.4f}")
        print(f"    Mean block length: {output_summary['mean_block_length']:.2f}")
        print(f"    Mutual information: {output_summary['mutual_information_lag1']:.4f}")
        
        # Compute simulation statistics
        stats = sim.compute_statistics(final_tape_obj)
        
        print(f"  Simulation:")
        print(f"    Phi (bit flip fraction): {stats['phi']:.4f}")
        print(f"    Final p0: {stats['outgoing']['p0']:.4f}")
        print(f"    Q_c (energy to cold): {stats['Q_c']:.4f}")
        print(f"    Delta S_B: {stats['outgoing']['DeltaS_B']:.4f}")
        
        # Store results
        results.append({
            'correlation_strength': strength,
            # Input metrics
            'input_nn_corr': input_summary['nearest_neighbor_correlation'],
            'input_autocorr_lag1': input_summary['autocorr_lag1'],
            'input_block_len': input_summary['mean_block_length'],
            'input_mi': input_summary['mutual_information_lag1'],
            'input_entropy': input_tape.get_entropy(),
            # Output metrics
            'output_nn_corr': output_summary['nearest_neighbor_correlation'],
            'output_autocorr_lag1': output_summary['autocorr_lag1'],
            'output_block_len': output_summary['mean_block_length'],
            'output_mi': output_summary['mutual_information_lag1'],
            'output_entropy': output_tape.get_entropy(),
            # Simulation metrics
            'input_p0': stats['incoming']['p0'],
            'phi': stats['phi'],
            'final_p0': stats['outgoing']['p0'],
            'Q_c': stats['Q_c'],
            'delta_S_b': stats['outgoing']['DeltaS_B'],
            # Correlation changes
            'delta_nn_corr': output_summary['nearest_neighbor_correlation'] - input_summary['nearest_neighbor_correlation'],
            'delta_block_len': output_summary['mean_block_length'] - input_summary['mean_block_length'],
            'delta_mi': output_summary['mutual_information_lag1'] - input_summary['mutual_information_lag1'],
        })
    
    print("\n" + "=" * 70)
    print("Sweep complete!")
    print("=" * 70)
    
    return results


def plot_correlation_sweep_results(results: List[Dict], correlation_type: str = 'markov',
                                   save_path: str = None):
    """
    Create comprehensive plots of correlation sweep results.
    
    Args:
        results: List of result dictionaries from run_correlation_sweep
        correlation_type: Type of correlation used
        save_path: Optional path to save the figure
    """
    strengths = [r['correlation_strength'] for r in results]
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Nearest Neighbor Correlation (Input vs Output)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(strengths, [r['input_nn_corr'] for r in results], 
            marker='o', label='Input', linewidth=2, markersize=6, color='steelblue')
    ax.plot(strengths, [r['output_nn_corr'] for r in results], 
            marker='s', label='Output', linewidth=2, markersize=6, color='coral')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Nearest Neighbor Correlation', fontsize=11)
    ax.set_title('NN Correlation: Input vs Output', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Change in NN Correlation
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(strengths, [r['delta_nn_corr'] for r in results], 
            marker='o', linewidth=2, markersize=6, color='purple')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Δ NN Correlation', fontsize=11)
    ax.set_title('Change in NN Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Phi vs Input Correlation
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(strengths, [r['phi'] for r in results], 
            marker='o', linewidth=2, markersize=6, color='green')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='φ = 0.5')
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Phi (Bit Flip Fraction)', fontsize=11)
    ax.set_title('Bit Flip Fraction vs Correlation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Block Length (Input vs Output)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(strengths, [r['input_block_len'] for r in results], 
            marker='o', label='Input', linewidth=2, markersize=6, color='steelblue')
    ax.plot(strengths, [r['output_block_len'] for r in results], 
            marker='s', label='Output', linewidth=2, markersize=6, color='coral')
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Mean Block Length', fontsize=11)
    ax.set_title('Block Length: Input vs Output', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Mutual Information (Input vs Output)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(strengths, [r['input_mi'] for r in results], 
            marker='o', label='Input', linewidth=2, markersize=6, color='steelblue')
    ax.plot(strengths, [r['output_mi'] for r in results], 
            marker='s', label='Output', linewidth=2, markersize=6, color='coral')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Mutual Information (lag=1)', fontsize=11)
    ax.set_title('Mutual Information: Input vs Output', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Q_c vs Input Correlation
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(strengths, [r['Q_c'] for r in results], 
            marker='o', linewidth=2, markersize=6, color='darkgreen')
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Energy to Cold (Q_c)', fontsize=11)
    ax.set_title('Energy Transfer vs Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Entropy (Input vs Output)
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(strengths, [r['input_entropy'] for r in results], 
            marker='o', label='Input', linewidth=2, markersize=6, color='steelblue')
    ax.plot(strengths, [r['output_entropy'] for r in results], 
            marker='s', label='Output', linewidth=2, markersize=6, color='coral')
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Entropy', fontsize=11)
    ax.set_title('Tape Entropy: Input vs Output', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Final p0 vs Input Correlation
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(strengths, [r['final_p0'] for r in results], 
            marker='o', linewidth=2, markersize=6, color='teal')
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('Final p₀', fontsize=11)
    ax.set_title('Output Bit Distribution vs Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Delta S_B vs Input Correlation
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(strengths, [r['delta_S_b'] for r in results], 
            marker='o', linewidth=2, markersize=6, color='darkred')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Input Correlation Strength', fontsize=11)
    ax.set_ylabel('ΔS_B (Entropy Change)', fontsize=11)
    ax.set_title('Entropy Change vs Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Correlation Sweep Analysis: {correlation_type.upper()} Correlation, p0={results[0]["input_p0"]}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")
    
    plt.show()


def compare_correlation_types(
    correlation_types: List[str] = None,
    correlation_strength: float = 0.7,
    N: int = 5000,
    p0: float = 1.0,
    demon_n: int = 3,
    tau: float = 1.0,
    phys_params: PhysParams = None
):
    """
    Compare how different correlation types affect simulation outcomes.
    
    Args:
        correlation_types: List of correlation types to compare
        correlation_strength: Fixed correlation strength for all types
        Other parameters same as run_correlation_sweep
    """
    if correlation_types is None:
        correlation_types = ['none', 'markov', 'block', 'periodic']
    
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8, DeltaE=1.0, gamma=1.0)
    
    results = {}
    
    print("\n" + "=" * 70)
    print(f"Comparing Correlation Types (strength = {correlation_strength:.2f})")
    print("=" * 70)
    
    for corr_type in correlation_types:
        print(f"\nTesting {corr_type} correlation...")
        
        # Special handling for 'none' type
        strength = 0.0 if corr_type == 'none' else correlation_strength
        
        # Create input tape
        input_tape = SmartTape(
            N=N, p0=p0, 
            correlation_type=corr_type,
            correlation_strength=strength,
            seed=42
        )
        
        # Analyze input
        input_summary = input_tape.get_correlation_summary()
        
        # Run simulation
        demon = Demon(n=demon_n, phys_params=phys_params, init_state='d0')
        sim = StackedDemonSimulation(demons=[demon], tape=input_tape, tau=tau)
        final_tape_obj, _, _ = sim.run_full_simulation()
        
        # Analyze output
        output_tape = SmartTape(N=N, p0=p0, tape_arr=final_tape_obj.tape_arr)
        output_summary = output_tape.get_correlation_summary()
        
        # Get stats
        stats = sim.compute_statistics(final_tape_obj)
        
        results[corr_type] = {
            'input': input_summary,
            'output': output_summary,
            'stats': stats
        }
        
        print(f"  Input NN corr: {input_summary['nearest_neighbor_correlation']:.4f}")
        print(f"  Output NN corr: {output_summary['nearest_neighbor_correlation']:.4f}")
        print(f"  Phi: {stats['phi']:.4f}")
        print(f"  Q_c: {stats['Q_c']:.4f}")
    
    # Plot comparison
    plot_correlation_type_comparison(results, correlation_strength)
    
    return results


def plot_correlation_type_comparison(results: Dict, correlation_strength: float):
    """Plot comparison of different correlation types."""
    types = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: NN Correlation
    ax = axes[0, 0]
    input_nn = [results[t]['input']['nearest_neighbor_correlation'] for t in types]
    output_nn = [results[t]['output']['nearest_neighbor_correlation'] for t in types]
    x = np.arange(len(types))
    width = 0.35
    ax.bar(x - width/2, input_nn, width, label='Input', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, output_nn, width, label='Output', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel('NN Correlation')
    ax.set_title('Nearest Neighbor Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Block Length
    ax = axes[0, 1]
    input_bl = [results[t]['input']['mean_block_length'] for t in types]
    output_bl = [results[t]['output']['mean_block_length'] for t in types]
    ax.bar(x - width/2, input_bl, width, label='Input', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, output_bl, width, label='Output', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel('Mean Block Length')
    ax.set_title('Mean Block Length')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Phi
    ax = axes[0, 2]
    phi_values = [results[t]['stats']['phi'] for t in types]
    ax.bar(types, phi_values, alpha=0.8, color='green')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Phi (Bit Flip Fraction)')
    ax.set_title('Bit Flip Fraction')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Mutual Information
    ax = axes[1, 0]
    input_mi = [results[t]['input']['mutual_information_lag1'] for t in types]
    output_mi = [results[t]['output']['mutual_information_lag1'] for t in types]
    ax.bar(x - width/2, input_mi, width, label='Input', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, output_mi, width, label='Output', alpha=0.8, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel('Mutual Information')
    ax.set_title('Mutual Information (lag=1)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Q_c
    ax = axes[1, 1]
    qc_values = [results[t]['stats']['Q_c'] for t in types]
    ax.bar(types, qc_values, alpha=0.8, color='darkgreen')
    ax.set_ylabel('Energy to Cold (Q_c)')
    ax.set_title('Energy Transfer')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Delta S_B
    ax = axes[1, 2]
    dsb_values = [results[t]['stats']['outgoing']['DeltaS_B'] for t in types]
    ax.bar(types, dsb_values, alpha=0.8, color='darkred')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('ΔS_B')
    ax.set_title('Entropy Change')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Correlation Type Comparison (strength = {correlation_strength:.2f})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('phase_6/plots/correlation_types_comparison_sim.png', dpi=150, bbox_inches='tight')
    print("\nSaved to: phase_6/plots/correlation_types_comparison_sim.png")
    plt.show()


if __name__ == "__main__":
    # Example 1: Sweep over Markov correlation strengths
    print("\n" + "="*70)
    print("EXAMPLE 1: Markov Correlation Sweep")
    print("="*70)
    results_markov = run_correlation_sweep(
        correlation_type='markov',
        correlation_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
        N=5000,
        p0=.7,
        demon_n=30,
        K=1,
        tau=10.0,
        phys_params=PhysParams(sigma=0.45, omega=0.55, DeltaE=2.0, gamma=1.0),
        seed=42
    )
    
    plot_correlation_sweep_results(
        results_markov, 
        correlation_type='markov',
        save_path='phase_6/plots/markov_correlation_sweep.png'
    )
    
    # Example 2: Sweep over Block correlation strengths
    print("\n" + "="*70)
    print("EXAMPLE 2: Block Correlation Sweep")
    print("="*70)
    results_block = run_correlation_sweep(
        correlation_type='block',
        correlation_strengths=[0.0, 0.2, 0.4, 0.6, 0.8],
        N=5000,
        p0=.7,
        demon_n=30,
        tau=10.0,
        seed=42
    )
    
    plot_correlation_sweep_results(
        results_block,
        correlation_type='block',
        save_path='phase_6/plots/block_correlation_sweep.png'
    )
    
    # Example 3: Compare different correlation types
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Correlation Types")
    print("="*70)
    results_comparison = compare_correlation_types(
        correlation_types=['none', 'markov', 'block', 'periodic'],
        correlation_strength=0.7,
        N=5000,
        p0=.7,
        demon_n=30,
        tau=10.0
    )
    
    print("\n" + "="*70)
    print("All analyses complete!")
    print("="*70)
