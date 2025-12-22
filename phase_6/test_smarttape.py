"""
Demonstration of SmartTape class with different correlation types.
Tests correlation preservation through demon simulation.
"""

from Tape import SmartTape, Tape
from Demon import Demon, PhysParams
from Simulation import StackedDemonSimulation
import numpy as np
import matplotlib.pyplot as plt


def demo_correlation_types():
    """Demonstrate different correlation types available in SmartTape."""
    N = 1000
    p0 = 0.7
    
    print("=" * 60)
    print("Demonstrating Different Correlation Types")
    print("=" * 60)
    
    # 1. No correlation (baseline)
    print("\n1. No Correlation (Independent bits)")
    tape_none = SmartTape(N=N, p0=p0, correlation_type='none', seed=42)
    summary_none = tape_none.get_correlation_summary()
    print(f"   Nearest neighbor correlation: {summary_none['nearest_neighbor_correlation']:.4f}")
    print(f"   Mean block length: {summary_none['mean_block_length']:.2f}")
    
    # 2. Markov correlation
    print("\n2. Markov Chain Correlation")
    for strength in [0.3, 0.6, 0.9]:
        tape_markov = SmartTape(N=N, p0=p0, correlation_type='markov', 
                               correlation_strength=strength, seed=42)
        summary = tape_markov.get_correlation_summary()
        print(f"   Strength={strength:.1f}: NN_corr={summary['nearest_neighbor_correlation']:.4f}, "
              f"Block_len={summary['mean_block_length']:.2f}")
    
    # 3. Block correlation
    print("\n3. Block Correlation")
    for strength in [0.3, 0.6, 0.9]:
        tape_block = SmartTape(N=N, p0=p0, correlation_type='block', 
                              correlation_strength=strength, seed=42)
        summary = tape_block.get_correlation_summary()
        print(f"   Strength={strength:.1f}: NN_corr={summary['nearest_neighbor_correlation']:.4f}, "
              f"Block_len={summary['mean_block_length']:.2f}")
    
    # 4. Periodic correlation
    print("\n4. Periodic Correlation")
    for strength in [0.3, 0.6, 0.9]:
        tape_periodic = SmartTape(N=N, p0=p0, correlation_type='periodic', 
                                 correlation_strength=strength, period=10, seed=42)
        summary = tape_periodic.get_correlation_summary()
        print(f"   Strength={strength:.1f}: NN_corr={summary['nearest_neighbor_correlation']:.4f}, "
              f"MI={summary['mutual_information_lag1']:.4f}")


def visualize_correlation_types():
    """Create visualization comparing different correlation types."""
    N = 1000
    p0 = 0.7
    strength = 0.7
    
    # Create tapes with different correlations
    tapes = {
        'None': SmartTape(N=N, p0=p0, correlation_type='none', seed=42),
        'Markov': SmartTape(N=N, p0=p0, correlation_type='markov', 
                           correlation_strength=strength, seed=42),
        'Block': SmartTape(N=N, p0=p0, correlation_type='block', 
                          correlation_strength=strength, seed=42),
        'Periodic': SmartTape(N=N, p0=p0, correlation_type='periodic', 
                             correlation_strength=strength, period=20, seed=42)
    }
    
    fig, axes = plt.subplots(len(tapes), 2, figsize=(14, 10))
    
    for idx, (name, tape) in enumerate(tapes.items()):
        # Plot tape visualization
        ax = axes[idx, 0]
        float_tape = np.array(tape.tape_arr, dtype=float)
        display_len = min(200, N)
        ax.imshow(float_tape[:display_len].reshape(1, -1), cmap='binary', aspect='auto')
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_title('Tape Pattern (first 200 bits)')
        if idx == len(tapes) - 1:
            ax.set_xlabel('Bit Position')
        
        # Plot autocorrelation
        ax = axes[idx, 1]
        lags, autocorr = tape.compute_autocorrelation(max_lag=50)
        ax.plot(lags, autocorr, marker='o', markersize=3, color='steelblue')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.2, 1.0)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_title('Autocorrelation Function')
        if idx == len(tapes) - 1:
            ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorr')
        
        # Add correlation value as text
        nn_corr = tape.compute_nearest_neighbor_correlation()
        ax.text(0.7, 0.85, f'NN corr: {nn_corr:.3f}', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('phase_6/plots/correlation_types_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to: phase_6/plots/correlation_types_comparison.png")
    plt.show()


def test_simulation_with_correlations():
    """Test how demon simulation affects differently correlated tapes."""
    N = 5000
    p0 = 1.0  # Start with all 0s
    correlation_strengths = [0.0, 0.3, 0.6, 0.9]
    
    # Setup demon
    phys_params = PhysParams(sigma=0.3, omega=0.8, DeltaE=1.0, gamma=1.0)
    demon = Demon(n=3, phys_params=phys_params, init_state='d0')
    tau = 1.0
    
    print("\n" + "=" * 60)
    print("Testing Demon Simulation with Correlated Tapes")
    print("=" * 60)
    print(f"Tape: N={N}, p0={p0}")
    print(f"Demon: n=3, tau={tau}")
    print(f"Physics: σ={phys_params.sigma}, ω={phys_params.omega}")
    print("=" * 60)
    
    results = []
    
    for strength in correlation_strengths:
        print(f"\nMarkov correlation strength: {strength:.1f}")
        
        # Create correlated tape
        tape = SmartTape(N=N, p0=p0, correlation_type='markov', 
                        correlation_strength=strength, seed=42)
        
        # Get initial correlations
        initial_summary = tape.get_correlation_summary()
        print(f"  Initial NN correlation: {initial_summary['nearest_neighbor_correlation']:.4f}")
        print(f"  Initial mean block length: {initial_summary['mean_block_length']:.2f}")
        
        # Run simulation
        sim = StackedDemonSimulation(demons=[demon], tape=tape, tau=tau)
        final_tape_obj, _, _ = sim.run_full_simulation()
        
        # Convert final tape to SmartTape for analysis
        final_smart_tape = SmartTape(N=N, p0=p0, tape_arr=final_tape_obj.tape_arr)
        
        # Get final correlations
        final_summary = final_smart_tape.get_correlation_summary()
        print(f"  Final NN correlation: {final_summary['nearest_neighbor_correlation']:.4f}")
        print(f"  Final mean block length: {final_summary['mean_block_length']:.2f}")
        
        # Compute statistics
        stats = sim.compute_statistics(final_tape_obj)
        print(f"  Phi (bit flip fraction): {stats['phi']:.4f}")
        print(f"  Final p0: {stats['outgoing']['p0']:.4f}")
        print(f"  Energy to cold (Q_c): {stats['Q_c']:.4f}")
        
        results.append({
            'strength': strength,
            'initial_nn_corr': initial_summary['nearest_neighbor_correlation'],
            'final_nn_corr': final_summary['nearest_neighbor_correlation'],
            'initial_block_len': initial_summary['mean_block_length'],
            'final_block_len': final_summary['mean_block_length'],
            'phi': stats['phi'],
            'final_p0': stats['outgoing']['p0'],
            'Q_c': stats['Q_c']
        })
    
    return results


def plot_correlation_effects(results):
    """Plot how correlations affect simulation outcomes."""
    strengths = [r['strength'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: NN correlation before/after
    ax = axes[0, 0]
    ax.plot(strengths, [r['initial_nn_corr'] for r in results], 
            marker='o', label='Initial', linewidth=2)
    ax.plot(strengths, [r['final_nn_corr'] for r in results], 
            marker='s', label='Final', linewidth=2)
    ax.set_xlabel('Initial Correlation Strength')
    ax.set_ylabel('Nearest Neighbor Correlation')
    ax.set_title('Correlation Preservation Through Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Block length before/after
    ax = axes[0, 1]
    ax.plot(strengths, [r['initial_block_len'] for r in results], 
            marker='o', label='Initial', linewidth=2)
    ax.plot(strengths, [r['final_block_len'] for r in results], 
            marker='s', label='Final', linewidth=2)
    ax.set_xlabel('Initial Correlation Strength')
    ax.set_ylabel('Mean Block Length')
    ax.set_title('Block Structure Through Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Phi vs initial correlation
    ax = axes[1, 0]
    ax.plot(strengths, [r['phi'] for r in results], 
            marker='o', linewidth=2, color='purple')
    ax.set_xlabel('Initial Correlation Strength')
    ax.set_ylabel('Phi (Bit Flip Fraction)')
    ax.set_title('Effect of Correlation on Bit Flips')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Q_c vs initial correlation
    ax = axes[1, 1]
    ax.plot(strengths, [r['Q_c'] for r in results], 
            marker='o', linewidth=2, color='green')
    ax.set_xlabel('Initial Correlation Strength')
    ax.set_ylabel('Energy to Cold (Q_c)')
    ax.set_title('Effect of Correlation on Energy Transfer')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_6/plots/correlation_effects.png', dpi=150, bbox_inches='tight')
    print("\nSaved effects plot to: phase_6/plots/correlation_effects.png")
    plt.show()


if __name__ == "__main__":
    # Demo 1: Show different correlation types
    demo_correlation_types()
    
    # Demo 2: Visualize correlation patterns
    visualize_correlation_types()
    
    # Demo 3: Test simulation with correlations
    results = test_simulation_with_correlations()
    
    # Demo 4: Plot effects
    plot_correlation_effects(results)
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
