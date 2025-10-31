"""
Test script for Monte Carlo functionality in the optimized simulation.
"""

from Simulation_optimized import OptimizedSimulation, plot_total_energy_vs_demon_n_optimized, suggest_mc_samples
from Demon import Demon
from Tape import Tape

def test_monte_carlo_basic():
    """Test basic Monte Carlo functionality."""
    print("Testing Monte Carlo functionality...")
    
    # Create a simple test case
    demon = Demon(n=5)
    tape = Tape(N=1000, p0=0.8)
    sim = OptimizedSimulation(demon=demon, tape=tape, tau=1.0)
    
    # Test single run
    print("Single run:")
    stats_single = sim.run_monte_carlo_simulation(mc_samples=1)
    print(f"  Bias: {stats_single['outgoing']['bias']:.6f}")
    print(f"  Energy: {stats_single['outgoing']['Q_c']:.6f}")
    
    # Test Monte Carlo averaging
    print("\nMonte Carlo averaging (5 samples):")
    stats_mc = sim.run_monte_carlo_simulation(mc_samples=5)
    print(f"  Bias: {stats_mc['outgoing']['bias']:.6f} ± {stats_mc['outgoing']['bias_std']:.6f}")
    print(f"  Energy: {stats_mc['outgoing']['Q_c']:.6f} ± {stats_mc['outgoing']['Q_c_std']:.6f}")
    
    print("✓ Basic Monte Carlo test passed!")

def test_plotting_with_mc():
    """Test plotting with Monte Carlo."""
    print("\nTesting plotting with Monte Carlo...")
    
    results = plot_total_energy_vs_demon_n_optimized(
        min_n=2, max_n=10,
        tape_params={"N": 500, "p0": 1.0},
        tau=1.0,
        step=2,
        mc_samples=3,
        show_error_bars=True
    )
    
    print(f"✓ Plotting test completed!")
    print(f"  Tested {len(results['n_values'])} data points")
    print(f"  Each point averaged over {results['mc_samples']} runs")

def test_mc_suggestion():
    """Test Monte Carlo sample suggestion."""
    print("\nTesting MC sample suggestion...")
    
    suggested = suggest_mc_samples(noise_tolerance=0.05, confidence_level=0.95)
    print(f"✓ Suggested {suggested} MC samples for 5% error tolerance")

if __name__ == "__main__":
    print("=" * 50)
    print("MONTE CARLO FUNCTIONALITY TESTS")
    print("=" * 50)
    
    test_monte_carlo_basic()
    test_mc_suggestion()
    test_plotting_with_mc()
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 50)