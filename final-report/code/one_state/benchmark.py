from Demon import Demon, PhysParams
from Tape import Tape
from Simulation import Simulation
from Simulation_optimized import OptimizedSimulation
import time
import numpy as np

def quick_benchmark():
    """Quick benchmark to demonstrate performance improvements."""
    
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Test parameters (smaller scale for quick demo)
    test_params = {
        "min_n": 2,
        "max_n": 20,  # Small range for quick test
        "tape_params": {"N": 1000, "p0": 1.0},  # Smaller tape
        "tau": 1.0
    }
    
    print(f"Test parameters: n={test_params['min_n']} to {test_params['max_n']}")
    print(f"Tape size: {test_params['tape_params']['N']} bits")
    print(f"tau: {test_params['tau']}")
    print("-" * 60)
    
    # Original version
    print("Testing ORIGINAL version...")
    start_time = time.time()
    
    original_results = []
    n_values = list(range(test_params['min_n'], test_params['max_n'] + 1))
    
    for n in n_values:
        demon = Demon(n=n)
        tape = Tape(**test_params['tape_params'])
        sim = Simulation(demon=demon, tape=tape, tau=test_params['tau'])
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        original_results.append(stats["outgoing"]["Q_c"])
    
    original_time = time.time() - start_time
    
    # Optimized version
    print("Testing OPTIMIZED version...")
    start_time = time.time()
    
    optimized_results = []
    base_tape = Tape(**test_params['tape_params'])
    
    for n in n_values:
        demon = Demon(n=n)
        sim = OptimizedSimulation(demon=demon, tape=base_tape, tau=test_params['tau'])
        final_tape, _, _ = sim.run_full_simulation_optimized()
        stats = sim.compute_statistics_optimized(final_tape)
        optimized_results.append(stats["outgoing"]["Q_c"])
    
    optimized_time = time.time() - start_time
    
    # Results comparison
    speedup = original_time / optimized_time
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Original time:   {original_time:.3f} seconds")
    print(f"Optimized time:  {optimized_time:.3f} seconds")
    print(f"Speedup:         {speedup:.1f}x faster")
    print(f"Time saved:      {original_time - optimized_time:.3f} seconds")
    print(f"Efficiency gain: {(1 - optimized_time/original_time)*100:.1f}%")
    
    # Accuracy check
    max_diff = np.max(np.abs(np.array(original_results) - np.array(optimized_results)))
    print(f"Max difference:  {max_diff:.6f} (should be small)")
    
    # Extrapolation to your original problem
    print("\n" + "-" * 60)
    print("EXTRAPOLATION TO YOUR ORIGINAL PROBLEM:")
    original_scale = 2000 - 2 + 1  # 1999 iterations
    original_tape_size = 20000
    test_scale = test_params['max_n'] - test_params['min_n'] + 1
    test_tape_size = test_params['tape_params']['N']
    
    scale_factor = (original_scale / test_scale) * (original_tape_size / test_tape_size)
    estimated_original_time = original_time * scale_factor
    estimated_optimized_time = optimized_time * scale_factor
    
    print(f"Estimated original time:    {estimated_original_time/3600:.1f} hours")
    print(f"Estimated optimized time:   {estimated_optimized_time/60:.1f} minutes")
    print(f"Time saved:                 {(estimated_original_time - estimated_optimized_time)/3600:.1f} hours")
    
    return speedup

if __name__ == "__main__":
    speedup = quick_benchmark()