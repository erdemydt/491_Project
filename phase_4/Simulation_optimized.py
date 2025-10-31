from Demon import Demon, PhysParams, T_H, T_C, DELTAE, GAMMA
from Tape import Tape
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import time

class OptimizedSimulation:
    """Optimized version of the Simulation class with significant performance improvements."""
    
    def __init__(self, demon: Demon, tape: Tape, tau: float):
        self.demon = demon
        self.init_tape_arr = tape.tape_arr
        self.tau = tau
        self.N = tape.N
        self.init_tape = tape
        self.total_energy_transferred = 0.0
        
        # Pre-compute and cache all possible rates for this demon
        self._precompute_rate_matrices()
    
    def _precompute_rate_matrices(self):
        """Pre-compute all possible transition rates as matrices for faster lookup."""
        n = self.demon.n
        
        # Create transition rate matrices for each bit state
        # intrinsic_rates[i][j] = rate from demon state i to demon state j (for same bit state)
        self.intrinsic_rates = np.zeros((n, n))
        
        # outgoing_rates[bit][demon_state] = rate for bit flip + demon transition
        self.outgoing_rates = np.zeros((2, n))
        
        # Fill intrinsic rates (demon transitions without bit flip)
        for i in range(n - 1):
            delta_e = self.demon.delta_e_values[i]
            # Forward transition (i -> i+1)
            self.intrinsic_rates[i][i+1] = self.demon.phys_params.gamma * np.exp(-delta_e / (2 * self.demon.phys_params.Th))
            # Backward transition (i+1 -> i)
            self.intrinsic_rates[i+1][i] = self.demon.phys_params.gamma * np.exp(delta_e / (2 * self.demon.phys_params.Th))
        
        # Fill outgoing rates (bit flip + demon transition)
        for i in range(n - 1):
            delta_e = self.demon.delta_e_values[i]
            omega = np.tanh(delta_e / (2 * self.demon.phys_params.Tc))
            # Bit 0 -> 1, demon i -> i+1
            self.outgoing_rates[0][i] = 1 - omega
            # Bit 1 -> 0, demon i+1 -> i
            self.outgoing_rates[1][i+1] = 1 + omega
    
    def run_gillespie_window_optimized(self, bit_state: int, demon_state: int) -> Tuple[int, int, float]:
        """Optimized Gillespie algorithm using integer states and pre-computed rates.
        
        Args:
            bit_state (int): Initial bit state (0 or 1)
            demon_state (int): Initial demon state (0 to n-1)
            
        Returns:
            final_bit_state (int): Final bit state
            final_demon_state (int): Final demon state  
            time_elapsed (float): Total time elapsed
        """
        time_elapsed = 0.0
        current_bit = bit_state
        current_demon = demon_state
        n = self.demon.n
        
        while time_elapsed < self.tau:
            # Collect all possible transitions and their rates
            transitions = []
            rates = []
            
            # Intrinsic demon transitions (same bit state)
            if current_demon > 0:  # Can go down
                transitions.append((current_bit, current_demon - 1))
                rates.append(self.intrinsic_rates[current_demon][current_demon - 1])
            
            if current_demon < n - 1:  # Can go up
                transitions.append((current_bit, current_demon + 1))
                rates.append(self.intrinsic_rates[current_demon][current_demon + 1])
            
            # Outgoing transitions (bit flip + demon transition)
            if current_bit == 0 and current_demon < n - 1:  # 0->1, demon up
                transitions.append((1, current_demon + 1))
                rates.append(self.outgoing_rates[0][current_demon])
            elif current_bit == 1 and current_demon > 0:  # 1->0, demon down
                transitions.append((0, current_demon - 1))
                rates.append(self.outgoing_rates[1][current_demon])
            
            if not rates:  # No transitions possible
                break
                
            total_rate = sum(rates)
            
            # Time to next event
            dt = np.random.exponential(1 / total_rate)
            if time_elapsed + dt > self.tau:
                break
                
            time_elapsed += dt
            
            # Choose which transition occurs
            rand = np.random.uniform(0, total_rate)
            cumulative_rate = 0.0
            for i, rate in enumerate(rates):
                cumulative_rate += rate
                if rand < cumulative_rate:
                    current_bit, current_demon = transitions[i]
                    break
        
        return current_bit, current_demon, time_elapsed
    
    def run_full_simulation_optimized(self) -> Tuple[Tape, Tape, List[int]]:
        """Optimized full simulation using integer states and vectorized operations where possible."""
        final_tape_arr = self.init_tape_arr.copy()
        
        # Convert string states to integers for faster processing
        bit_states = np.array([int(bit) for bit in final_tape_arr])
        
        # Parse initial demon state
        current_demon_state = int(self.demon.current_state[1:])  # Remove 'd' prefix
        demon_states_sequence = [current_demon_state]
        
        # Process each bit
        for i in range(self.N):
            final_bit, final_demon, _ = self.run_gillespie_window_optimized(
                bit_states[i], current_demon_state
            )
            bit_states[i] = final_bit
            current_demon_state = final_demon
            demon_states_sequence.append(current_demon_state)
        
        # Convert back to string format for compatibility
        final_tape_arr = np.array([str(bit) for bit in bit_states])
        final_tape = Tape(N=self.N, p0=self.init_tape.p0, tape_arr=final_tape_arr)
        
        return final_tape, self.init_tape, demon_states_sequence

    def compute_statistics_optimized(self, final_tape: Tape) -> dict:
        """Optimized statistics computation using vectorized operations."""
        # Use numpy for faster counting
        initial_arr = np.array(self.init_tape.tape_arr, dtype=int)
        final_arr = np.array(final_tape.tape_arr, dtype=int)
        
        initial_p0 = np.mean(initial_arr == 0)
        initial_p1 = 1 - initial_p0
        final_p0 = np.mean(final_arr == 0)
        final_p1 = 1 - final_p0
        
        # Vectorized entropy calculation
        def fast_entropy(p0, p1):
            entropy = 0.0
            if p0 > 0: entropy -= p0 * np.log(p0)
            if p1 > 0: entropy -= p1 * np.log(p1)
            return entropy
        
        initial_entropy = fast_entropy(initial_p0, initial_p1)
        final_entropy = fast_entropy(final_p0, final_p1)
        
        delta_s_b = final_entropy - initial_entropy
        bias_in = initial_p0 - initial_p1
        bias_out = final_p0 - final_p1
        phi = final_p1 - initial_p1
        
        # Energy calculation
        delta_e = self.demon.get_delta_e_for_state_n(0)
        total_energy_transferred = phi * self.N * delta_e
        
        stats = {
            "incoming": {
                "distribution": {'0': initial_p0, '1': initial_p1},
                "p0": initial_p0,
                "p1": initial_p1,
                "entropy": initial_entropy,
                "bias": bias_in
            },
            "outgoing": {
                "distribution": {'0': final_p0, '1': final_p1},
                "p0": final_p0,
                "p1": final_p1,
                "entropy": final_entropy,
                "DeltaS_B": delta_s_b,
                "phi": phi,
                "Q_c": total_energy_transferred,
                "bias": bias_out
            },
            "demon": {
                "final_state": self.demon.current_state,
                "pu": 1.0 / self.demon.n  # Placeholder
            }
        }
        return stats

    def run_monte_carlo_simulation(self, mc_samples: int = 1, seed_offset: int = 0) -> dict:
        """Run multiple simulations and average the results to reduce noise.
        
        Args:
            mc_samples (int): Number of Monte Carlo samples to average over
            seed_offset (int): Offset for random seed to ensure different random sequences
            
        Returns:
            dict: Averaged statistics with standard deviations
        """
        if mc_samples == 1:
            # Single run - use original method
            final_tape, _, demon_sequence = self.run_full_simulation_optimized()
            return self.compute_statistics_optimized(final_tape)
        
        # Multiple runs for averaging
        all_stats = []
        original_seed = np.random.get_state()
        
        for run in range(mc_samples):
            # Set different seed for each run
            np.random.seed(seed_offset + run * 1000)
            
            # Run simulation
            final_tape, _, _ = self.run_full_simulation_optimized()
            stats = self.compute_statistics_optimized(final_tape)
            all_stats.append(stats)
        
        # Restore original random state
        np.random.set_state(original_seed)
        
        # Average the results
        averaged_stats = self._average_statistics(all_stats)
        return averaged_stats
    
    def _average_statistics(self, stats_list: List[dict]) -> dict:
        """Average statistics from multiple runs and compute standard deviations."""
        n_runs = len(stats_list)
        
        # Extract all values for averaging
        values = {
            'incoming_p0': [s['incoming']['p0'] for s in stats_list],
            'incoming_p1': [s['incoming']['p1'] for s in stats_list],
            'incoming_entropy': [s['incoming']['entropy'] for s in stats_list],
            'incoming_bias': [s['incoming']['bias'] for s in stats_list],
            'outgoing_p0': [s['outgoing']['p0'] for s in stats_list],
            'outgoing_p1': [s['outgoing']['p1'] for s in stats_list],
            'outgoing_entropy': [s['outgoing']['entropy'] for s in stats_list],
            'outgoing_DeltaS_B': [s['outgoing']['DeltaS_B'] for s in stats_list],
            'outgoing_phi': [s['outgoing']['phi'] for s in stats_list],
            'outgoing_Q_c': [s['outgoing']['Q_c'] for s in stats_list],
            'outgoing_bias': [s['outgoing']['bias'] for s in stats_list]
        }
        
        # Compute means and standard deviations
        means = {key: np.mean(vals) for key, vals in values.items()}
        stds = {key: np.std(vals, ddof=1) if n_runs > 1 else 0.0 for key, vals in values.items()}
        
        # Build averaged statistics dictionary
        averaged_stats = {
            "incoming": {
                "distribution": {'0': means['incoming_p0'], '1': means['incoming_p1']},
                "p0": means['incoming_p0'],
                "p1": means['incoming_p1'],
                "entropy": means['incoming_entropy'],
                "bias": means['incoming_bias'],
                # Add standard deviations
                "p0_std": stds['incoming_p0'],
                "p1_std": stds['incoming_p1'],
                "entropy_std": stds['incoming_entropy'],
                "bias_std": stds['incoming_bias']
            },
            "outgoing": {
                "distribution": {'0': means['outgoing_p0'], '1': means['outgoing_p1']},
                "p0": means['outgoing_p0'],
                "p1": means['outgoing_p1'],
                "entropy": means['outgoing_entropy'],
                "DeltaS_B": means['outgoing_DeltaS_B'],
                "phi": means['outgoing_phi'],
                "Q_c": means['outgoing_Q_c'],
                "bias": means['outgoing_bias'],
                # Add standard deviations
                "p0_std": stds['outgoing_p0'],
                "p1_std": stds['outgoing_p1'],
                "entropy_std": stds['outgoing_entropy'],
                "DeltaS_B_std": stds['outgoing_DeltaS_B'],
                "phi_std": stds['outgoing_phi'],
                "Q_c_std": stds['outgoing_Q_c'],
                "bias_std": stds['outgoing_bias']
            },
            "demon": {
                "final_state": self.demon.current_state,
                "pu": 1.0 / self.demon.n
            },
            "monte_carlo": {
                "n_samples": n_runs,
                "confidence_95_percent": 1.96 * np.array(list(stds.values())) / np.sqrt(n_runs)
            }
        }
        return averaged_stats


def calculate_optimal_step(min_n: int, max_n: int, target_points: int = 100) -> int:
    """Calculate an optimal step size to get approximately target_points samples.
    
    Args:
        min_n (int): Minimum n value
        max_n (int): Maximum n value  
        target_points (int): Desired number of sample points
        
    Returns:
        int: Recommended step size
    """
    total_range = max_n - min_n + 1
    step = max(1, total_range // target_points)
    actual_points = len(range(min_n, max_n + 1, step))
    print(f"Recommended step={step} will give {actual_points} points (target: {target_points})")
    return step


def suggest_sampling_strategy(min_n: int, max_n: int, tape_size: int, 
                            max_runtime_minutes: float = 10.0) -> dict:
    """Suggest the best sampling strategy based on problem size and time constraints.
    
    Args:
        min_n (int): Minimum n value
        max_n (int): Maximum n value
        tape_size (int): Size of the tape (N)
        max_runtime_minutes (float): Maximum acceptable runtime in minutes
        
    Returns:
        dict: Recommended parameters for the simulation
    """
    total_range = max_n - min_n + 1
    
    # Rough estimation of time per simulation (very approximate)
    # Based on: tape_size * average_transitions_per_bit * computation_overhead
    estimated_time_per_sim = (tape_size / 10000) * 0.1  # seconds per simulation
    total_estimated_time = total_range * estimated_time_per_sim
    
    print(f"Problem size analysis:")
    print(f"  Range: n={min_n} to {max_n} ({total_range} points)")
    print(f"  Tape size: {tape_size}")
    print(f"  Estimated time for full range: {total_estimated_time/60:.1f} minutes")
    print(f"  Target runtime: {max_runtime_minutes} minutes")
    
    if total_estimated_time/60 <= max_runtime_minutes:
        # Can run full range
        return {
            "strategy": "full_range",
            "use_sampling": False,
            "step": None,
            "max_samples": None,
            "estimated_time_minutes": total_estimated_time/60
        }
    
    # Need sampling - decide between step vs logarithmic
    target_points = int(max_runtime_minutes * 60 / estimated_time_per_sim)
    target_points = min(target_points, 500)  # Cap at reasonable number
    
    if total_range > 1000:
        # Large range - logarithmic sampling is better
        return {
            "strategy": "logarithmic",
            "use_sampling": True,
            "step": None,
            "max_samples": target_points,
            "estimated_time_minutes": target_points * estimated_time_per_sim / 60
        }
    else:
        # Medium range - step sampling is fine
        step = calculate_optimal_step(min_n, max_n, target_points)
        return {
            "strategy": "step",
            "use_sampling": False,
            "step": step,
            "max_samples": None,
            "estimated_time_minutes": target_points * estimated_time_per_sim / 60
        }


def plot_total_energy_vs_demon_n_optimized(min_n: int, max_n: int, tape_params: dict = None, tau: float = 1.0, 
                                          use_sampling: bool = True, max_samples: int = 200, step: int = None,
                                          mc_samples: int = 1, show_error_bars: bool = True):
    """Highly optimized version with multiple performance improvements and Monte Carlo averaging.
    
    Args:
        min_n (int): Minimum number of demon states
        max_n (int): Maximum number of demon states
        tape_params (dict): Tape parameters (N, p0, etc.)
        tau (float): Interaction time
        use_sampling (bool): Whether to use intelligent logarithmic sampling
        max_samples (int): Maximum number of samples when using sampling
        step (int): Step size for linear sampling (overrides use_sampling if provided)
        mc_samples (int): Number of Monte Carlo samples per data point (>=1 for noise reduction)
        show_error_bars (bool): Whether to show error bars on plots when mc_samples > 1
    """
    
    print(f"Starting optimized simulation: n={min_n} to {max_n}, N={tape_params['N']}, tau={tau}")
    if mc_samples > 1:
        print(f"Using Monte Carlo averaging: {mc_samples} samples per data point")
    start_time = time.time()
    
    # Strategy 1: Choose sampling strategy
    if step is not None:
        # Linear sampling with specified step
        n_values = list(range(min_n, max_n + 1, step))
        print(f"Using step sampling: {len(n_values)} points with step={step} (instead of {max_n - min_n + 1})")
    elif use_sampling and (max_n - min_n + 1) > max_samples:
        # Use logarithmic sampling for better coverage of the range
        n_values = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), max_samples).astype(int))
        n_values = n_values[n_values >= min_n]  # Ensure minimum
        print(f"Using logarithmic sampling: {len(n_values)} points instead of {max_n - min_n + 1}")
    else:
        n_values = list(range(min_n, max_n + 1))
        print(f"Using full range: {len(n_values)} points")
    
    # Strategy 2: Reuse the same tape for all simulations (if appropriate for your physics)
    # This assumes the initial tape state doesn't need to be different for each demon
    base_tape = Tape(**tape_params)
    
    # Strategy 3: Pre-allocate result arrays
    total_energy_values = np.zeros(len(n_values))
    bias_vals = np.zeros(len(n_values))
    total_energy_stds = np.zeros(len(n_values))
    bias_stds = np.zeros(len(n_values))
    
    # Strategy 4: Batch processing with progress reporting
    checkpoint_interval = max(1, len(n_values) // 20)  # Report every 5%
    
    for idx, n in enumerate(n_values):
        # Create demon with cached computations
        demon = Demon(n=n)
        
        # Use the same tape for consistency and speed
        sim = OptimizedSimulation(demon=demon, tape=base_tape, tau=tau)
        
        # Run Monte Carlo simulation (single run if mc_samples=1)
        stats = sim.run_monte_carlo_simulation(mc_samples=mc_samples, seed_offset=idx * 42)
        
        total_energy_values[idx] = stats["outgoing"]["Q_c"]
        bias_vals[idx] = stats["outgoing"]["bias"]
        
        # Store standard deviations if available
        if mc_samples > 1:
            total_energy_stds[idx] = stats["outgoing"]["Q_c_std"]
            bias_stds[idx] = stats["outgoing"]["bias_std"]
        
        # Progress reporting
        if idx % checkpoint_interval == 0 or idx == len(n_values) - 1:
            elapsed = time.time() - start_time
            progress = (idx + 1) / len(n_values) * 100
            eta = elapsed / (idx + 1) * (len(n_values) - idx - 1)
            if mc_samples > 1:
                print(f"{progress:.1f}% completed (n={n}, MC={mc_samples}), elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            else:
                print(f"{progress:.1f}% completed (n={n}), elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Simulation completed in {total_time:.2f} seconds")
    
    # Plotting
    phys_params = demon.phys_params
    
    plt.figure(figsize=(12, 8))
    
    # Energy plot
    plt.subplot(2, 1, 1)
    if mc_samples > 1 and show_error_bars:
        plt.errorbar(n_values, total_energy_values, yerr=total_energy_stds, 
                    marker='o', markersize=3, capsize=2, capthick=1, 
                    label=f'MC avg (n={mc_samples})')
    else:
        plt.plot(n_values, total_energy_values, marker='o', markersize=3)
    
    plt.xlabel('Number of Demon States (n)')
    plt.ylabel('Total Energy Transferred to Cold Reservoir (Q_c)')
    title_suffix = f' [MC avg: {mc_samples} samples]' if mc_samples > 1 else ''
    plt.title(f'Energy Transfer vs n (ΔE={phys_params.DeltaE}, Th={phys_params.Th}, Tc={phys_params.Tc}, bias_in={tape_params["p0"]*2-1:.1f}){title_suffix}')
    plt.grid(True)
    if mc_samples > 1 and show_error_bars:
        plt.legend()
    
    # Bias plot
    plt.subplot(2, 1, 2)
    if mc_samples > 1 and show_error_bars:
        plt.errorbar(n_values, bias_vals, yerr=bias_stds, 
                    marker='o', markersize=3, capsize=2, capthick=1, 
                    color='cyan', label=f'MC avg (n={mc_samples})')
    else:
        plt.plot(n_values, bias_vals, marker='o', markersize=3, color='cyan')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Demon States (n)')
    plt.ylabel('Bias Out')
    plt.title(f'Bias Out vs n (ΔE={phys_params.DeltaE}, Th={phys_params.Th}, Tc={phys_params.Tc}, bias_in={tape_params["p0"]*2-1:.1f}){title_suffix}')
    plt.grid(True)
    if mc_samples > 1 and show_error_bars:
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Return results with error information
    results = {
        'n_values': n_values,
        'total_energy_values': total_energy_values,
        'bias_vals': bias_vals,
        'mc_samples': mc_samples
    }
    
    if mc_samples > 1:
        results.update({
            'total_energy_stds': total_energy_stds,
            'bias_stds': bias_stds
        })
    
    return results


def analyze_monte_carlo_convergence(min_n: int, max_n: int, tape_params: dict, tau: float = 1.0,
                                   mc_samples_list: List[int] = [1, 5, 10, 25, 50], 
                                   test_points: int = 5):
    """Analyze how Monte Carlo averaging reduces noise with increasing sample size.
    
    Args:
        min_n, max_n: Range for demon states 
        tape_params: Tape parameters
        tau: Interaction time
        mc_samples_list: List of MC sample sizes to test
        test_points: Number of n values to test (evenly spaced)
        
    Returns:
        dict: Results showing noise reduction vs MC sample size
    """
    print("=" * 60)
    print("MONTE CARLO CONVERGENCE ANALYSIS")
    print("=" * 60)
    
    # Select test points
    n_test = np.linspace(min_n, max_n, test_points, dtype=int)
    
    results = {
        'mc_samples': mc_samples_list,
        'n_values': n_test,
        'bias_std_vs_mc': [],
        'energy_std_vs_mc': []
    }
    
    base_tape = Tape(**tape_params)
    
    for mc_samples in mc_samples_list:
        print(f"\nTesting MC samples = {mc_samples}")
        bias_stds = []
        energy_stds = []
        
        for n in n_test:
            demon = Demon(n=n)
            sim = OptimizedSimulation(demon=demon, tape=base_tape, tau=tau)
            
            if mc_samples == 1:
                # For single sample, estimate std by running multiple times
                temp_stats = []
                for _ in range(10):  # Run 10 times to estimate noise
                    final_tape, _, _ = sim.run_full_simulation_optimized()
                    stats = sim.compute_statistics_optimized(final_tape)
                    temp_stats.append(stats)
                
                bias_values = [s["outgoing"]["bias"] for s in temp_stats]
                energy_values = [s["outgoing"]["Q_c"] for s in temp_stats]
                bias_stds.append(np.std(bias_values))
                energy_stds.append(np.std(energy_values))
            else:
                # Use built-in MC averaging
                stats = sim.run_monte_carlo_simulation(mc_samples=mc_samples)
                bias_stds.append(stats["outgoing"]["bias_std"])
                energy_stds.append(stats["outgoing"]["Q_c_std"])
        
        results['bias_std_vs_mc'].append(np.mean(bias_stds))
        results['energy_std_vs_mc'].append(np.mean(energy_stds))
        print(f"  Average bias std: {np.mean(bias_stds):.6f}")
        print(f"  Average energy std: {np.mean(energy_stds):.6f}")
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.loglog(mc_samples_list, results['bias_std_vs_mc'], 'o-', label='Observed')
    # Theoretical 1/sqrt(N) line
    theoretical = results['bias_std_vs_mc'][0] / np.sqrt(np.array(mc_samples_list))
    plt.loglog(mc_samples_list, theoretical, '--', alpha=0.7, label='1/√N theoretical')
    plt.xlabel('MC Samples')
    plt.ylabel('Average Bias Standard Deviation')
    plt.title('Bias Noise vs Monte Carlo Samples')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.loglog(mc_samples_list, results['energy_std_vs_mc'], 'o-', label='Observed')
    theoretical = results['energy_std_vs_mc'][0] / np.sqrt(np.array(mc_samples_list))
    plt.loglog(mc_samples_list, theoretical, '--', alpha=0.7, label='1/√N theoretical')
    plt.xlabel('MC Samples')
    plt.ylabel('Average Energy Standard Deviation')
    plt.title('Energy Noise vs Monte Carlo Samples')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results


def suggest_mc_samples(noise_tolerance: float = 0.01, confidence_level: float = 0.95) -> int:
    """Suggest number of MC samples needed to achieve desired noise level.
    
    Args:
        noise_tolerance: Desired relative standard error (e.g., 0.01 = 1%)
        confidence_level: Confidence level for error bars (e.g., 0.95 = 95%)
        
    Returns:
        int: Suggested number of MC samples
    """
    # For 95% confidence, we need 1.96 * std_error < tolerance
    # std_error = std / sqrt(n)
    # So we need: 1.96 * std / sqrt(n) < tolerance
    # n > (1.96 * std / tolerance)^2
    
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
    
    # Rough estimate: assume std ~ 0.1 (10% of typical signal)
    estimated_std = 0.1
    min_samples = ((z_score * estimated_std) / noise_tolerance) ** 2
    
    suggested = int(np.ceil(min_samples))
    
    print(f"Noise reduction analysis:")
    print(f"  Target relative error: {noise_tolerance*100:.1f}%")
    print(f"  Confidence level: {confidence_level*100:.0f}%")
    print(f"  Suggested MC samples: {suggested}")
    print(f"  Expected error reduction: {1/np.sqrt(suggested):.3f} vs single run")
    
    return max(1, suggested)


# Comparison function to benchmark improvements
def benchmark_comparison(min_n: int = 2, max_n: int = 50, tape_params: dict = None, tau: float = 1.0, step: int = None):
    """Compare performance between original and optimized versions.
    
    Args:
        min_n (int): Minimum number of demon states
        max_n (int): Maximum number of demon states
        tape_params (dict): Tape parameters
        tau (float): Interaction time
        step (int): Step size for sampling (optional)
    """
    
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 1.0}  # Smaller for benchmarking
    
    # Original version timing
    print("Running original version...")
    from Simulation import plot_total_energy_vs_demon_n
    
    start_time = time.time()
    # We'll simulate the original function but with limited range for comparison
    total_energy_values_orig = []
    bias_vals_orig = []
    n_values = list(range(min_n, max_n + 1))
    
    for n in n_values:
        demon = Demon(n=n)
        tape = Tape(**tape_params)
        from Simulation import Simulation
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        total_energy_values_orig.append(stats["outgoing"]["Q_c"])
        bias_vals_orig.append(stats["outgoing"]["bias"])
    
    original_time = time.time() - start_time
    
    # Optimized version timing
    print("Running optimized version...")
    start_time = time.time()
    n_vals_opt, energy_opt, bias_opt = plot_total_energy_vs_demon_n_optimized(
        min_n=min_n, max_n=max_n, tape_params=tape_params, tau=tau, use_sampling=False, step=step
    )
    optimized_time = time.time() - start_time
    
    # Results
    speedup = original_time / optimized_time
    print(f"\nBENCHMARK RESULTS:")
    print(f"Original time: {original_time:.2f} seconds")
    print(f"Optimized time: {optimized_time:.2f} seconds") 
    print(f"Speedup: {speedup:.1f}x faster")
    print(f"Range tested: n={min_n} to {max_n} ({len(n_values)} points)")
    print(f"Tape size: {tape_params['N']} bits")


if __name__ == "__main__":
    # Example parameters
    test_params = {
        "min_n": 20,
        "max_n": 50,
        "tape_params": {"N": 8000, "p0": 1.0},
        "tau": 20.0
    }
    
    # Get recommendation for sampling strategy
    print("=" * 60)
    print("SAMPLING STRATEGY RECOMMENDATION")
    print("=" * 60)
    
    recommendation = suggest_sampling_strategy(
        min_n=test_params["min_n"], 
        max_n=test_params["max_n"], 
        tape_size=test_params["tape_params"]["N"], 
        max_runtime_minutes=5.0
    )
    
    print(f"\nRecommended strategy: {recommendation['strategy']}")
    print(f"Estimated runtime: {recommendation['estimated_time_minutes']:.1f} minutes")
    
    # Get recommendation for Monte Carlo samples
    print(f"\n{'='*60}")
    print("MONTE CARLO SAMPLING RECOMMENDATION")
    print("=" * 60)
    
    suggested_mc = suggest_mc_samples(noise_tolerance=0.05, confidence_level=0.95)  # 5% error tolerance

    
    mc_samples_to_use = min(suggested_mc, 10)  # Cap at 10 for demo
    
    results_mc = plot_total_energy_vs_demon_n_optimized(
        **test_params,
        step=5,
        mc_samples=50,
        show_error_bars=True
    )



