"""
Phase 9: Parameter Sweep Analysis
==================================
This module explores how the Maxwell's demon system behaves across different 
sigma and omega parameter values.

Sigma (σ): Related to hot reservoir temperature - controls intrinsic transitions
Omega (ω): Related to cold reservoir temperature - controls bit-flip transitions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase_6'))

from Demon import Demon, PhysParams, T_H, T_C, DELTAE, GAMMA
from Tape import Tape
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import itertools


class ParameterSweepSimulation:
    """Simulation class for sweeping through different sigma and omega values.
    
    This class reuses the infrastructure from phase_6 to systematically test
    how the system responds to different parameter combinations.
    
    Attributes:
        demon_n (int): Number of demon states
        tau (float): Interaction time
        N (int): Number of bits on tape
        p0 (float): Initial probability of bit being 0
    """
    
    def __init__(self, demon_n: int = 2, tau: float = 1.0, 
                 N: int = 5000, p0: float = 1.0):
        """Initialize the parameter sweep simulation.
        
        Args:
            demon_n (int): Number of states per demon
            tau (float): Interaction time per demon
            N (int): Number of bits on tape
            p0 (float): Initial probability of bit being 0
        """
        self.demon_n = demon_n
        self.tau = tau
        self.N = N
        self.p0 = p0
    
    def run_single_configuration(self, sigma: float, omega: float, 
                                 verbose: bool = False) -> Dict:
        """Run simulation for a single (sigma, omega) configuration.
        
        Args:
            sigma (float): Intrinsic transition parameter
            omega (float): Outgoing transition parameter
            verbose (bool): Print detailed output
        
        Returns:
            dict: Results including phi, bias, Q_c, delta_S_b, etc.
        """
        # Create physical parameters
        phys_params = PhysParams(
            sigma=sigma,
            omega=omega,
            DeltaE=DELTAE,
            gamma=GAMMA
        )
        
        # Create demon and tape
        demon = Demon(n=self.demon_n, phys_params=phys_params, init_state='d0')
        tape = Tape(N=self.N, p0=self.p0)
        
        # Import the simulation class from phase_6
        from Simulation import StackedDemonSimulation
        
        # Run simulation with single demon (K=1)
        sim = StackedDemonSimulation(demons=[demon], tape=tape, tau=self.tau)
        final_tape, initial_tape, demon_states_history = sim.run_full_simulation()
        
        # Compute statistics
        stats = sim.compute_statistics(final_tape)
        
        # Add sigma and omega to results
        stats['sigma'] = sigma
        stats['omega'] = omega
        stats['Th'] = phys_params.Th
        stats['Tc'] = phys_params.Tc
        
        if verbose:
            print(f"\nσ={sigma:.3f}, ω={omega:.3f}")
            print(f"  Th={stats['Th']:.3f}, Tc={stats['Tc']:.3f}")
            print(f"  φ={stats['phi']:.4f}")
            print(f"  Bias out={stats['outgoing']['bias']:.4f}")
            print(f"  Q_c={stats['Q_c']:.4f}")
            print(f"  ΔS_B={stats['outgoing']['DeltaS_B']:.4f}")
        
        return stats
    
    def sweep_sigma_omega_grid(self, sigma_values: List[float], 
                               omega_values: List[float],
                               verbose: bool = False) -> Dict:
        """Sweep through a grid of sigma and omega values.
        
        Args:
            sigma_values (List[float]): List of sigma values to test
            omega_values (List[float]): List of omega values to test
            verbose (bool): Print progress
        
        Returns:
            dict: Results organized by (sigma, omega) pairs
        """
        results = {}
        total_runs = len(sigma_values) * len(omega_values)
        run_count = 0
        
        for sigma in sigma_values:
            for omega in omega_values:
                run_count += 1
                if verbose:
                    print(f"Progress: {run_count}/{total_runs} - σ={sigma:.3f}, ω={omega:.3f}", end='\r')
                
                try:
                    stats = self.run_single_configuration(sigma, omega, verbose=False)
                    results[(sigma, omega)] = stats
                except Exception as e:
                    print(f"\nError at σ={sigma}, ω={omega}: {e}")
                    results[(sigma, omega)] = None
        
        if verbose:
            print()  # New line after progress
        
        return results
    
    def sweep_sigma_fixed_omega(self, sigma_values: List[float], 
                                omega_fixed: float = 0.8,
                                verbose: bool = True) -> Dict:
        """Sweep sigma values while keeping omega fixed.
        
        Args:
            sigma_values (List[float]): List of sigma values to test
            omega_fixed (float): Fixed omega value
            verbose (bool): Print progress
        
        Returns:
            dict: Results for each sigma value
        """
        return self.sweep_sigma_omega_grid(sigma_values, [omega_fixed], verbose)
    
    def sweep_omega_fixed_sigma(self, omega_values: List[float], 
                                sigma_fixed: float = 0.3,
                                verbose: bool = True) -> Dict:
        """Sweep omega values while keeping sigma fixed.
        
        Args:
            omega_values (List[float]): List of omega values to test
            sigma_fixed (float): Fixed sigma value
            verbose (bool): Print progress
        
        Returns:
            dict: Results for each omega value
        """
        return self.sweep_sigma_omega_grid([sigma_fixed], omega_values, verbose)


def plot_heatmap(results: Dict, output_key: str = 'phi', 
                 title: str = None, save_path: str = None):
    """Create a heatmap visualization of results.
    
    Args:
        results (dict): Results from sweep_sigma_omega_grid
        output_key (str): Which output to visualize ('phi', 'Q_c', 'DeltaS_B', etc.)
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    # Extract unique sigma and omega values
    sigma_values = sorted(set(sigma for sigma, omega in results.keys()))
    omega_values = sorted(set(omega for sigma, omega in results.keys()))
    
    # Create grid for heatmap
    grid = np.zeros((len(omega_values), len(sigma_values)))
    
    # Fill grid with values
    for i, omega in enumerate(omega_values):
        for j, sigma in enumerate(sigma_values):
            if results.get((sigma, omega)) is not None:
                if output_key == 'DeltaS_B':
                    grid[i, j] = results[(sigma, omega)]['outgoing']['DeltaS_B']
                elif output_key == 'bias':
                    grid[i, j] = results[(sigma, omega)]['outgoing']['bias']
                else:
                    grid[i, j] = results[(sigma, omega)][output_key]
            else:
                grid[i, j] = np.nan
    
    # Create plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
    
    # Set ticks
    plt.xticks(range(len(sigma_values)), [f'{s:.2f}' for s in sigma_values])
    plt.yticks(range(len(omega_values)), [f'{o:.2f}' for o in omega_values])
    
    plt.xlabel('σ (sigma)', fontsize=12)
    plt.ylabel('ω (omega)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(output_key, fontsize=12)
    
    if title is None:
        title = f'{output_key} vs σ and ω'
    plt.title(title, fontsize=13)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_1d_sweep(results: Dict, sweep_param: str = 'sigma',
                  output_keys: List[str] = None,
                  title: str = None, save_path: str = None):
    """Plot results from a 1D parameter sweep.
    
    Args:
        results (dict): Results from sweep
        sweep_param (str): Which parameter was swept ('sigma' or 'omega')
        output_keys (List[str]): Which outputs to plot
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    if output_keys is None:
        output_keys = ['phi', 'Q_c']
    
    # Extract parameter values and sort
    param_values = sorted(set(key[0] if sweep_param == 'sigma' else key[1] 
                             for key in results.keys()))
    
    # Prepare data for each output
    data = {key: [] for key in output_keys}
    
    for param in param_values:
        # Find the result for this parameter value
        result_key = next((k for k in results.keys() if (k[0] == param if sweep_param == 'sigma' else k[1] == param)), None)
        
        if result_key and results[result_key] is not None:
            for key in output_keys:
                if key == 'DeltaS_B':
                    data[key].append(results[result_key]['outgoing']['DeltaS_B'])
                elif key == 'bias':
                    data[key].append(results[result_key]['outgoing']['bias'])
                else:
                    data[key].append(results[result_key][key])
        else:
            for key in output_keys:
                data[key].append(np.nan)
    
    # Create plot
    fig, axes = plt.subplots(1, len(output_keys), figsize=(6*len(output_keys), 5))
    if len(output_keys) == 1:
        axes = [axes]
    
    for ax, key in zip(axes, output_keys):
        ax.plot(param_values, data[key], marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(f'{sweep_param}', fontsize=12)
        ax.set_ylabel(key, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{key} vs {sweep_param}', fontsize=11)
    
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 9: Parameter Sweep Analysis")
    print("=" * 60)
    
    # Initialize simulator
    sim = ParameterSweepSimulation(
        demon_n=2,
        tau=1.0,
        N=5000,
        p0=1.0
    )
    
    # Example 1: Sweep sigma with fixed omega
    print("\n1. Sweeping σ with fixed ω=0.8")
    print("-" * 60)
    sigma_values = np.linspace(0.1, 0.9, 9)
    results_sigma = sim.sweep_sigma_fixed_omega(
        sigma_values=sigma_values,
        omega_fixed=0.8,
        verbose=True
    )
    plot_1d_sweep(results_sigma, sweep_param='sigma', 
                  output_keys=['phi', 'Q_c'],
                  title='System Response vs σ (ω=0.8)')
    
    # Example 2: Sweep omega with fixed sigma
    print("\n2. Sweeping ω with fixed σ=0.3")
    print("-" * 60)
    omega_values = np.linspace(0.1, 0.9, 9)
    results_omega = sim.sweep_omega_fixed_sigma(
        omega_values=omega_values,
        sigma_fixed=0.3,
        verbose=True
    )
    plot_1d_sweep(results_omega, sweep_param='omega',
                  output_keys=['phi', 'Q_c'],
                  title='System Response vs ω (σ=0.3)')
    
    # Example 3: 2D grid sweep (smaller grid for speed)
    print("\n3. 2D Grid Sweep")
    print("-" * 60)
    sigma_grid = np.linspace(0.2, 0.8, 5)
    omega_grid = np.linspace(0.2, 0.8, 5)
    results_grid = sim.sweep_sigma_omega_grid(
        sigma_values=sigma_grid,
        omega_values=omega_grid,
        verbose=True
    )
    
    # Plot heatmaps for different outputs
    plot_heatmap(results_grid, output_key='phi', 
                 title='Bit Flip Fraction (φ) vs σ and ω')
    plot_heatmap(results_grid, output_key='Q_c',
                 title='Energy to Cold Reservoir (Q_c) vs σ and ω')
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
