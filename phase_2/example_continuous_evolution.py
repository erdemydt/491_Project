#!/usr/bin/env python3
"""
Example demonstrating the continuous evolution approach with trajectory tracing.
"""

from monteCarlo import *
import matplotlib.pyplot as plt

def demo_trajectory():
    """Show a single trajectory evolution for visualization."""
    
    # Physical setup
    Th, Tc = 4.0, 1.6       # temperatures
    DeltaE = 1.0            # energy scale
    phys = PhysParams(Th=Th, Tc=Tc, DeltaE=DeltaE, gamma_hot=2.0, kappa_cold=1.5)
    
    print(f"Physical parameters:")
    print(f"  Th={Th}, Tc={Tc}, ΔE={DeltaE}")
    print(f"  σ (hot bias) = {phys.sigma:.4f}")
    print(f"  ω (cold bias) = {phys.omega:.4f}")
    print(f"  γ (hot rate) = {phys.gamma_hot}")
    print(f"  κ (cold rate) = {phys.kappa_cold}")
    
    # Simulation parameters
    epsilon_in = 0.8
    p0_in, _ = probs_from_epsilon(epsilon_in)
    T = 0.2         # total interaction time
    dt = 0.01       # time step
    
    print(f"\nSimulation parameters:")
    print(f"  Incoming bias ε = {epsilon_in}")
    print(f"  Total interaction time T = {T}")
    print(f"  Time step dt = {dt}")
    
    # Run one traced trajectory
    demon_init = "u"
    b_in, d_final, b_out, traj = step_with_continuous_evolution_traced(
        demon_init, p0_in, T, dt, phys, seed=42
    )
    
    print(f"\nSingle trajectory:")
    print(f"  Initial demon: {demon_init}")
    print(f"  Incoming bit: {b_in}")
    print(f"  Final demon: {d_final}")
    print(f"  Outgoing bit: {b_out}")
    print(f"  Number of time steps: {len(traj)}")
    
    # Count state changes
    changes = []
    for i in range(1, len(traj)):
        if traj[i][1] != traj[i-1][1]:
            changes.append((traj[i-1][0], traj[i-1][1], traj[i][1]))
    
    print(f"\nState changes during evolution:")
    if changes:
        for t, from_state, to_state in changes:
            print(f"  t={t:.3f}: {from_state} → {to_state}")
    else:
        print("  No state changes occurred")
    
    # Show state occupancy over time
    states = ["0u", "0d", "1u", "1d"]
    state_times = {s: 0.0 for s in states}
    
    for i in range(len(traj) - 1):
        state = traj[i][1]
        dt_step = traj[i+1][0] - traj[i][0]
        state_times[state] += dt_step
    
    print(f"\nState occupancy fractions:")
    for state in states:
        frac = state_times[state] / T
        print(f"  {state}: {frac:.3f} ({state_times[state]:.4f}/{T})")

def compare_methods():
    """Compare single-jump vs continuous evolution results."""
    
    # Physical setup
    phys = PhysParams(Th=4.0, Tc=1.6, DeltaE=1.0, gamma_hot=1.0, kappa_cold=1.0)
    
    # Common parameters
    N = 50000
    epsilon_in = 0.7
    p0_in, _ = probs_from_epsilon(epsilon_in)
    
    print(f"\nComparison with N={N} trials:")
    print(f"Incoming bias ε = {epsilon_in}")
    
    # Method 1: Single jump per window
    t_single = 0.05
    stats_single = run_sim(N, t_single, phys, p0_in, seed=123)
    
    # Method 2: Continuous evolution 
    T_cont = 0.05
    dt_cont = 0.001
    stats_cont = run_sim_continuous(N, T_cont, dt_cont, phys, p0_in, seed=123)
    
    print(f"\nResults:")
    print(f"{'Method':<20} {'ε_out':<10} {'P(u)':<10} {'P(0_out)':<10}")
    print("-" * 50)
    
    eps_single = stats_single['outgoing']['epsilon']
    pu_single = stats_single['demon']['pu']
    p0_single = stats_single['outgoing']['p0']
    
    eps_cont = stats_cont['outgoing']['epsilon'] 
    pu_cont = stats_cont['demon']['pu']
    p0_cont = stats_cont['outgoing']['p0']
    
    print(f"{'Single jump':<20} {eps_single:<10.4f} {pu_single:<10.4f} {p0_single:<10.4f}")
    print(f"{'Continuous':<20} {eps_cont:<10.4f} {pu_cont:<10.4f} {p0_cont:<10.4f}")
    print(f"{'Difference':<20} {abs(eps_single-eps_cont):<10.4f} {abs(pu_single-pu_cont):<10.4f} {abs(p0_single-p0_cont):<10.4f}")

if __name__ == "__main__":
    demo_trajectory()
    compare_methods()