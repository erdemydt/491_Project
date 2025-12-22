"""
Quick test script to verify phase_6 functionality
"""
from Demon import Demon, PhysParams
from Tape import Tape
from Simulation import StackedDemonSimulation
import numpy as np

print("=" * 60)
print("PHASE 6 - STACKED DEMONS TEST")
print("=" * 60)

# Test 1: Basic stacked demon simulation
print("\n[Test 1] Basic Stacked Demon Simulation (K=3)")
print("-" * 60)

phys_params = PhysParams(
    sigma=0.3,
    omega=0.5,
    DeltaE=1.0,
    gamma=1.0,
    delta_e_mode='per_state',
    preserve_mode='sigma_omega'
)

demons = [Demon(n=3, phys_params=phys_params, init_state='d0') for _ in range(3)]
tape = Tape(N=100, p0=0.9)

sim = StackedDemonSimulation(demons=demons, tape=tape, tau=1.0)
final_tape, initial_tape, demon_history = sim.run_full_simulation()

stats = sim.compute_statistics(final_tape)

print(f"Initial p0: {stats['incoming']['p0']:.4f}")
print(f"Final p0: {stats['outgoing']['p0']:.4f}")
print(f"Phi (bit flip fraction): {stats['phi']:.4f}")
print(f"Bias in: {stats['incoming']['bias']:.4f}")
print(f"Bias out: {stats['outgoing']['bias']:.4f}")
print(f"Energy transferred Q_c: {stats['Q_c']:.4f}")
print(f"Entropy change ΔS_B: {stats['outgoing']['DeltaS_B']:.6f}")
print(f"Number of demons K: {stats['K']}")

# Test 2: Per-state mode
print("\n[Test 2] Per-State DeltaE Mode")
print("-" * 60)

phys_params_per_state = PhysParams(
    sigma=0.3,
    omega=0.5,
    DeltaE=1.0,
    gamma=1.0,
    delta_e_mode='per_state',
    preserve_mode='sigma_omega'
)

print(f"DeltaE mode: {phys_params_per_state.delta_e_mode}")
print(f"DeltaE (per state): {phys_params_per_state.DeltaE}")
print(f"DeltaE per state: {phys_params_per_state.get_delta_e_per_state():.4f}")
print(f"Sigma: {phys_params_per_state.sigma:.4f}")
print(f"Omega: {phys_params_per_state.omega:.4f}")
print(f"T_H: {phys_params_per_state.Th:.4f}")
print(f"T_C: {phys_params_per_state.Tc:.4f}")

# Test 3: Total mode with preserved temperatures
print("\n[Test 3] Total DeltaE Mode - Preserve Temperatures")
print("-" * 60)

phys_params_total_temp = PhysParams(
    Th=1.6,
    Tc=1.0,
    DeltaE=4.0,  # Total from ground to top
    gamma=1.0,
    delta_e_mode='total',
    preserve_mode='temperatures',
    demon_n=5  # 5 states means 4 transitions
)

print(f"DeltaE mode: {phys_params_total_temp.delta_e_mode}")
print(f"DeltaE (total): {phys_params_total_temp.DeltaE}")
print(f"DeltaE per state: {phys_params_total_temp.get_delta_e_per_state():.4f}")
print(f"Expected per state: {4.0 / (5-1):.4f}")
print(f"Sigma: {phys_params_total_temp.sigma:.4f}")
print(f"Omega: {phys_params_total_temp.omega:.4f}")
print(f"T_H (preserved): {phys_params_total_temp.Th:.4f}")
print(f"T_C (preserved): {phys_params_total_temp.Tc:.4f}")

# Test 4: Total mode with preserved sigma/omega
print("\n[Test 4] Total DeltaE Mode - Preserve Sigma/Omega")
print("-" * 60)

phys_params_total_sigma = PhysParams(
    sigma=0.3,
    omega=0.5,
    DeltaE=3.0,  # Total from ground to top
    gamma=1.0,
    delta_e_mode='total',
    preserve_mode='sigma_omega',
    demon_n=4  # 4 states means 3 transitions
)

print(f"DeltaE mode: {phys_params_total_sigma.delta_e_mode}")
print(f"DeltaE (total): {phys_params_total_sigma.DeltaE}")
print(f"DeltaE per state: {phys_params_total_sigma.get_delta_e_per_state():.4f}")
print(f"Expected per state: {3.0 / (4-1):.4f}")
print(f"Sigma (preserved): {phys_params_total_sigma.sigma:.4f}")
print(f"Omega (preserved): {phys_params_total_sigma.omega:.4f}")
print(f"T_H: {phys_params_total_sigma.Th:.4f}")
print(f"T_C: {phys_params_total_sigma.Tc:.4f}")

# Test 5: Verify demon energy calculations
print("\n[Test 5] Demon Energy Calculations")
print("-" * 60)

demon = Demon(n=4, phys_params=phys_params_per_state, init_state='d0')
print(f"Number of states: {demon.n}")
print(f"State energies:")
for i in range(demon.n):
    print(f"  State d{i}: {demon.get_energy_of_state(i):.4f}")
print(f"Total delta E: {demon.get_total_delta_e():.4f}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
