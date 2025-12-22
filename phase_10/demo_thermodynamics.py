"""
Demonstration of thermodynamic analysis for Phase 10 demons.

This script shows how to use the new thermodynamic tracking capabilities
to analyze energy and entropy exchanges with hot/cold reservoirs.
"""

import numpy as np
from Demon import TwoBitDemon, SingleBitDemon, PhysParams
from Tape import TwoBitTape
from Simulation import compare_demons_with_thermodynamics

# Physical parameters
phys_params = PhysParams(
    Th=1.6,
    Tc=1.0,
    DeltaE=1.0,
    gamma=1.0
)

# Tape parameters
tape_params = {
    'N': 10000,
    'p0': 1.,
    'seed': 42
}

# Interaction time
tau = 10.0

print("=" * 60)
print("Phase 10: Thermodynamic Analysis Demo")
print("=" * 60)
print(f"\nPhysical Parameters:")
print(f"  T_h = {phys_params.Th:.2f}")
print(f"  T_c = {phys_params.Tc:.2f}")
print(f"  ΔE = {phys_params.DeltaE:.2f}")
print(f"  γ = {phys_params.gamma:.2f}")
print(f"  σ = {phys_params.sigma:.4f} (intrinsic bias)")
print(f"  ω = {phys_params.omega:.4f} (cooperative bias)")
print(f"\nTape: N = {tape_params['N']}, p0 = {tape_params['p0']}")
print(f"Interaction time: τ = {tau}")

# Create custom two-bit demon with default transitions
two_bit_demon = TwoBitDemon(phys_params=phys_params, init_state='d')
# Default transitions: d00↔u01, u11↔d10

print(f"\nTwo-Bit Demon Transitions:")
for (d_from, b1_from, b2_from), (d_to, b1_to, b2_to) in two_bit_demon.cooperative_transitions.items():
    print(f"  {d_from}{b1_from}{b2_from} ↔ {d_to}{b1_to}{b2_to}")

print("\n" + "=" * 60)
print("Running comparison with thermodynamic tracking...")
print("=" * 60)

# Run comparison with thermodynamics
comparison = compare_demons_with_thermodynamics(
    tape_params=tape_params,
    phys_params=phys_params,
    tau=tau,
    seed=42,
    two_bit_demon=two_bit_demon,
    title="Default Transitions: d00↔u01, u11↔d10",
    plot=True
)

# Print summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

two_bit_thermo = comparison['two_bit']['thermodynamics']
single_bit_thermo = comparison['single_bit']['thermodynamics']

two_bit_stats = comparison['two_bit']['stats']
single_bit_stats = comparison['single_bit']['stats']

print("\n--- TWO-BIT DEMON ---")
print(f"\nEnergy:")
print(f"  Q_h (from hot) = {two_bit_thermo['energy']['Q_h']:.6f}")
print(f"  Q_c (to cold)  = {two_bit_thermo['energy']['Q_c']:.6f}")
print(f"  Q_total        = {two_bit_thermo['energy']['Q_total']:.6f}")

print(f"\nEntropy:")
print(f"  ΔS_h (hot reservoir)   = {two_bit_thermo['entropy']['S_h']:.6f}")
print(f"  ΔS_c (cold reservoir)  = {two_bit_thermo['entropy']['S_c']:.6f}")
print(f"  ΔS_total (production)  = {two_bit_thermo['entropy']['S_total_production']:.6f}")

print(f"\nTransitions:")
print(f"  Intrinsic up:      {two_bit_thermo['transitions']['intrinsic_up']}")
print(f"  Intrinsic down:    {two_bit_thermo['transitions']['intrinsic_down']}")
print(f"  Cooperative up:    {two_bit_thermo['transitions']['cooperative_up']}")
print(f"  Cooperative down:  {two_bit_thermo['transitions']['cooperative_down']}")

print(f"\nBit flips:")
print(f"  0 → 1: {two_bit_thermo['bit_flips']['0_to_1']}")
print(f"  1 → 0: {two_bit_thermo['bit_flips']['1_to_0']}")
print(f"  Net:   {two_bit_thermo['bit_flips']['0_to_1'] - two_bit_thermo['bit_flips']['1_to_0']}")

print(f"\nTape statistics:")
print(f"  φ = {two_bit_stats['phi']:.6f}")
print(f"  ΔS_tape = {two_bit_stats['changes']['delta_entropy']:.6f}")
print(f"  Δ(correlation) = {two_bit_stats['changes']['delta_pair_correlation']:.6f}")

print("\n--- SINGLE-BIT DEMON ---")
print(f"\nEnergy:")
print(f"  Q_h (from hot) = {single_bit_thermo['energy']['Q_h']:.6f}")
print(f"  Q_c (to cold)  = {single_bit_thermo['energy']['Q_c']:.6f}")
print(f"  Q_total        = {single_bit_thermo['energy']['Q_total']:.6f}")

print(f"\nEntropy:")
print(f"  ΔS_h (hot reservoir)   = {single_bit_thermo['entropy']['S_h']:.6f}")
print(f"  ΔS_c (cold reservoir)  = {single_bit_thermo['entropy']['S_c']:.6f}")
print(f"  ΔS_total (production)  = {single_bit_thermo['entropy']['S_total_production']:.6f}")

print(f"\nTransitions:")
print(f"  Intrinsic up:      {single_bit_thermo['transitions']['intrinsic_up']}")
print(f"  Intrinsic down:    {single_bit_thermo['transitions']['intrinsic_down']}")
print(f"  Cooperative up:    {single_bit_thermo['transitions']['cooperative_up']}")
print(f"  Cooperative down:  {single_bit_thermo['transitions']['cooperative_down']}")

print(f"\nBit flips:")
print(f"  0 → 1: {single_bit_thermo['bit_flips']['0_to_1']}")
print(f"  1 → 0: {single_bit_thermo['bit_flips']['1_to_0']}")
print(f"  Net:   {single_bit_thermo['bit_flips']['0_to_1'] - single_bit_thermo['bit_flips']['1_to_0']}")

print(f"\nTape statistics:")
print(f"  φ = {single_bit_stats['phi']:.6f}")
print(f"  ΔS_tape = {single_bit_stats['changes']['delta_entropy']:.6f}")

print("\n" + "=" * 60)
print("SECOND LAW CHECK")
print("=" * 60)
print(f"\nTwo-Bit:    ΔS_total = {two_bit_thermo['entropy']['S_total_production']:.6f} {'✓ ≥ 0' if two_bit_thermo['entropy']['S_total_production'] >= 0 else '✗ < 0'}")
print(f"Single-Bit: ΔS_total = {single_bit_thermo['entropy']['S_total_production']:.6f} {'✓ ≥ 0' if single_bit_thermo['entropy']['S_total_production'] >= 0 else '✗ < 0'}")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
