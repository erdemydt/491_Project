"""
Comprehensive examples for phase_6 stacked demon simulation.

This file demonstrates various use cases and configurations.
"""
from Simulation import plot_output_vs_K
from Demon import PhysParams
import numpy as np

print("=" * 70)
print("PHASE 6 COMPREHENSIVE EXAMPLES")
print("=" * 70)

# =============================================================================
# EXAMPLE 1: Standard per-state mode
# =============================================================================
print("\n[EXAMPLE 1] Standard Per-State Mode")
print("Each demon state transition has fixed DeltaE = 0.5")
print("Sigma and omega define reservoir interaction strengths")
print("-" * 70)

plot_output_vs_K(
    K_values=list(range(1, 8)),
    output='phi',
    tape_params={"N": 3000, "p0": 1.0},
    demon_n=5,
    tau=1.0,
    phys_params=PhysParams(
        sigma=0.3, 
        omega=0.46, 
        DeltaE=0.5, 
        gamma=1.0,
        delta_e_mode='per_state',
        preserve_mode='sigma_omega'
    )
)

# =============================================================================
# EXAMPLE 2: Total mode with fixed temperatures
# =============================================================================
print("\n[EXAMPLE 2] Total DeltaE Mode - Fixed Temperatures")
print("Total energy from ground to top = 2.0")
print("Temperatures Th=1.6, Tc=1.0 are preserved")
print("Sigma and omega will adjust based on demon_n")
print("-" * 70)

plot_output_vs_K(
    K_values=list(range(1, 8)),
    output='phi',
    tape_params={"N": 3000, "p0": 1.0},
    demon_n=5,
    tau=1.0,
    phys_params=PhysParams(
        Th=1.6,
        Tc=1.0,
        DeltaE=2.0,  # Total energy span
        gamma=1.0,
        delta_e_mode='total',
        preserve_mode='temperatures',
        demon_n=5
    )
)

# =============================================================================
# EXAMPLE 3: Total mode with fixed sigma/omega
# =============================================================================
print("\n[EXAMPLE 3] Total DeltaE Mode - Fixed Sigma/Omega")
print("Total energy from ground to top = 3.0")
print("Sigma=0.2, Omega=0.7 are preserved")
print("Temperatures will adjust based on demon_n")
print("-" * 70)

plot_output_vs_K(
    K_values=list(range(1, 8)),
    output='phi',
    tape_params={"N": 3000, "p0": 1.0},
    demon_n=5,
    tau=1.0,
    phys_params=PhysParams(
        sigma=0.2,
        omega=0.7,
        DeltaE=3.0,  # Total energy span
        gamma=1.0,
        delta_e_mode='total',
        preserve_mode='sigma_omega',
        demon_n=5
    )
)

# =============================================================================
# EXAMPLE 4: Bias out vs K (non-uniform input tape)
# =============================================================================
print("\n[EXAMPLE 4] Bias Out vs K")
print("Input tape has p0=0.8 (bias in = 0.6)")
print("Observing how bias changes with multiple demons")
print("-" * 70)

plot_output_vs_K(
    K_values=list(range(1, 10)),
    output='bias_out',
    tape_params={"N": 3000, "p0": 0.8},
    demon_n=3,
    tau=2.0,
    phys_params=PhysParams(
        sigma=0.3,
        omega=0.6,
        DeltaE=1.0,
        gamma=1.0,
        delta_e_mode='per_state',
        preserve_mode='sigma_omega'
    )
)

# =============================================================================
# EXAMPLE 5: Energy transfer vs K
# =============================================================================
print("\n[EXAMPLE 5] Energy Transfer (Q_c) vs K")
print("Total energy transferred to cold reservoir")
print("Shows cumulative effect of stacked demons")
print("-" * 70)

plot_output_vs_K(
    K_values=list(range(1, 10)),
    output='Q_c',
    tape_params={"N": 3000, "p0": 0.9},
    demon_n=4,
    tau=1.5,
    phys_params=PhysParams(
        sigma=0.25,
        omega=0.55,
        DeltaE=0.8,
        gamma=1.0,
        delta_e_mode='per_state',
        preserve_mode='sigma_omega'
    )
)

# =============================================================================
# EXAMPLE 6: Entropy change vs K
# =============================================================================
print("\n[EXAMPLE 6] Entropy Change (ΔS_B) vs K")
print("Bit tape entropy change through stacked demons")
print("-" * 70)

plot_output_vs_K(
    K_values=list(range(1, 10)),
    output='delta_S_b',
    tape_params={"N": 3000, "p0": 0.95},
    demon_n=4,
    tau=1.0,
    phys_params=PhysParams(
        sigma=0.3,
        omega=0.5,
        DeltaE=1.0,
        gamma=1.0,
        delta_e_mode='per_state',
        preserve_mode='sigma_omega'
    )
)

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED!")
print("=" * 70)
