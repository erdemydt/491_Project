"""
Demo for input entropy analysis with p0 sweep from 0 to 0.5
"""

import numpy as np
from Demon import TwoBitDemon, SingleBitDemon, PhysParams
from Tape import TwoBitTape
from Simulation import sweep_p0_input_entropy_analysis

# Physical parameters
phys_params = PhysParams(sigma=0.4, omega=0.6)

# Create a custom two-bit demon
print("=" * 60)
print("Input Entropy Analysis Demo")
print("=" * 60)

custom_two_bit_demon = TwoBitDemon(phys_params=phys_params, init_state='d')
print(f"\nTwo-Bit Demon Transitions:")
for (d_from, b1_from, b2_from), (d_to, b1_to, b2_to) in custom_two_bit_demon.cooperative_transitions.items():
    print(f"  {d_from}{b1_from}{b2_from} ↔ {d_to}{b1_to}{b2_to}")

print(f"\nPhysical Parameters:")
print(f"  σ = {phys_params.sigma:.4f} (intrinsic bias)")
print(f"  ω = {phys_params.omega:.4f} (cooperative bias)")

# Run the analysis
results = sweep_p0_input_entropy_analysis(
    tape_size=4000,
    tau=2.0,
    phys_params=phys_params,
    seed_base=100,
    two_bit_demon=custom_two_bit_demon,
    n_points=11
)

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
