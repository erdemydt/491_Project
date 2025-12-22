# Stacked Demon Simulation (Phase 7)

## Overview

This implementation features a **stacked demon** that interacts with **2 bits at a time** (bit pairs) instead of individual bits. The demon is restricted to **3 states**: low (d0), medium (d1), and high (d2).

## Key Concepts

### Demon States
- **d0 (low)**: Ground state, energy = 0
- **d1 (medium)**: First excited state, energy = ΔE₁
- **d2 (high)**: Second excited state, energy = ΔE₁ + ΔE₂

### Bit Pair States
The demon processes 2 bits at a time:
- **00**: Both bits are 0
- **01**: First bit 0, second bit 1
- **10**: First bit 1, second bit 0
- **11**: Both bits are 1

### Transition Rules

#### Intrinsic Transitions (Hot Reservoir)
The demon exchanges energy with the hot reservoir without changing the bits:
- `XX_d0 ↔ XX_d1` (energy gap ΔE₁)
- `XX_d1 ↔ XX_d2` (energy gap ΔE₂)

#### Outgoing Transitions (Cold Reservoir + Bit Flips)
The demon can flip bits while exchanging energy with the cold reservoir:

**Single bit flips (d0 ↔ d1):**
- `00_d0 → 01_d1` or `00_d0 → 10_d1` (demon gains ΔE₁, one bit flips 0→1)
- `01_d1 → 00_d0` or `10_d1 → 00_d0` (demon loses ΔE₁, one bit flips 1→0)

**Single bit flips (d1 ↔ d2):**
- `01_d1 → 11_d2` or `10_d1 → 11_d2` (demon gains ΔE₂, one bit flips 0→1)
- `11_d2 → 01_d1` or `11_d2 → 10_d1` (demon loses ΔE₂, one bit flips 1→0)

**Double bit flips (d0 ↔ d2):**
- `00_d0 → 11_d2` (demon gains ΔE₁ + ΔE₂, both bits flip 0→1)
- `11_d2 → 00_d0` (demon loses ΔE₁ + ΔE₂, both bits flip 1→0)

### Transition Rates

**Hot reservoir rates** (intrinsic transitions):
```
r(dᵢ → dⱼ) = γ exp(-ΔEᵢⱼ / (2Tₕ))
```

**Cold reservoir rates** (outgoing transitions):
- For transitions involving ΔEₖ:
  - Upward (0→1 flip): `1 - ωₖ` where `ωₖ = tanh(ΔEₖ / (2Tс))`
  - Downward (1→0 flip): `1 + ωₖ`
- For double flips: multiply the individual probabilities

## Files

### `StackedDemon.py`
Defines the `StackedDemon` class with:
- 3 energy states (d0, d1, d2)
- Precomputed transition rates for all transitions
- Physical parameters via `StackedPhysParams`

### `StackedTape.py`
Defines the `StackedTape` class that:
- Stores bits but processes them in pairs
- Analyzes pair distributions
- Computes both bit-level and pair-level entropy

### `StackedSimulation.py`
Main simulation engine using the Gillespie algorithm:
- Processes tape in consecutive bit pairs
- Tracks demon state evolution
- Computes comprehensive statistics

## Usage

### Basic Example

```python
from StackedDemon import StackedDemon, StackedPhysParams
from StackedTape import StackedTape
from StackedSimulation import StackedSimulation

# Define physical parameters
phys_params = StackedPhysParams(
    DeltaE_1=0.5,  # Energy gap d0→d1
    DeltaE_2=0.5,  # Energy gap d1→d2
    gamma=1.0,     # Hot reservoir rate
    Th=2.0,        # Hot temperature
    Tc=1.0         # Cold temperature
)

# Create tape with 1000 bits (500 pairs)
tape = StackedTape(N=1000, p0=0.9)

# Create demon
demon = StackedDemon(phys_params=phys_params)

# Run simulation with interaction time τ=5.0 per pair
sim = StackedSimulation(demon, tape, tau=5.0)
input_tape, output_tape, demon_seq = sim.run_full_simulation()

# Analyze results
stats = sim.compute_statistics(input_tape, output_tape)
print(f"Bit flip fraction: {stats['phi']:.4f}")
print(f"Pair change fraction: {stats['pair_flip_fraction']:.4f}")
```

### Running the Demo

```bash
python phase_7/StackedSimulation.py
```

This runs a complete demonstration showing:
- Input/output tape analysis
- Bit-level and pair-level statistics
- Demon state distribution
- Entropy changes

### Plotting φ vs τ

Uncomment the code at the bottom of `StackedSimulation.py`:

```python
plot_phi_vs_tau(
    tau_values=np.linspace(0.1, 20.0, 20).tolist(),
    tape_params={'N': 1000, 'p0': 0.9},
    phys_params=StackedPhysParams(DeltaE_1=0.5, DeltaE_2=0.5, gamma=1.0, Th=2.0, Tc=1.0)
)
```

## Key Differences from Single-Bit Demon

1. **Grouped Processing**: Bits are processed in pairs instead of individually
2. **Fixed 3 States**: Demon always has exactly 3 states (not n states)
3. **Richer Transitions**: Can have single-bit flips or double-bit flips
4. **Pair Statistics**: Can analyze correlations between adjacent bits
5. **Two Energy Gaps**: ΔE₁ and ΔE₂ can be different

## Output Statistics

The simulation tracks:
- **φ**: Bit flip fraction (individual bits changed)
- **Pair flip fraction**: Fraction of bit pairs that changed
- **Bias in/out**: p₀ - p₁ for input and output
- **Entropy change**: Both bit-level and pair-level
- **Demon state distribution**: Time spent in d0, d1, d2
- **Pair distributions**: Frequencies of 00, 01, 10, 11

## Physical Interpretation

This model can represent scenarios where:
- The demon has **quantized energy levels** with specific gaps
- Information is processed in **chunks** (pairs) rather than individually
- There are **multiple energy scales** in the system (ΔE₁ vs ΔE₂)
- Correlations between adjacent bits matter

## Example Results

For typical parameters (p₀=0.9, τ=5.0, ΔE₁=ΔE₂=0.5, Th=2.0, Tc=1.0):
- Bit flip fraction: ~0.40
- Pair change fraction: ~0.64
- Entropy increases significantly
- Demon spends roughly equal time in all 3 states

The higher pair change fraction compared to bit flip fraction indicates that when pairs change, often only one bit flips rather than both.
