# Phase 10: Two-Bit Demon

## Overview

This phase implements a **two-bit demon** that interacts with pairs of bits simultaneously, as opposed to the standard single-bit demon. This allows us to explore cooperative transitions that depend on the joint state of two bits.

## Key Components

### 1. Demon (`Demon.py`)

#### TwoBitDemon
A demon with two states: **Up (u)** and **Down (d)** that interacts with pairs of bits.

**Allowed Transitions:**
- **Intrinsic transitions** (contact with hot reservoir): `u ↔ d`
- **Cooperative transitions** (contact with cold reservoir):
  - Default: `d00 ↔ u01` and `u10 ↔ d11`
  - Custom transitions can be added via `add_cooperative_transition()`

**Key Features:**
- Configurable cooperative transitions
- Detailed balance preserved
- Transition rates computed from physical parameters (σ, ω)

#### SingleBitDemon
The standard single-bit demon for comparison purposes.

### 2. Tape (`Tape.py`)

#### TwoBitTape
A tape designed for two-bit interactions with special initialization modes:

**Initialization Modes:**
- `'random'`: Independent random bits based on p0
- `'half_split'`: First half all 0s, second half all 1s
- `'half_split_reverse'`: First half all 1s, second half all 0s
- `'alternating'`: Alternating 0s and 1s
- `'pair_distribution'`: Control fraction of "00" and "11" pairs

**Analysis Features:**
- Pair distribution computation
- Intra-pair correlation (correlation between bits in the same pair)
- Inter-pair correlation (correlation between consecutive pairs)
- Mutual information between paired bits
- Entropy (both bit-level and pair-level)

### 3. Simulation (`Simulation.py`)

#### TwoBitDemonSimulation
Processes the tape in pairs using the Gillespie algorithm.

#### SingleBitDemonSimulation  
Processes the tape one bit at a time (for comparison).

#### Comparison Functions
- `compare_demons()`: Run both demons on the same initial tape and compare results
- `sweep_tau()`: Compare demons across different interaction times

## Usage

### Basic Example

```python
from Demon import TwoBitDemon, PhysParams
from Tape import TwoBitTape
from Simulation import TwoBitDemonSimulation, compare_demons

# Set up physical parameters
phys_params = PhysParams(sigma=0.3, omega=0.8)

# Create tape with all zeros
tape = TwoBitTape(N=1000, p0=1.0, init_mode='random')

# Create demon
demon = TwoBitDemon(phys_params=phys_params)

# Run simulation
sim = TwoBitDemonSimulation(demon=demon, tape=tape, tau=2.0)
final_tape, demon_history, stats = sim.run_simulation()

print(f"Flip fraction φ = {stats['phi']:.4f}")
```

### Comparing Demons

```python
# Compare two-bit vs single-bit demon on the same tape
tape_params = {'N': 2000, 'p0': 1.0, 'init_mode': 'random'}
phys_params = PhysParams(sigma=0.3, omega=0.8)

comparison = compare_demons(tape_params, phys_params, tau=2.0, plot=True)
```

### Custom Transitions

```python
# Create demon and add custom cooperative transition
demon = TwoBitDemon(phys_params=phys_params)

# Add: d01 ↔ u11 (when in state d with bits '01', can go to u with bits '11')
demon.add_cooperative_transition('d', '0', '1', 'u', '1', '1')

# View all transitions
demon.print_transition_table()
```

### Controlled Pair Distribution

```python
# Create tape with specific pair distribution
# 50% "00" pairs, 30% "11" pairs, remaining split between "01" and "10"
tape = TwoBitTape(
    N=1000,
    init_mode='pair_distribution',
    pair_00_frac=0.5,
    pair_11_frac=0.3
)

tape.plot_correlation_analysis()
```

## Physical Interpretation

### Default Cooperative Transitions

The default transitions `d00 ↔ u01` and `u10 ↔ d11` create a bias toward increasing the number of 1s:

1. **d00 → u01**: Demon absorbs energy and flips second bit (0→1)
2. **u10 → d11**: Demon releases energy and flips second bit (0→1)

Both transitions preferentially convert 0s to 1s.

### Energy Transfer

- **Q_c**: Energy transferred to the cold reservoir ≈ φ × ΔE
- **φ**: Fraction of bits flipped during the simulation

### Correlation Analysis

The two-bit demon can create or destroy correlations between adjacent bits:
- **Intra-pair correlation**: How correlated are the two bits within each pair?
- **Mutual information**: Information shared between paired bits
- **Δ correlation**: Change in correlation from initial to final tape

## Comparison with Single-Bit Demon

| Aspect | Two-Bit Demon | Single-Bit Demon |
|--------|---------------|------------------|
| Interaction unit | Bit pairs | Individual bits |
| Cooperative transitions | Can involve 2 bits | Always involves 1 bit |
| Correlation creation | Can create pair correlations | No direct pair effects |
| Computational cost | Lower (N/2 interactions) | Higher (N interactions) |

## Files

- `Demon.py`: TwoBitDemon and SingleBitDemon classes
- `Tape.py`: TwoBitTape class with correlation analysis
- `Simulation.py`: Simulation classes and comparison functions
- `Demon.MD`: Original specification document

## Dependencies

- numpy
- matplotlib
- dataclasses (standard library)
