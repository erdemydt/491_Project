# Phase 6: Stacked Demons Simulation

This phase implements an improved simulation where bits interact sequentially with K demons stacked on top of each other.

## Key Features

### 1. Stacked Demons (K demons)
- Each bit interacts with K demons sequentially
- After passing through all K demons, we get the output bit
- All demons have the same number of states (n)
- Each demon maintains its own state throughout the simulation

### 2. Delta_E Configuration Modes

The `PhysParams` class now supports two modes for interpreting `DeltaE`:

#### **Per-State Mode** (`delta_e_mode='per_state'`)
- `DeltaE` represents the energy difference between consecutive demon states
- This is the traditional interpretation from phase_4
- Example: If `DeltaE=1.0` and `n=5`, then each transition (d0→d1, d1→d2, etc.) has energy 1.0

#### **Total Mode** (`delta_e_mode='total'`)
- `DeltaE` represents the total energy difference from ground state (d0) to top state (d_{n-1})
- The energy per state transition is automatically calculated as `DeltaE / (n-1)`
- Requires `demon_n` parameter to be specified
- Example: If `DeltaE=4.0` and `n=5`, then each transition has energy 4.0/(5-1) = 1.0

### 3. Preserve Mode Options

When using `delta_e_mode='total'`, you can choose what to preserve:

#### **Preserve Sigma/Omega** (`preserve_mode='sigma_omega'`)
- Keeps the transition parameters σ and ω constant
- Recalculates temperatures T_H and T_C based on the new per-state energy
- Use this when you want to maintain the same transition behavior regardless of demon size

#### **Preserve Temperatures** (`preserve_mode='temperatures'`)
- Keeps temperatures T_H and T_C constant
- Recalculates σ and ω based on the new per-state energy
- Use this when the physical reservoirs have fixed temperatures

## File Structure

```
phase_6/
├── Demon.py          # Enhanced Demon class with new PhysParams
├── Tape.py           # Simple Tape class (streamlined from phase_4)
├── Simulation.py     # StackedDemonSimulation class and plotting
└── README.md         # This file
```

## Usage Examples

### Example 1: Per-State Mode with Sigma/Omega
```python
from Simulation import plot_output_vs_K
from Demon import PhysParams

# Each demon state transition has DeltaE=0.5
# Sigma and omega define the reservoir interactions
plot_output_vs_K(
    K_values=list(range(1, 11)),
    output='phi',
    tape_params={"N": 5000, "p0": 1.0},
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
```

### Example 2: Total Mode with Fixed Temperatures
```python
# Total energy from ground to top is DeltaE=2.0
# Temperatures are fixed at Th=1.6, Tc=1.0
plot_output_vs_K(
    K_values=list(range(1, 11)),
    output='phi',
    tape_params={"N": 5000, "p0": 1.0},
    demon_n=5,
    tau=1.0,
    phys_params=PhysParams(
        Th=1.6,
        Tc=1.0,
        DeltaE=2.0,
        gamma=1.0,
        delta_e_mode='total',
        preserve_mode='temperatures',
        demon_n=5  # Required for total mode
    )
)
```

### Example 3: Total Mode with Fixed Sigma/Omega
```python
# Total energy is 3.0, sigma/omega stay constant
# Temperatures will be recalculated based on demon_n
plot_output_vs_K(
    K_values=list(range(1, 11)),
    output='bias_out',
    tape_params={"N": 5000, "p0": 0.8},
    demon_n=3,
    tau=2.0,
    phys_params=PhysParams(
        sigma=0.2,
        omega=0.7,
        DeltaE=3.0,
        gamma=1.0,
        delta_e_mode='total',
        preserve_mode='sigma_omega',
        demon_n=3  # Required for total mode
    )
)
```

## Output Options

The `plot_output_vs_K` function supports these outputs:
- `'phi'`: Bit flip fraction (p'₁ - p₁)
- `'bias_out'`: Output bias (p₀ - p₁)
- `'Q_c'`: Energy transferred to cold reservoir
- `'delta_S_b'`: Entropy change of the bit tape

## Class: StackedDemonSimulation

### Constructor
```python
StackedDemonSimulation(demons: List[Demon], tape: Tape, tau: float)
```

### Key Methods

- `run_full_simulation()`: Process entire tape through K demons
  - Returns: (final_tape, initial_tape, demon_states_history)
  
- `compute_statistics(final_tape)`: Calculate output statistics
  - Returns: Dictionary with phi, bias, Q_c, delta_S_b, etc.

- `process_bit_through_demons(bit_value, demon_states)`: Process one bit through all K demons
  - Returns: (final_bit, updated_demon_states)

## Class: PhysParams (Enhanced)

### New Parameters
- `delta_e_mode`: `'per_state'` or `'total'`
- `preserve_mode`: `'sigma_omega'` or `'temperatures'`
- `demon_n`: Number of demon states (required for `'total'` mode)

### Methods
- `get_delta_e_per_state()`: Get energy per state transition
- `recalculate_for_new_demon_n(new_n)`: Create new params for different demon size

## Running the Examples

```bash
cd phase_6
python Simulation.py
```

This will run three examples showing different configuration modes.

## Self-Reliance

This folder is completely self-contained with its own:
- Demon class
- Tape class  
- Simulation class
- All necessary imports

No dependencies on other phase folders!
