# Phase 6 Quick Start Guide

## Installation
No installation needed! Just navigate to the phase_6 folder:
```bash
cd phase_6
```

## Quick Test
Run the basic tests to verify everything works:
```bash
python test_basic.py
```

Expected output: All tests pass ✓

## Run Examples
To see the simulation in action with plots:
```bash
python examples.py
```

This will show 6 different examples with matplotlib plots.

## Run Main Simulation
The main file includes 3 example scenarios:
```bash
python Simulation.py
```

## Quick Usage

### Simplest Example (K=3 demons, per-state mode)
```python
from Demon import Demon, PhysParams
from Tape import Tape
from Simulation import StackedDemonSimulation

# Create physics parameters
phys_params = PhysParams(
    sigma=0.3, omega=0.5, DeltaE=1.0, gamma=1.0,
    delta_e_mode='per_state',
    preserve_mode='sigma_omega'
)

# Create 3 demons
demons = [Demon(n=5, phys_params=phys_params) for _ in range(3)]

# Create tape
tape = Tape(N=1000, p0=0.9)

# Run simulation
sim = StackedDemonSimulation(demons=demons, tape=tape, tau=1.0)
final_tape, initial_tape, history = sim.run_full_simulation()

# Get statistics
stats = sim.compute_statistics(final_tape)
print(f"Phi: {stats['phi']:.4f}")
print(f"Bias out: {stats['outgoing']['bias']:.4f}")
```

### Plot Output vs K
```python
from Simulation import plot_output_vs_K
from Demon import PhysParams

plot_output_vs_K(
    K_values=list(range(1, 11)),  # Test K from 1 to 10
    output='phi',                  # Can be: 'phi', 'bias_out', 'Q_c', 'delta_S_b'
    tape_params={"N": 5000, "p0": 1.0},
    demon_n=5,
    tau=1.0,
    phys_params=PhysParams(
        sigma=0.3, omega=0.46, DeltaE=0.5, gamma=1.0,
        delta_e_mode='per_state',
        preserve_mode='sigma_omega'
    )
)
```

## Key Parameters

### PhysParams Options

**Delta_E Mode:**
- `'per_state'`: DeltaE is energy between consecutive states
- `'total'`: DeltaE is total energy from ground to top (requires `demon_n`)

**Preserve Mode:**
- `'sigma_omega'`: Keep σ,ω constant, recalculate T_H,T_C
- `'temperatures'`: Keep T_H,T_C constant, recalculate σ,ω

**Initialization:**
- Using temps: `PhysParams(Th=1.6, Tc=1.0, DeltaE=1.0, gamma=1.0, ...)`
- Using rates: `PhysParams(sigma=0.3, omega=0.5, DeltaE=1.0, gamma=1.0, ...)`

### Output Types for Plotting
- `'phi'`: Bit flip fraction
- `'bias_out'`: Output bias
- `'Q_c'`: Energy transferred
- `'delta_S_b'`: Entropy change

## File Overview
- **Demon.py**: Demon class with enhanced PhysParams
- **Tape.py**: Simple tape class
- **Simulation.py**: Main simulation logic + plotting
- **test_basic.py**: Unit tests
- **examples.py**: Comprehensive examples
- **README.md**: Full documentation
- **SUMMARY.md**: Implementation details

## Need Help?
1. Check README.md for detailed documentation
2. Look at examples.py for usage patterns
3. Run test_basic.py to see what works
4. Read SUMMARY.md for design decisions

## Common Patterns

### Compare Per-State vs Total Mode
```python
# Per-state: Each transition has DeltaE=1.0
phys1 = PhysParams(sigma=0.3, omega=0.5, DeltaE=1.0, gamma=1.0,
                   delta_e_mode='per_state', preserve_mode='sigma_omega')

# Total: Total span is 4.0, so per-state = 4.0/(5-1) = 1.0
phys2 = PhysParams(sigma=0.3, omega=0.5, DeltaE=4.0, gamma=1.0,
                   delta_e_mode='total', preserve_mode='sigma_omega', demon_n=5)

# Both give same per-state energy!
```

### Test Different K Values
```python
for K in [1, 3, 5, 10]:
    demons = [Demon(n=3, phys_params=phys_params) for _ in range(K)]
    sim = StackedDemonSimulation(demons=demons, tape=tape, tau=1.0)
    final_tape, _, _ = sim.run_full_simulation()
    stats = sim.compute_statistics(final_tape)
    print(f"K={K}: phi={stats['phi']:.4f}")
```

Happy simulating! 🎉
