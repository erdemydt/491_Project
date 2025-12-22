# Phase 9: Parameter Sweep Analysis

This phase explores how the Maxwell's demon system responds to different physical parameter values, specifically **sigma (σ)** and **omega (ω)**.

## Parameters

### Sigma (σ)
- **Physical meaning**: Related to the hot reservoir temperature (Th)
- **Formula**: σ = tanh(ΔE / (2·Th))
- **Range**: (-1, 1)
- **Controls**: Intrinsic demon transitions between energy states
- **Higher σ**: Cooler hot reservoir, slower intrinsic transitions

### Omega (ω)
- **Physical meaning**: Related to the cold reservoir temperature (Tc)
- **Formula**: ω = tanh(ΔE / (2·Tc))
- **Range**: (-1, 1)
- **Controls**: Bit-flip transitions coupled with demon energy changes
- **Higher ω**: Cooler cold reservoir, stronger coupling to bit flips

## Key Questions

1. **Does the system behave consistently across different (σ, ω) combinations?**
2. **Are there optimal parameter ranges for maximum bit flipping (φ)?**
3. **How does energy transfer (Q_c) vary with parameters?**
4. **What is the relationship between σ, ω and entropy change?**

## Usage

### Basic Parameter Sweep

```python
from parameter_sweep import ParameterSweepSimulation, plot_1d_sweep

# Initialize simulator
sim = ParameterSweepSimulation(
    demon_n=2,    # Number of demon states
    tau=1.0,      # Interaction time
    N=5000,       # Number of bits
    p0=1.0        # Initial probability of 0
)

# Sweep sigma with fixed omega
import numpy as np
sigma_values = np.linspace(0.1, 0.9, 9)
results = sim.sweep_sigma_fixed_omega(
    sigma_values=sigma_values,
    omega_fixed=0.8
)

# Plot results
plot_1d_sweep(results, sweep_param='sigma', output_keys=['phi', 'Q_c'])
```

### 2D Grid Sweep

```python
from parameter_sweep import plot_heatmap

# Sweep both parameters
sigma_grid = np.linspace(0.2, 0.8, 10)
omega_grid = np.linspace(0.2, 0.8, 10)
results = sim.sweep_sigma_omega_grid(
    sigma_values=sigma_grid,
    omega_values=omega_grid
)

# Visualize as heatmap
plot_heatmap(results, output_key='phi', 
             title='Bit Flip Fraction vs σ and ω')
```

## Output Metrics

Each simulation returns:
- **φ (phi)**: Fraction of bits flipped (p₁_out - p₁_in)
- **Q_c**: Total energy transferred to cold reservoir
- **ΔS_B**: Entropy change of the bit tape
- **bias**: Difference between p₀ and p₁
- **Th, Tc**: Effective reservoir temperatures

## Files

- `parameter_sweep.py`: Main simulation and analysis code
- `README.md`: This documentation
- `plots/`: Generated visualizations (created automatically)

## Dependencies

This phase reuses infrastructure from `phase_6`:
- `Demon.py`: Demon class with physical parameters
- `Tape.py`: Tape class for bit strings
- `Simulation.py`: StackedDemonSimulation class

## Expected Behavior

Based on thermodynamic principles:
- **Low σ, High ω**: Maximum bit flipping (large temperature difference)
- **High σ, Low ω**: Minimal bit flipping (small temperature difference)
- **σ ≈ ω**: Near-equilibrium behavior
- **Energy transfer should scale with (ω - σ) roughly**
