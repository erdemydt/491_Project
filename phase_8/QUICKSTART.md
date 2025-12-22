# Phase 8: Quick Start Guide

## Installation & Setup

No additional dependencies needed beyond phase_6/phase_7 requirements:
- numpy
- matplotlib

## Basic Usage

### 1. Run the Demo

```bash
cd phase_8
python CompetingSimulation.py
```

This will:
- Run a simulation with K=5 demons
- Show bit flip statistics
- Display interaction distribution (how many times each demon won)
- Create a bar plot of demon interactions

### 2. Simple Example

```python
from CompetingDemon import CompetingDemon, PhysParams
from Tape import Tape
from CompetingSimulation import CompetingDemonSimulation

# Create physical parameters
params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)

# Create 5 identical demons with 3 states each
K = 5
demons = [CompetingDemon(n=3, phys_params=params, demon_id=k) 
          for k in range(K)]

# Create tape with 5000 bits, all starting as 0
tape = Tape(N=5000, p0=1.0)

# Run simulation with interaction time τ=1.0
sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=1.0)
final_tape, _, demon_history, wins = sim.run_full_simulation()

# Get statistics
stats = sim.compute_statistics(final_tape)

print(f"Bit flip fraction: {stats['phi']:.4f}")
print(f"Interaction distribution: {stats['interaction_fractions']}")
```

### 3. Plot φ vs K

```python
from CompetingSimulation import plot_output_vs_K
from CompetingDemon import PhysParams

plot_output_vs_K(
    K_values=list(range(1, 21)),
    output='phi',
    tape_params={"N": 5000, "p0": 1.0},
    demon_n=3,
    tau=1.0,
    phys_params=PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
)
```

### 4. Run Analysis Suite

```bash
python analysis.py
```

This runs three analyses:
1. φ vs K with error bars (multiple trials)
2. Interaction distribution heatmap
3. Demon state evolution over tape

### 5. Compare with Phase 7

```bash
python compare_with_phase7.py
```

Shows side-by-side comparison of competitive vs sequential models.

## Parameter Guide

### Physical Parameters (`PhysParams`)

```python
params = PhysParams(
    DeltaE=1.0,   # Energy gap between demon states
    gamma=1.0,    # Hot reservoir coupling
    Th=1.6,       # Hot temperature
    Tc=1.0        # Cold temperature
)
```

**Derived parameters** (computed automatically):
- `sigma = tanh(DeltaE/(2*Th))`: Hot reservoir parameter
- `omega = tanh(DeltaE/(2*Tc))`: Cold reservoir parameter

**Effect on behavior**:
- Higher `Th` → weaker hot reservoir coupling → slower intrinsic transitions
- Higher `Tc` → weaker cold reservoir coupling → fewer bit flips
- Higher `DeltaE` → stronger temperature dependence

### Simulation Parameters

```python
# Number of competing demons
K = 5

# Number of states per demon
demon_n = 3  # States: d0, d1, d2

# Interaction time per bit
tau = 1.0

# Tape parameters
N = 5000     # Number of bits
p0 = 1.0     # Probability of initial bit = 0
```

**Effect on behavior**:
- Higher `K` → more competition → more asymmetric interactions
- Higher `n` → more states → more complex dynamics
- Higher `tau` → longer interactions → more state changes
- Higher `p0` → more 0's initially → more potential to flip to 1

## Output Metrics

### Primary Outputs

- **φ (phi)**: Bit flip fraction = (# of bits that changed) / N
- **Bias out**: p₀ - p₁ in final tape
- **ΔS_B**: Entropy change of bit tape
- **Q_c**: Energy transferred to cold reservoir = φ × ΔE

### Interaction Statistics

- **interaction_counts**: Array of length K showing # wins per demon
- **interaction_fractions**: Normalized version (sums to 1.0)
- **demon_states_history**: List of K lists, each showing state trajectory

### Interpretation

**φ ≈ 0**: Few bits flipped
- Demons not effective
- Check if τ too small or temperatures wrong

**φ ≈ 0.5**: Maximum flipping
- Optimal demon operation
- Good energy extraction

**φ > 0.5**: More than half flipped
- Very effective demons
- Check if p0 and parameters allow this

**Unequal interaction_fractions**: Expected!
- Demons in different states compete differently
- High-rate states win more often

## Common Experiments

### Experiment 1: Effect of K

```python
# How does # of demons affect performance?
K_values = [1, 2, 3, 5, 7, 10, 15, 20]
plot_output_vs_K(K_values, output='phi', ...)
```

**Expected**: φ might increase then plateau as K grows.

### Experiment 2: Effect of τ

```python
# How does interaction time affect competition?
tau_values = np.logspace(-1, 1, 20)  # 0.1 to 10

results = []
for tau in tau_values:
    # ... run simulation with this tau
    results.append(phi)

plt.plot(tau_values, results)
plt.xlabel('τ')
plt.ylabel('φ')
```

**Expected**: φ increases with τ, then saturates.

### Experiment 3: State Distribution

```python
# Do demons separate into different states?
from analysis import analyze_demon_state_evolution

analyze_demon_state_evolution(K=10, demon_n=5, tau=2.0)
```

**Look for**: 
- Do demons cluster in certain states?
- Do winners tend to be in specific states?

### Experiment 4: Parameter Scan

```python
# Scan σ and ω
sigma_values = np.linspace(0.1, 0.9, 10)
omega_values = np.linspace(0.1, 0.9, 10)

results = np.zeros((len(sigma_values), len(omega_values)))

for i, sigma in enumerate(sigma_values):
    for j, omega in enumerate(omega_values):
        # Create params with desired sigma, omega
        # Need to back-calculate Th, Tc from sigma, omega
        # ...
        results[i, j] = phi

plt.imshow(results, extent=[...])
plt.xlabel('ω')
plt.ylabel('σ')
```

## Troubleshooting

### Issue: All demons win equally

**Possible causes**:
- τ too large → all demons equilibrate to same state distribution
- n=2 → limited state space, less competition
- Initial states identical and no divergence yet

**Solutions**:
- Reduce τ
- Increase n
- Run longer tape (larger N)
- Check if rates are correctly computed

### Issue: One demon wins everything

**Possible causes**:
- That demon stuck in very high-rate state
- Other demons stuck in low-rate states
- Positive feedback too strong

**Solutions**:
- Check demon state histories
- Verify transition rates are reasonable
- May be interesting phenomenon!

### Issue: φ ≈ 0

**Possible causes**:
- τ too small
- Temperatures wrong
- σ or ω near 0

**Solutions**:
- Increase τ
- Check temperature calculations
- Verify σ = tanh(ΔE/(2Th)) is reasonable (0.2-0.8 range)

### Issue: φ seems independent of K

**Possible causes**:
- Competition not effective in current parameter regime
- τ so small that state changes rare
- All demons in same state always

**Solutions**:
- Increase n (more states = more competition)
- Check interaction distribution
- Try different τ range

## Files Overview

```
phase_8/
├── CompetingDemon.py          # Demon class
├── Tape.py                     # Tape class
├── CompetingSimulation.py      # Main simulation + plotting
├── analysis.py                 # Advanced analysis tools
├── compare_with_phase7.py      # Phase 7 vs Phase 8 comparison
├── README.md                   # Full documentation
├── CONCEPTS.md                 # Theoretical concepts
└── QUICKSTART.md              # This file
```

## Next Steps

1. **Run demo**: `python CompetingSimulation.py`
2. **Try different K**: Uncomment plot_output_vs_K in main
3. **Run analysis**: `python analysis.py`
4. **Explore parameters**: Modify physical parameters and observe
5. **Compare models**: `python compare_with_phase7.py`

## Questions to Explore

1. Is there an optimal K for maximum φ?
2. How does interaction distribution depend on demon_n?
3. Can we predict which demon will win based on its state?
4. What's the relationship between σ, ω and interaction inequality?
5. Does competitive model extract more energy than sequential?

Happy exploring! 🚀
