# SmartTape: Correlated Bit Tapes for Demon Simulations

## Overview

`SmartTape` is an enhanced version of the `Tape` class that supports creating tapes with correlated bits and provides comprehensive correlation analysis tools.

## Features

### 1. Correlation Types

#### None (Independent Bits)
```python
tape = SmartTape(N=1000, p0=0.7, correlation_type='none')
```
- Default behavior, same as original `Tape` class
- Each bit is independently sampled

#### Markov Chain Correlation
```python
tape = SmartTape(N=1000, p0=0.7, correlation_type='markov', 
                 correlation_strength=0.7)
```
- First-order Markov chain where each bit depends on the previous bit
- `correlation_strength`: 0.0 = independent, 1.0 = maximum correlation
- Creates smooth transitions between bit values

#### Block Correlation
```python
tape = SmartTape(N=1000, p0=0.7, correlation_type='block', 
                 correlation_strength=0.8, block_size=10)
```
- Creates blocks/runs of identical bits
- `correlation_strength`: controls average block size
- `block_size`: optional parameter for explicit average block size
- Good for testing behavior with clustered patterns

#### Periodic Correlation
```python
tape = SmartTape(N=1000, p0=0.7, correlation_type='periodic', 
                 correlation_strength=0.9, period=20)
```
- Creates repeating patterns with optional noise
- `correlation_strength`: 1.0 = perfect periodicity, 0.0 = random
- `period`: length of the repeating pattern
- Useful for studying periodic signal processing

### 2. Correlation Analysis Methods

#### Get Correlation Summary
```python
summary = tape.get_correlation_summary()
print(summary['nearest_neighbor_correlation'])  # Correlation between adjacent bits
print(summary['mean_block_length'])             # Average run length
print(summary['mutual_information_lag1'])       # Information sharing
```

Returns comprehensive dictionary with:
- `nearest_neighbor_correlation`: Pearson correlation of adjacent bits
- `autocorr_lag1`, `autocorr_lag5`: Autocorrelation at different lags
- `mutual_information_lag1`: Mutual information between adjacent bits
- `mean_block_length`, `max_block_length`, `n_blocks`: Block statistics
- `autocorrelation_full`: Full autocorrelation function (lags, values)

#### Compute Autocorrelation
```python
lags, autocorr = tape.compute_autocorrelation(max_lag=50)
```
- Returns autocorrelation function up to specified lag
- Useful for identifying correlation length scales

#### Nearest Neighbor Correlation
```python
nn_corr = tape.compute_nearest_neighbor_correlation()
```
- Single number summarizing adjacent bit correlation
- Range: -1 (anti-correlated) to +1 (perfectly correlated)

#### Block Statistics
```python
block_stats = tape.compute_block_statistics()
print(block_stats['mean_block_length'])
print(block_stats['max_block_length'])
print(block_stats['n_blocks'])
```
- Analyzes runs of identical bits
- Useful for understanding clustering

#### Mutual Information
```python
mi = tape.compute_mutual_information(lag=1)
```
- Information-theoretic measure of dependence
- Non-negative (0 = independent)

### 3. Visualization

#### Comprehensive Correlation Analysis Plot
```python
tape.plot_correlation_analysis(max_lag=50)
```

Creates a 2×2 plot showing:
1. Tape bit pattern visualization
2. Autocorrelation function
3. Block length distribution
4. Summary statistics

## Example Workflows

### Basic Usage
```python
from Tape import SmartTape

# Create a correlated tape
tape = SmartTape(N=5000, p0=0.7, correlation_type='markov', 
                 correlation_strength=0.6, seed=42)

# Analyze correlations
summary = tape.get_correlation_summary()
print(f"NN correlation: {summary['nearest_neighbor_correlation']:.4f}")

# Visualize
tape.plot_correlation_analysis()
```

### Comparing Different Correlations
```python
# Create tapes with different correlations
tapes = {
    'uncorrelated': SmartTape(N=1000, p0=0.7, correlation_type='none'),
    'weak_markov': SmartTape(N=1000, p0=0.7, correlation_type='markov', 
                             correlation_strength=0.3),
    'strong_markov': SmartTape(N=1000, p0=0.7, correlation_type='markov', 
                               correlation_strength=0.9),
}

for name, tape in tapes.items():
    summary = tape.get_correlation_summary()
    print(f"{name}: NN_corr={summary['nearest_neighbor_correlation']:.4f}")
```

### Testing Simulation with Correlations
```python
from Tape import SmartTape
from Demon import Demon, PhysParams
from Simulation import StackedDemonSimulation

# Create correlated input tape
input_tape = SmartTape(N=5000, p0=1.0, correlation_type='markov', 
                       correlation_strength=0.7, seed=42)

# Analyze initial correlations
initial_summary = input_tape.get_correlation_summary()
print(f"Initial NN correlation: {initial_summary['nearest_neighbor_correlation']:.4f}")

# Run simulation
demon = Demon(n=3, phys_params=PhysParams(sigma=0.3, omega=0.8))
sim = StackedDemonSimulation(demons=[demon], tape=input_tape, tau=1.0)
final_tape_obj, _, _ = sim.run_full_simulation()

# Convert output to SmartTape for analysis
output_tape = SmartTape(N=5000, p0=0.5, tape_arr=final_tape_obj.tape_arr)

# Analyze final correlations
final_summary = output_tape.get_correlation_summary()
print(f"Final NN correlation: {final_summary['nearest_neighbor_correlation']:.4f}")

# Compare
print(f"Correlation change: {final_summary['nearest_neighbor_correlation'] - initial_summary['nearest_neighbor_correlation']:.4f}")
```

## Demonstration Scripts

### `smarttape_demo.py`
Quick demonstration of basic SmartTape features:
```bash
python phase_6/smarttape_demo.py
```

### `test_smarttape.py`
Comprehensive tests including:
- All correlation types demonstrated
- Visual comparisons of correlation patterns
- Simulation with correlated tapes
- Analysis of how correlations affect demon behavior

```bash
python phase_6/test_smarttape.py
```

## Understanding Correlation Metrics

### Nearest Neighbor Correlation
- **Range**: -1 to +1
- **Interpretation**: 
  - 0: No correlation (independent bits)
  - +1: Perfect positive correlation (bits tend to be the same)
  - -1: Perfect negative correlation (bits tend to alternate)
- **When to use**: Quick single-number summary

### Autocorrelation Function
- **Range**: -1 to +1 for each lag
- **Interpretation**: How much a bit at position `i` correlates with bit at position `i+lag`
- **When to use**: Understanding correlation decay over distance

### Mutual Information
- **Range**: 0 to ∞ (typically 0 to ~0.5 for binary data)
- **Interpretation**: How much knowing one bit tells you about another
- **When to use**: Information-theoretic analysis, captures non-linear dependence

### Mean Block Length
- **Range**: 1 to N
- **Interpretation**: Average length of runs of identical bits
- **When to use**: Understanding clustering/grouping behavior
- **Note**: Independent bits with p0=0.5 have mean block length ≈ 2

## Key Findings

From `test_smarttape.py` experiments:

1. **Markov correlation** creates smooth, gradually changing patterns
   - Correlation strength directly controls nearest-neighbor correlation
   - Good for modeling physical systems with memory

2. **Block correlation** creates distinct clusters
   - Very high nearest-neighbor correlation even at moderate strength
   - Useful for studying rare events or burst phenomena

3. **Periodic correlation** creates repeating structures
   - Can have low nearest-neighbor correlation but high correlation at period
   - Useful for studying resonance effects

4. **Demon simulation** tends to:
   - Reduce correlations (decorrelating effect)
   - Break up large blocks
   - Output becomes more random/independent

## Future Extensions

Potential additions to SmartTape:
- Long-range correlation (power-law decay)
- Multi-scale correlations (hierarchical structure)
- Conditional correlations (context-dependent)
- Time-varying correlation patterns
- Cross-correlation between multiple tapes

## Notes

- All `SmartTape` instances are fully compatible with existing `Simulation` code
- The correlation analysis methods can be applied to any tape, even those created by simulations
- Correlation metrics are computed efficiently even for large tapes (N ~ 10^5)
- Seeds are used consistently for reproducibility

## References

- Autocorrelation: Standard statistical time-series analysis
- Mutual Information: Information theory (Shannon, 1948)
- Block statistics: Run-length encoding and clustering analysis
