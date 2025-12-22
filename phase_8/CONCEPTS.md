# Phase 8: Competitive Demons - Key Concepts

## The Core Difference: Competition vs Sequential

### Phase 7: Sequential Interaction
```
Bit 0 → Demon 0 (τ) → Demon 1 (τ) → ... → Demon K-1 (τ) → Output Bit 0
Bit 1 → Demon 0 (τ) → Demon 1 (τ) → ... → Demon K-1 (τ) → Output Bit 1
...
```
- Each demon sees every bit
- Total interaction time per bit: K × τ
- All demons change state for each bit

### Phase 8: Competitive Interaction
```
For Bit 0:
  - Demon 0 in state d_i → total_rate_0 → dt_0 ~ Exp(1/total_rate_0)
  - Demon 1 in state d_j → total_rate_1 → dt_1 ~ Exp(1/total_rate_1)
  - ...
  - Demon K-1 in state d_k → total_rate_{K-1} → dt_{K-1} ~ Exp(1/total_rate_{K-1})
  
  Winner = argmin(dt_0, dt_1, ..., dt_{K-1})
  
  Only winner interacts with bit for time τ
  Only winner's state changes
```
- Only one demon interacts per bit
- Total interaction time per bit: τ
- Only winning demon changes state
- Demons in different states have different winning probabilities

## Why This Is Fundamentally Different

### 1. State Persistence
**Sequential (Phase 7)**: All demons process every bit, so their states are highly correlated.

**Competitive (Phase 8)**: Demons that don't win stay in the same state, creating state diversity.

### 2. Interaction Probability
**Sequential (Phase 7)**: Every demon interacts with equal frequency (100% of bits).

**Competitive (Phase 8)**: Demons with higher total transition rates win more often.

### 3. Time Scales
**Sequential (Phase 7)**: Total time = K × τ × N

**Competitive (Phase 8)**: Total time = τ × N

### 4. State-Dependent Dynamics
**Competitive model** has emergent behavior:
- A demon in a high-energy state (many possible transitions) has high total_rate
- High total_rate → shorter average dt
- Shorter dt → higher winning probability
- More wins → more opportunities to change state

This creates a **feedback loop** between demon state and interaction probability!

## Mathematical Detail: Why dt Competition Works

For a demon in joint state (bit, demon_state):

1. **Total rate** = sum of all possible transition rates
   ```
   total_rate = Σ(intrinsic rates) + Σ(outgoing rates)
   ```

2. **Time to next event** follows exponential distribution:
   ```
   dt ~ Exponential(λ = total_rate)
   ```
   
   Mean: E[dt] = 1/total_rate
   
   Higher total_rate → smaller average dt → higher winning probability

3. **Competition**: 
   ```
   P(demon k wins | all demon states) = P(dt_k < min(dt_j for j ≠ k))
   ```
   
   For exponential distributions, this has a closed form:
   ```
   P(demon k wins) = total_rate_k / Σ(total_rate_j for all j)
   ```

## Example: State-Dependent Winning

Consider K=2 demons, n=3 states:

**Scenario 1**: Both demons in d0
- Both have same total_rate
- Equal probability of winning (50%-50%)
- Symmetric interaction

**Scenario 2**: Demon 0 in d0, Demon 1 in d2
- Demon in d2 has more possible transitions (can go to d1, can interact with bits)
- Higher total_rate → higher winning probability
- Asymmetric interaction

**Scenario 3**: After many bits
- Demons evolve to different states
- State distribution reflects past interactions
- Interaction probabilities continuously adjust

## Key Questions to Explore

1. **Does competition lead to state separation?**
   - Do demons tend to occupy different states over time?
   - Is there a stable state distribution?

2. **Is there an optimal K?**
   - Does adding more demons improve performance (higher φ)?
   - Or does competition reduce efficiency?

3. **How does τ affect competition?**
   - Small τ: Less time to change state during interaction → "frozen" competition
   - Large τ: More state exploration → dynamic competition

4. **What about non-uniform initial states?**
   - If demons start in different states, how does this affect outcomes?

5. **Energy efficiency**
   - Does competitive model extract more/less energy than sequential?
   - Is there a relationship between interaction distribution and Q_c?

## Expected Phenomena

### Phenomenon 1: Interaction Inequality
Even with identical demons, we expect **unequal interaction counts** due to:
- Random state divergence
- State-dependent transition rates
- Positive feedback (high-rate states win more → explore more)

### Phenomenon 2: K Saturation
As K increases, we might see diminishing returns:
- For K=1: Single demon, baseline performance
- For K=2-5: Increased competition, possibly higher φ
- For K>>1: Too much competition, most demons rarely interact
- Prediction: φ(K) might plateau or even decrease for large K

### Phenomenon 3: τ-Dependent Regimes
- **Small τ** (< 1/γ): Demons don't have time to change state much
  - Competition based on initial state distribution
  - Less dynamic behavior
  
- **Large τ** (>> 1/γ): Demons explore many states during each interaction
  - Competition becomes more dynamic
  - State distributions might converge to quasi-equilibrium

### Phenomenon 4: State Oscillations
If demon states create oscillating total rates:
- High-rate state wins → interacts → drops to low-rate state
- Low-rate state loses → doesn't interact → stays low-rate
- Might see oscillation between demons

## Implementation Notes

### Random Number Generation
We draw K independent exponentials per bit:
```python
for k in range(K):
    total_rate = sum(rates_for_demon_k)
    dt[k] = np.random.exponential(1.0 / total_rate)

winner = np.argmin(dt)
```

### Efficiency Consideration
For large K and N:
- Need to compute K × N total rates
- Need to draw K × N exponential samples
- O(K × N) complexity

### Validation
To ensure correct implementation:
1. For K=1, should match single demon simulation
2. Check that Σ(interaction_counts) = N
3. Verify detailed balance in equilibrium (if reached)
4. Compare with analytical predictions (if available)

## Connection to Physical Systems

This competitive mechanism resembles:

1. **Chemical kinetics**: Multiple reaction pathways competing
2. **Gillespie algorithm**: Standard SSA has similar structure
3. **Queueing theory**: Multiple servers with different rates
4. **Neural networks**: Winner-take-all dynamics

The key insight: **Competition + Randomness → Emergent Behavior**

Even with identical components (demons), the system develops heterogeneity through:
- Random fluctuations in initial interactions
- State-dependent feedback
- Path-dependent evolution
