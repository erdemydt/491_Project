# Phase 8: Competing Demons with Competitive Sampling

## Overview

In this phase, we implement a **competitive interaction model** where K identical demons compete to interact with each bit on the tape. Unlike Phase 7 where demons interact sequentially, here all K demons compete simultaneously, and the demon with the shortest time-to-next-event wins the interaction.

## Key Concept: Competitive Sampling

For each bit on the tape:

1. All K demons are in potentially different states (d0, d1, ..., d_{n-1})
2. For each demon, we calculate its total transition rate based on its current joint state (bit_value, demon_state)
3. Each demon draws a time-to-next-event `dt` from an exponential distribution: `dt ~ Exponential(1/total_rate)`
4. The demon with the **smallest dt** wins and interacts with the bit
5. Only the winning demon's state changes; other demons remain in their current states

## Why This Is Interesting

This model explores whether demons in different states can collectively produce different behavior compared to:
- A single demon (K=1)
- Sequential interaction (Phase 7)

Key questions:
- Do demons in higher energy states interact more or less frequently?
- Does having K demons in different states affect the bit flip fraction φ?
- How does the interaction distribution change with different parameters (σ, ω, τ, n)?

## Files

- `CompetingDemon.py`: Demon class with competitive sampling support
- `Tape.py`: Simple tape implementation (same as Phase 6)
- `CompetingSimulation.py`: Main simulation engine with competitive selection
- `README.md`: This file

## Implementation Details

### Competitive Selection Mechanism

```python
def select_winning_demon(bit_value, demon_states):
    dts = []
    for each demon k:
        joint_state = f'{bit_value}_{demon_states[k]}'
        total_rate = sum(all transition rates from joint_state)
        
        if total_rate > 0:
            dt = exponential(1/total_rate)
        else:
            dt = infinity
        
        dts.append(dt)
    
    winning_idx = argmin(dts)
    return winning_idx
```

### Interaction Process

After selecting the winning demon:
1. Run Gillespie algorithm for time window τ
2. Update bit value and winning demon's state
3. Other demons' states remain unchanged
4. Track which demon won for statistics

## Key Differences from Phase 7

| Aspect | Phase 7 (Sequential) | Phase 8 (Competitive) |
|--------|---------------------|----------------------|
| Interaction | Sequential through K demons | Competitive sampling |
| Demon states | All change for each bit | Only winner changes |
| Time model | τ per demon | τ total per bit |
| State correlation | All demons process same bit | Demons in different states compete |

## Usage

### Run a Single Simulation

```python
from CompetingDemon import CompetingDemon, PhysParams
from Tape import Tape
from CompetingSimulation import CompetingDemonSimulation

# Create parameters
phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)

# Create K demons
K = 5
demons = [CompetingDemon(n=3, phys_params=phys_params, 
                        init_state='d0', demon_id=k) for k in range(K)]

# Create tape
tape = Tape(N=5000, p0=1.0)

# Run simulation
sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=1.0)
final_tape, initial_tape, demon_states_history, interaction_counts = \
    sim.run_full_simulation()

# Compute statistics
stats = sim.compute_statistics(final_tape)
print(f"φ = {stats['phi']:.4f}")
print(f"Interaction distribution: {stats['interaction_fractions']}")
```

### Plot φ vs K

```python
from CompetingSimulation import plot_output_vs_K

plot_output_vs_K(
    K_values=list(range(1, 21)),
    output='phi',
    tape_params={"N": 5000, "p0": 1.0},
    demon_n=3,
    tau=1.0,
    phys_params=PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
)
```

### Run Demo

```bash
cd phase_8
python CompetingSimulation.py
```

## Statistics Tracked

- **Bit flip fraction (φ)**: Fraction of bits that changed
- **Bias change**: Change in (p₀ - p₁)
- **Entropy change (ΔS_B)**: Change in bit entropy
- **Energy transferred (Q_c)**: Energy to cold reservoir
- **Interaction counts**: How many times each demon won
- **Interaction fractions**: Fraction of bits each demon interacted with

## Expected Behavior

### Hypothesis 1: State Distribution Affects Interaction Probability
Demons in different states have different total transition rates, leading to different dt distributions. Demons with higher total rates (more possible transitions) will draw smaller dt values on average and win more often.

### Hypothesis 2: K Dependence
As K increases:
- More competition for each bit
- Interaction distribution may become more skewed toward certain states
- Overall behavior may differ from sequential K-demon model

### Hypothesis 3: τ Dependence
For small τ:
- Less time for demon states to change during interaction
- Demons tend to stay in their initial states longer
- Competition is more "frozen"

For large τ:
- Demons have time to explore multiple states during each interaction
- Competition dynamics become more complex

## Parameters

### Physical Parameters (PhysParams)
- `DeltaE`: Energy gap between consecutive demon states
- `gamma`: Transition rate with hot reservoir
- `Th`: Hot reservoir temperature
- `Tc`: Cold reservoir temperature
- `sigma = tanh(DeltaE/(2*Th))`: Hot reservoir parameter (computed)
- `omega = tanh(DeltaE/(2*Tc))`: Cold reservoir parameter (computed)

### Simulation Parameters
- `K`: Number of competing demons
- `n`: Number of states per demon
- `tau`: Interaction time per bit
- `N`: Number of bits on tape
- `p0`: Initial probability of bit being 0

## Future Explorations

1. **Non-identical demons**: What if demons have different parameters?
2. **Memory effects**: Track correlation between winning demon's state and bit outcome
3. **Efficiency analysis**: Compare energy extraction vs sequential model
4. **Optimal K**: Is there an optimal number of demons for maximum φ?
5. **State dynamics**: Analyze how demon state distributions evolve over tape

## Notes

- All demons are identical in this implementation
- Demons only change state when they win an interaction
- The competitive mechanism naturally creates asymmetry even with identical demons
- This is a fundamentally different model from stacking demons sequentially
