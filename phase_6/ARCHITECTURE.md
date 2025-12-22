# Phase 6 Architecture Diagram

## Stacked Demon Flow

```
INPUT TAPE (N bits)
==================
[b₀, b₁, b₂, b₃, ..., bₙ]
  |   |   |   |       |
  v   v   v   v       v

Each bit passes through K demons sequentially:

  Bit bᵢ
    |
    v
┌─────────────────┐
│   Demon 1       │  State: d₀ → d₁ → ... → dₙ
│  (n states)     │  Interaction time: τ
└─────────────────┘
    |
    | (modified bit)
    v
┌─────────────────┐
│   Demon 2       │  State: d₀ → d₁ → ... → dₙ
│  (n states)     │  Interaction time: τ
└─────────────────┘
    |
    | (modified bit)
    v
      ...
    |
    v
┌─────────────────┐
│   Demon K       │  State: d₀ → d₁ → ... → dₙ
│  (n states)     │  Interaction time: τ
└─────────────────┘
    |
    | (final bit)
    v
  Output bᵢ'

OUTPUT TAPE (N bits)
===================
[b₀', b₁', b₂', b₃', ..., bₙ']
```

## PhysParams Configuration Modes

### Per-State Mode
```
DeltaE = 1.0 (per state)
Demon with n=5 states:

d₀ ──(ΔE=1.0)→ d₁ ──(ΔE=1.0)→ d₂ ──(ΔE=1.0)→ d₃ ──(ΔE=1.0)→ d₄

Total Energy Span: 4.0
```

### Total Mode
```
DeltaE = 4.0 (total), n=5
Per-state energy = 4.0/(5-1) = 1.0

d₀ ──(1.0)→ d₁ ──(1.0)→ d₂ ──(1.0)→ d₃ ──(1.0)→ d₄

Total Energy Span: 4.0 ✓
```

## Preserve Mode Comparison

### Preserve Sigma/Omega
```
Given: σ=0.3, ω=0.5, DeltaE_total=4.0, n=5

Calculate:
  ΔE_per_state = 4.0/4 = 1.0

Since σ and ω are preserved:
  T_H = ΔE_per_state / (2·arctanh(σ)) = 1.0/(2·arctanh(0.3)) ≈ 1.615
  T_C = ΔE_per_state / (2·arctanh(ω)) = 1.0/(2·arctanh(0.5)) ≈ 0.910

Result: Temperatures adjust to maintain σ,ω
```

### Preserve Temperatures
```
Given: T_H=1.6, T_C=1.0, DeltaE_total=4.0, n=5

Calculate:
  ΔE_per_state = 4.0/4 = 1.0

Since T_H and T_C are preserved:
  σ = tanh(ΔE_per_state/(2·T_H)) = tanh(1.0/3.2) ≈ 0.303
  ω = tanh(ΔE_per_state/(2·T_C)) = tanh(1.0/2.0) ≈ 0.462

Result: Transition rates adjust to maintain T_H,T_C
```

## Class Hierarchy

```
┌──────────────────┐
│   PhysParams     │
├──────────────────┤
│ - DeltaE         │
│ - gamma          │
│ - delta_e_mode   │
│ - preserve_mode  │
│ - Th, Tc         │
│ - sigma, omega   │
│ - demon_n        │
└────────┬─────────┘
         │
         │ used by
         v
┌──────────────────┐
│     Demon        │
├──────────────────┤
│ - n (states)     │
│ - phys_params    │
│ - delta_e_values │
│ - energy_values  │
│ - intrinsic_rates│
│ - outgoing_rates │
└────────┬─────────┘
         │
         │ K demons
         v
┌─────────────────────────────┐
│  StackedDemonSimulation     │
├─────────────────────────────┤
│ - demons: List[Demon]       │
│ - K: int                    │
│ - tape: Tape                │
│ - tau: float                │
├─────────────────────────────┤
│ + run_full_simulation()     │
│ + compute_statistics()      │
│ + process_bit_through_demons()│
└─────────────────────────────┘
         │
         │ uses
         v
┌──────────────────┐
│      Tape        │
├──────────────────┤
│ - N (bits)       │
│ - p0 (prob)      │
│ - tape_arr       │
├──────────────────┤
│ + get_entropy()  │
└──────────────────┘
```

## Gillespie Algorithm Flow (Single Demon)

```
Start: Joint state "0_d2" (bit=0, demon=d2)
  |
  v
┌──────────────────────────────┐
│ Get all possible transitions │
│ - Intrinsic: d2→d1, d2→d3   │
│ - Outgoing: 0_d2→1_d3        │
└──────────┬───────────────────┘
           v
┌──────────────────────────────┐
│ Calculate total rate         │
│ rate_total = Σ rates         │
└──────────┬───────────────────┘
           v
┌──────────────────────────────┐
│ Sample time to next event    │
│ dt ~ Exp(1/rate_total)       │
└──────────┬───────────────────┘
           v
     ┌─────────────┐
     │ dt < τ ?    │
     └──┬──────┬───┘
        │ Yes  │ No
        v      v
    ┌───────┐ Return
    │Choose │ current
    │event  │ state
    └───┬───┘
        v
    Update state
    time += dt
        │
        └──→ Loop back
```

## Energy Flow in Stacked System

```
HOT RESERVOIR (Th)
       ↕ γ·exp(...)
    [Demon 1]  ←→  [Bit stream]
       ↕ 1±ω         ↓
COLD RESERVOIR (Tc)  ↓
                     ↓
HOT RESERVOIR (Th)   ↓
       ↕ γ·exp(...)  ↓
    [Demon 2]  ←→  [Bit stream]
       ↕ 1±ω         ↓
COLD RESERVOIR (Tc)  ↓
                     ↓
       ...          ...
                     ↓
HOT RESERVOIR (Th)   ↓
       ↕ γ·exp(...)  ↓
    [Demon K]  ←→  [Bit stream]
       ↕ 1±ω         ↓
COLD RESERVOIR (Tc)  ↓
                     ↓
                 [Output]

Total Q_c = Σ(energy from all K demons)
```

## Statistics Computed

```
Input Tape:  [b₀, b₁, ..., bₙ]
             p₀ (prob of 0)
             p₁ (prob of 1)
             
    ↓ Process through K demons

Output Tape: [b₀', b₁', ..., bₙ']
             p₀' (prob of 0)
             p₁' (prob of 1)

Computed Statistics:
━━━━━━━━━━━━━━━━━━━━
• φ (phi) = p₁' - p₁
  → Bit flip fraction

• bias_in = p₀ - p₁
  bias_out = p₀' - p₁'
  → Tape bias

• Q_c = φ · N · ΔE
  → Energy to cold reservoir

• ΔS_B = S(p₀') - S(p₀)
  → Entropy change
  where S(p) = -p·ln(p) - (1-p)·ln(1-p)
```

## File Dependencies

```
Simulation.py
   ↓ imports
   ├→ Demon.py
   │    ↓ imports
   │    └→ PhysParams (internal)
   │
   └→ Tape.py
        (no dependencies)

test_basic.py → Demon.py, Tape.py, Simulation.py
examples.py → Simulation.py (which imports Demon, Tape)
```

## Typical Workflow

```
1. Define Parameters
   ┌─────────────────┐
   │  PhysParams     │
   │  - Choose mode  │
   │  - Set values   │
   └────────┬────────┘
            v

2. Create Demons
   ┌─────────────────┐
   │ Create K demons │
   │ with n states   │
   └────────┬────────┘
            v

3. Create Tape
   ┌─────────────────┐
   │ Generate tape   │
   │ N bits, p₀      │
   └────────┬────────┘
            v

4. Run Simulation
   ┌─────────────────┐
   │ StackedDemon    │
   │ Simulation      │
   └────────┬────────┘
            v

5. Analyze Results
   ┌─────────────────┐
   │ compute_stats() │
   │ Plot results    │
   └─────────────────┘
```

This architecture provides maximum flexibility while maintaining clean separation of concerns!
