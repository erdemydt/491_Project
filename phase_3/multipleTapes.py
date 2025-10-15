import numpy as np
from datetime import datetime
import math
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import random


def generate_tape(N, p0, seed=None):
    """
    Generate a tape with N bits, where each bit is 0 with probability p0 and 1 otherwise.
    """
    rng = np.random.default_rng(seed or 7)
    return rng.choice([0, 1], size=N, p=[p0, 1 - p0])

def beta(T: float, kB: float = 1.0) -> float:
    """Inverse temperature β = 1/(k_B T). Default k_B = 1 for convenience."""
    if T <= 0:
        raise ValueError("Temperature must be > 0")
    return 1.0 / (kB * T)

def bias_from_T(T: float, DeltaE: float, kB: float = 1.0) -> float:
    """
    Return the canonical bias parameter tanh(β ΔE / 2).
    This is what we use for both σ (hot) and ω (cold).
    """
    return math.tanh(0.5 * beta(T, kB) * DeltaE)

def probs_from_delta_bias(bias: float) -> Tuple[float, float]:
    """
    Convert a bias  = p0 - p1 (with p0+p1=1) to (p0, p1).
     ∈ [-1, 1].  p0 = (1+)/2, p1 = (1-)/2
    """
    if not -1.0 <= bias <= 1.0:
        raise ValueError("bias must be in [-1, 1]")
    p0 = 0.5 * (1.0 + bias)
    p1 = 1.0 - p0
    return p0, p1

def bias_from_probs(p0: float, p1: Optional[float] = None) -> float:
    """Return bias = p0 - p1. If p1 is None, use p1 = 1 - p0."""
    if p1 is None:
        p1 = 1.0 - p0
    if p0 < 0 or p1 < 0 or abs(p0 + p1 - 1.0) > 1e-12:
        raise ValueError("probabilities must be nonnegative and sum to 1")
    return p0 - p1

@dataclass
class PhysParams:
    # Physical knobs (you set these)
    Th: float = 0.0                 # hot-bath temperature
    Tc: float = 0.0                 # cold-bath temperature
    DeltaE: float = 1.0             # energy quantum exchanged via cooperative flip
    kB: float = 1.0           # Boltzmann constant (default 1.0 units)

    # Timescales for Poisson clocks (you set these too)
    gamma_hot: float = 1.0    # overall scale for intrinsic (hot) demon flips
    kappa_cold: float = 1.0   # overall scale for cooperative (cold) flips

    # Derived biases (auto-filled post-init)
    sigma: float = 0.0        # hot bias: tanh(β_h ΔE/2)
    omega: float = 0.0        # cold bias: tanh(β_c ΔE/2)

    def __post_init__(self):
        if( self.Th != 0 or self.Tc != 0 ):
            self.sigma = bias_from_T(self.Th, self.DeltaE, self.kB)  # hot-bath bias
            self.omega = bias_from_T(self.Tc, self.DeltaE, self.kB)  # cold-bath bias
        elif ( self.sigma != 0 and self.omega != 0 ):
            try:
                self.Th = self.DeltaE / (2.0 * self.kB * math.atanh(self.sigma))  # hot-bath temperature
                self.Tc = self.DeltaE / (2.0 * self.kB * math.atanh(self.omega))  # cold-bath temperature
            except ValueError:
                raise ValueError(f"invalid sigma or omega: sigma={self.sigma}, omega={self.omega}")
        else:
            raise ValueError("Must specify either (Th, Tc) or (sigma, omega)")
        
        

def intrinsic_rates(sigma: float, gamma_hot: float) -> Tuple[float, float]:
    """
    Demon intrinsic (hot-bath) flips:
      d -> u : gamma*(1 - sigma)
      u -> d : gamma*(1 + sigma)
    """
    d_to_u = gamma_hot * (1.0 - sigma)
    u_to_d = gamma_hot * (1.0 + sigma)
    return d_to_u, u_to_d

def cooperative_rates(omega: float, kappa_cold: float) -> Tuple[float, float]:
    """
    Cooperative (cold-bath) flips on joint states:
      0d -> 1u : kappa*(1 - omega)
      1u -> 0d : kappa*(1 + omega)
    """
    d0_to_u1 = kappa_cold * (1.0 - omega)
    u1_to_d0 = kappa_cold * (1.0 + omega)
    return d0_to_u1, u1_to_d0

# State labels (strings for readability): "0u", "0d", "1u", "1d"
def outgoing_rates(state: str, phys: PhysParams) -> Dict[str, float]:
    """
    List the enabled single-step transitions and their rates from the given state.
    """
    d_to_u, u_to_d = intrinsic_rates(phys.sigma, phys.gamma_hot)
    d0_to_u1, u1_to_d0 = cooperative_rates(phys.omega, phys.kappa_cold)

    rates = {}
    if state == "0u":
        rates["0d"] = u_to_d                         # intrinsic only
    elif state == "0d":
        rates["0u"] = d_to_u                         # intrinsic
        rates["1u"] = d0_to_u1                       # cooperative
    elif state == "1u":
        rates["1d"] = u_to_d                         # intrinsic
        rates["0d"] = u1_to_d0                       # cooperative (reverse)
    elif state == "1d":
        rates["1u"] = d_to_u                         # intrinsic
    else:
        raise ValueError(f"Unknown state {state}")
    return rates



# -------------------------
# Wiring bits ↔ demon per window
# -------------------------


def demon_bit_to_joint(bit: str, demon: str) -> str:
    """Combine bit ('0'/'1') and demon ('u'/'d') into joint label like '0u'."""
    return f"{bit}{demon}"

def joint_to_bit_demon(joint: str) -> Tuple[str, str]:
    """Inverse of demon_bit_to_joint."""
    return joint[0], joint[1]



def gillespie_within_window(state, tau, phys, rng=random):
    """
    Simulate exact CTMC within one fixed interaction window of length tau.
    state: initial joint state (e.g., '0u')
    tau: duration of demon-bit interaction
    phys: parameters (with sigma, omega, gamma_hot, kappa_cold)
    returns: final joint state at time tau
    """
    t = 0.0
    s = state
    while True:
        rates = outgoing_rates(s, phys)
        rtot = sum(rates.values())
        if rtot <= 0:
            # absorbing state: no more jumps possible
            return s
        # exponential waiting time to next event
        dt = rng.expovariate(rtot)
        if t + dt > tau:
            # next event would happen after window closes
            return s
        # otherwise, event occurs
        t += dt
        # choose which event
        x = rng.random() * rtot
        cum = 0.0
        for nxt, r in rates.items():
            cum += r
            if x < cum:
                s = nxt
                break
        # loop continues with new state, less time remaining
def step_with_fresh_bit_gillespie(demon_state, incoming_bit, tau, phys, rng=random):
    b_in = str(incoming_bit)
    joint0 = b_in + demon_state
    jointf = gillespie_within_window(joint0, tau, phys, rng)
    b_out, d_out = jointf[0], jointf[1]
    return b_in, d_out, b_out


def run_sim_gillespie_windows(
    tau: float,
    phys: PhysParams,
    p0_in: float,
    demon_init: str = "u",
    incoming_bits: np.ndarray = None,
    seed: Optional[int] = None
) -> dict:
    """
    Run N interaction windows with *independent fresh bits* each time (legacy version).
    Tracks incoming and outgoing bit biases and demon occupancy.
    Uses exact Gillespie simulation within each window.
    """
    if incoming_bits is None:
        raise ValueError("incoming_bits must be provided for this function")
    rng = random.Random(seed)
    N = len(incoming_bits)
    # stats
    in_counts = {"0": 0, "1": 0}
    out_counts = {"0": 0, "1": 0}
    demon_counts = {"u": 0, "d": 0}
    outgoing_tape = np.zeros(N, dtype=int)
    demon = demon_init
    for i in range(N):
        b_in, demon, b_out = step_with_fresh_bit_gillespie(demon, incoming_bits[i], tau, phys, rng)
        in_counts[b_in] += 1
        out_counts[b_out] += 1
        demon_counts[demon] += 1
        outgoing_tape[i] = b_out

    # biases
    p0_in_emp = in_counts["0"] / N
    p1_in_emp = 1.0 - p0_in_emp
    eps_in_emp = p0_in_emp - p1_in_emp

    p0_out_emp = out_counts["0"] / N
    p1_out_emp = 1.0 - p0_out_emp
    eps_out_emp = p0_out_emp - p1_out_emp

    # demon occupancy
    pu_emp = demon_counts["u"] / N
    pd_emp = 1.0 - pu_emp
    delta_S_B = S_of_delta(eps_out_emp) - S_of_delta(eps_in_emp)
    return {
        "incoming": {
            "p0": p0_in_emp, "p1": p1_in_emp, "bias": eps_in_emp,
            "counts": in_counts
        },
        "outgoing": {
            "p0": p0_out_emp, "p1": p1_out_emp, "bias": eps_out_emp,
            "counts": out_counts,
            "tape": outgoing_tape,
            "DeltaS_B": delta_S_B
        },
        "demon": {
            "pu": pu_emp, "pd": pd_emp, "counts": demon_counts
        }
    }
    
def run_gillespie_multiple_tapes(
    N: int,
    tau: float,
    phys: PhysParams,
    p0_in: float,
    demon_init: str = "u",
    n_tapes: int = 10,
    seed: Optional[int] = None
) -> dict:
    """
    Run one tape multiple times, each time re-inserting it
    """
    rng = np.random.default_rng(seed or 7)
    tape = generate_tape(N, p0_in, seed=7)
    stats_list = []
    for i in range(n_tapes):
        stats = run_sim_gillespie_windows(tau, phys, p0_in, demon_init, tape, seed=rng.integers(1e9))
        stats_list.append(stats)
        tape = stats["outgoing"]["tape"]  # re-insert outgoing tape as new incoming tape
    return stats_list
def delta_out_from_phi(delta_in, Phi):
    """Eq. (6): δ' = δ - 2Φ (since Φ = p1' - p1)."""
    return float(delta_in - 2.0 * Phi)

def DeltaS_B(delta_in, Phi):
    """Eq. (9): ΔS_B = S(δ') - S(δ)."""
    delta_out = delta_out_from_phi(delta_in, Phi)
    return S_of_delta(delta_out) - S_of_delta(delta_in)
def S_of_delta(delta):
    """Shannon info per bit for a Bernoulli( p0=(1+δ)/2, p1=(1-δ)/2 ).
       Natural logs (nats)."""
    p0 = 0.5 * (1.0 + delta)
    p1 = 0.5 * (1.0 - delta)
    # guard: 0 log 0 := 0
    terms = []
    if p0 > 0: terms.append(-p0 * np.log(p0))
    if p1 > 0: terms.append(-p1 * np.log(p1))
    return float(sum(terms))

T_H = 1.1
T_C = 1.0
DeltaE = 1
gamma_hot = 1.0
kappa_cold = 1.0
tau = 1.0
N = 3000
p0_in = 0.0  # incoming bit distribution (p0)
phys = PhysParams(Th=T_H, Tc=T_C, DeltaE=DeltaE, gamma_hot=gamma_hot, kappa_cold=kappa_cold)
if __name__ == "__main__":
    # Example usage: run one tape multiple times
    stats_list = run_gillespie_multiple_tapes(N, tau, phys, p0_in, demon_init="u", n_tapes=300, seed=7)
    phi_values = []
    bias_values = []
    delta_s_b_values = []
    for i, stats in enumerate(stats_list):
        print(f"Tape {i+1}:")
        print("  Incoming bias _in:", stats["incoming"]["bias"])
        print("  Outgoing bias _out:", stats["outgoing"]["bias"])
        phi_values.append(stats["incoming"]["p1"] - stats["outgoing"]["p1"])
        bias_values.append(stats["outgoing"]["bias"])
        delta_s_b_values.append(stats["outgoing"]["DeltaS_B"])
        print()
    # Plot phi values over tapes
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()

    # Second axes for bias values
    line2 = ax1.plot(range(1, len(bias_values) + 1), bias_values, marker='o', color='blue', label='Outgoing Bias')
    ax1.set_ylabel('Outgoing Bias')
    
    # ax2 = ax1.twinx()
    # line1 = ax2.plot(range(1, len(phi_values) + 1), phi_values, marker='x', label='φ (Change in p1)')
    # ax2.set_xlabel('Tape Number')
    # ax2.set_ylabel('φ (Change in p1)')
    # Combine legends from both axes
    lines =  line2 #+ line1
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)
    plt.title(f'Change in Bit Bias φ over Multiple Tapes for p0_in = {p0_in}, T_H={T_H}, T_C={T_C}, ΔE={DeltaE}, τ={tau}')
    ax1.grid()
    # New figure for Delta S_B
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(delta_s_b_values) + 1), delta_s_b_values, marker='o', color='green')
    plt.xlabel('Tape Number')
    plt.ylabel('ΔS_B (Bit Entropy Change)')
    plt.title(f'Bit Entropy Change ΔS_B over Multiple Tapes for p0_in = {p0_in}, T_H={T_H}, T_C={T_C}, ΔE={DeltaE}, τ={tau}')
    plt.grid()
    
    plt.show()
