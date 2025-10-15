import math
import random
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, List, Iterable, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import utilityFunctions as uf
# -------------------------
# Thermo / parameter helpers
# -------------------------

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

def probs_from_delta_bias(eps: float) -> Tuple[float, float]:
    """
    Convert a bias ε = p0 - p1 (with p0+p1=1) to (p0, p1).
    ε ∈ [-1, 1].  p0 = (1+ε)/2, p1 = (1-ε)/2
    """
    if not -1.0 <= eps <= 1.0:
        raise ValueError("epsilon must be in [-1, 1]")
    p0 = 0.5 * (1.0 + eps)
    p1 = 1.0 - p0
    return p0, p1

def bias_from_probs(p0: float, p1: Optional[float] = None) -> float:
    """Return bias = p0 - p1. If p1 is None, use p1 = 1 - p0."""
    if p1 is None:
        p1 = 1.0 - p0
    if p0 < 0 or p1 < 0 or abs(p0 + p1 - 1.0) > 1e-12:
        raise ValueError("probabilities must be nonnegative and sum to 1")
    return p0 - p1

# -------------------------
# Model parameters
# -------------------------

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

# -------------------------
# Rates → one-step probabilities (no matrices)
# -------------------------

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

def one_step_distribution(current: str, t: float, phys: PhysParams) -> Dict[str, float]:
    """
    Exact per-window distribution using only rates and proportions (at most one jump in the window):
      P(stay) = exp(-r_tot t)
      P(jump i) = (r_i / r_tot) * (1 - exp(-r_tot t))
    """
    outs = outgoing_rates(current, phys)
    rtot = sum(outs.values())
    if rtot <= 0:
        return {current: 1.0}

    stay = math.exp(-rtot * t)
    dist = {current: stay}
    comp = 1.0 - stay
    for nxt, r in outs.items():
        dist[nxt] = dist.get(nxt, 0.0) + (r / rtot) * comp

    # tidy normalization
    z = sum(dist.values())
    for k in list(dist.keys()):
        dist[k] /= z
    return dist

def one_step_sample(current: str, t: float, phys: PhysParams, rng: random.Random = random) -> str:
    """Sample next state according to one_step_distribution."""
    dist = one_step_distribution(current, t, phys)
    x = rng.random()
    acc = 0.0
    for s, p in dist.items():
        acc += p
        if x < acc:
            return s
    return current

# -------------------------
# Wiring bits ↔ demon per window
# -------------------------

def draw_incoming_bit(p0_in: float, rng: random.Random = random) -> str:
    """Sample an incoming bit state '0' or '1' from p0_in."""
    return "0" if rng.random() < p0_in else "1"

def demon_bit_to_joint(bit: str, demon: str) -> str:
    """Combine bit ('0'/'1') and demon ('u'/'d') into joint label like '0u'."""
    return f"{bit}{demon}"

def joint_to_bit_demon(joint: str) -> Tuple[str, str]:
    """Inverse of demon_bit_to_joint."""
    return joint[0], joint[1]

def step_with_continuous_evolution(
    demon_state: str,
    p0_in: float,
    T: float,
    dt: float,
    phys: PhysParams,
    rng: random.Random = random,
    cumulative: bool = False
) -> Tuple[str, str, str]:
    """
    One interaction window with continuous evolution:
      - sample a fresh incoming bit from p0_in
      - form the initial joint state
      - evolve continuously for time T in steps of dt
      - if cumulative=True, use cumulative time for flips
      - return (incoming_bit, demon_state_after, outgoing_bit)
    """
    b_in = draw_incoming_bit(p0_in, rng)
    joint_state = demon_bit_to_joint(b_in, demon_state)
   
    # Continuous evolution loop
    time_elapsed = 0.0
    other_time = 0.0
    while time_elapsed < T:
        remaining_time = T - time_elapsed
        step_time = min(dt, remaining_time)
        in_time = step_time
        other_time += step_time
        if cumulative:
            in_time = other_time

        # Take one step with the current joint state
        end_state = one_step_sample(joint_state, in_time, phys, rng)
        if end_state != joint_state:
            joint_state = end_state
            other_time = 0.0
        time_elapsed += step_time
    
    # Extract final states
    b_out, d_out = joint_to_bit_demon(end_state)
    return b_in, d_out, b_out

def step_with_continuous_evolution_traced(
    demon_state: str,
    p0_in: float,
    T: float,
    dt: float,
    phys: PhysParams,
    rng: random.Random = random
) -> Tuple[str, str, str, List[Tuple[float, str]]]:
    """
    Same as step_with_continuous_evolution but also returns the full trajectory.
    Returns (incoming_bit, demon_state_after, outgoing_bit, trajectory)
    where trajectory is a list of (time, joint_state) pairs.
    """
    b_in = draw_incoming_bit(p0_in, rng)
    joint_state = demon_bit_to_joint(b_in, demon_state)
    
    # Track the trajectory
    trajectory = [(0.0, joint_state)]
    
    # Continuous evolution loop
    time_elapsed = 0.0
    while time_elapsed < T:
        remaining_time = T - time_elapsed
        step_time = min(dt, remaining_time)
        
        # Take one step with the current joint state
        joint_state = one_step_sample(joint_state, step_time, phys, rng)
        time_elapsed += step_time
        trajectory.append((time_elapsed, joint_state))
    
    # Extract final states
    b_out, d_out = joint_to_bit_demon(joint_state)
    return b_in, d_out, b_out, trajectory

def step_with_fresh_bit(
    demon_state: str,
    p0_in: float,
    t: float,
    phys: PhysParams,
    rng: random.Random = random
) -> Tuple[str, str, str]:
    """
    One interaction window (legacy single-jump version):
      - sample a fresh incoming bit from p0_in
      - form the joint state
      - evolve with one-step distribution
      - return (incoming_bit, demon_state_after, outgoing_bit)
    """
    b_in = draw_incoming_bit(p0_in, rng)
    joint0 = demon_bit_to_joint(b_in, demon_state)
    joint1 = one_step_sample(joint0, t, phys, rng)
    b_out, d_out = joint_to_bit_demon(joint1)
    return b_in, d_out, b_out

# -------------------------
# Sim drivers + bias accounting
# -------------------------

def run_sim_continuous(
    N: int,
    T: float,
    dt: float,
    phys: PhysParams,
    p0_in: float,
    demon_init: str = "u",
    cumulative: bool = False,
    seed: Optional[int] = None
) -> dict:
    """
    Run N interaction windows with continuous evolution over time T with steps dt.
    Each bit interacts with the demon for the full duration T, allowing multiple
    state changes during each interaction window.
    """
    rng = random.Random(seed)

    # stats
    in_counts = {"0": 0, "1": 0}
    out_counts = {"0": 0, "1": 0}
    demon_counts = {"u": 0, "d": 0}

    demon = demon_init
    for _ in range(N):
        b_in, demon, b_out = step_with_continuous_evolution(demon, p0_in, T, dt, phys, rng,cumulative)
        in_counts[b_in] += 1
        out_counts[b_out] += 1
        demon_counts[demon] += 1

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

    return {
        "incoming": {
            "p0": p0_in_emp, "p1": p1_in_emp, "epsilon": eps_in_emp,
            "counts": in_counts
        },
        "outgoing": {
            "p0": p0_out_emp, "p1": p1_out_emp, "epsilon": eps_out_emp,
            "counts": out_counts
        },
        "demon": {
            "pu": pu_emp, "pd": pd_emp, "counts": demon_counts
        }
    }

def run_sim(
    N: int,
    t: float,
    phys: PhysParams,
    p0_in: float,
    demon_init: str = "u",
    seed: Optional[int] = None
) -> dict:
    """
    Run N interaction windows with *independent fresh bits* each time (legacy version).
    Tracks incoming and outgoing bit biases and demon occupancy.
    """
    rng = random.Random(seed)

    # stats
    in_counts = {"0": 0, "1": 0}
    out_counts = {"0": 0, "1": 0}
    demon_counts = {"u": 0, "d": 0}

    demon = demon_init
    for _ in range(N):
        b_in, demon, b_out = step_with_fresh_bit(demon, p0_in, t, phys, rng)
        in_counts[b_in] += 1
        out_counts[b_out] += 1
        demon_counts[demon] += 1

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

    return {
        "incoming": {
            "p0": p0_in_emp, "p1": p1_in_emp, "epsilon": eps_in_emp,
            "counts": in_counts
        },
        "outgoing": {
            "p0": p0_out_emp, "p1": p1_out_emp, "epsilon": eps_out_emp,
            "counts": out_counts
        },
        "demon": {
            "pu": pu_emp, "pd": pd_emp, "counts": demon_counts
        }
    }

# -------------------------
# (Optional) simple heat bookkeeping per window (coarse)
# -------------------------
def estimate_heat_c_to_h_over_path(
    path: Iterable[Tuple[str, str]],
    DeltaE: float
) -> float:
    """
    Given a sequence of (joint_before, joint_after) pairs, estimate net heat
    transferred from cold→hot by counting cooperative flips:
      0d -> 1u : +ΔE from cold to hot
      1u -> 0d : -ΔE to cold (reverse)
    This is *coarse* because the one-jump-per-window rule ignores multiple flips.
    """
    Q = 0.0
    for j0, j1 in path:
        if j0 == "0d" and j1 == "1u":
            Q += DeltaE
        elif j0 == "1u" and j1 == "0d":
            Q -= DeltaE
    return Q
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
def step_with_fresh_bit_gillespie(demon_state, p0_in, tau, phys, rng=random):
    b_in = "0" if rng.random() < p0_in else "1"
    joint0 = b_in + demon_state
    jointf = gillespie_within_window(joint0, tau, phys, rng)
    b_out, d_out = jointf[0], jointf[1]
    return b_in, d_out, b_out
def run_sim_gillespie_windows( N: int,
    tau: float,
    phys: PhysParams,
    p0_in: float,
    demon_init: str = "u",
    seed: Optional[int] = None
) -> dict:
    """
    Run N interaction windows with *independent fresh bits* each time (legacy version).
    Tracks incoming and outgoing bit biases and demon occupancy.
    Uses exact Gillespie simulation within each window.
    """
    rng = random.Random(seed)

    # stats
    in_counts = {"0": 0, "1": 0}
    out_counts = {"0": 0, "1": 0}
    demon_counts = {"u": 0, "d": 0}

    demon = demon_init
    for _ in range(N):
        b_in, demon, b_out = step_with_fresh_bit_gillespie(demon, p0_in, tau, phys, rng)
        in_counts[b_in] += 1
        out_counts[b_out] += 1
        demon_counts[demon] += 1

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

    return {
        "incoming": {
            "p0": p0_in_emp, "p1": p1_in_emp, "epsilon": eps_in_emp,
            "counts": in_counts
        },
        "outgoing": {
            "p0": p0_out_emp, "p1": p1_out_emp, "epsilon": eps_out_emp,
            "counts": out_counts
        },
        "demon": {
            "pu": pu_emp, "pd": pd_emp, "counts": demon_counts
        }
    }
def get_sigma_from_omega_epsilon(omega, epsilon):
    """Given omega and epsilon, compute sigma."""
    if not -1.0 < omega < 1.0:
        raise ValueError("omega must be in (-1, 1)")
    if not -1.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [-1, 1]")
    sigma = (epsilon - omega) / (-1 + epsilon * omega)

    return sigma
def plot_phi_map_gillespie(
    N: int,
    tau: float,
    demon_init: str = "u",
    n_points: int = 50,
    omega = 0.5,
    seed: Optional[int] = None
):
    """
    Plot the information current Φ = p1' - p1 over a matrix of delta and epsilon.
    Uses Gillespie simulation within each window.
    """
    phi_values = [[], []]
    delta_range = np.linspace(-1, 1, n_points, endpoint=True)
    epsilon_range = np.linspace(0, 1, n_points, endpoint=False)
    percentage_comp = 0.0
    for eps in epsilon_range:
        row = []
        for delta in delta_range:
            sigma = get_sigma_from_omega_epsilon(omega, eps)
            try:
                phys = PhysParams(sigma=sigma, omega=omega, gamma_hot=1.0, kappa_cold=1.0)
            except ValueError:
                raise ValueError(f"Invalid parameters for sigma={sigma}, omega={omega}, eps={eps}")
            phys = PhysParams(sigma=sigma, omega=omega, gamma_hot=1.0, kappa_cold=1.0)
            p0_in, p1_in = probs_from_delta_bias(delta)
            sta = run_sim_gillespie_windows(N=N, tau=tau, phys=phys, p0_in=p0_in, demon_init=demon_init, seed=seed)
            phi_emp = sta["outgoing"]["p1"] - sta["incoming"]["p1"]
            row.append(phi_emp)
        percentage_comp += 1.0
        print(f"Completed {percentage_comp / len(epsilon_range) * 100:.1f}%")
        phi_values[0].append(delta)
        phi_values[1].append(row)
    plt.figure(figsize=(8, 6))
    max_abs_phi = max(abs(np.min(phi_values[1])), abs(np.max(phi_values[1])))/1.5

    plt.imshow(phi_values[1], extent=(-1, 1, 0, 1), aspect='auto', origin='lower', cmap='bwr', vmin=-max_abs_phi, vmax=max_abs_phi)
    plt.colorbar(label='Φ (Information Current)')
    plt.xlabel('Incoming Bit Bias δ')
    plt.ylabel('Incoming Bit Bias ε')
    plt.title(f'Information Current Φ over δ and ε (ω={omega}, N={N}, τ={tau})')
    plt.show()

def getSolutionWithPhys(
    phys: PhysParams,
    delta: float,
    tau: float,
    useDetSolution: bool = True
) -> Dict[str, float]:
    """Get the deterministic solution given physical parameters, delta, and tau."""
    sigma = phys.sigma
    omega = phys.omega
    gamma = phys.gamma_hot
    kappa = phys.kappa_cold
    if not useDetSolution:
        return run_sim_gillespie_windows(N=10000, tau=tau, phys=phys, p0_in=probs_from_delta_bias(delta)[0], demon_init="u", seed=None)
    try:
        det = uf.deterministic_solution(gamma, sigma, omega, tau, delta)
        # print("Deterministic Phi:", det["Phi"])
        # print("Outgoing bit distribution:", det["pB_out"])
        # raise SystemExit("Debugging")
    except ValueError:
        raise ValueError(f"Invalid parameters for sigma={sigma}, omega={omega}, delta={delta}, tau={tau}")
    return det

def plot_phi_per_parameter_slice(
    N: int,
    demon_init: str = "u",
    n_points: int = 50,
    delta_fixed = 0.0,
    fixed_params: Dict[str, float] = {"Th": 1.6, "Tc": 1.0, "DeltaE": 1.0, "gamma_hot": 1.0, "tau": 1.0},
    param_name: str = "tau",
    params_range: Tuple[float, float] = (0.01, 5.0),
    gillespie: bool = True,
    plotNegativeDelta: bool = False,
    useDetSolution: bool = False,
    seed: Optional[int] = None
):
    """
    Plot the information current Φ = p1' - p1 over a slice of given parameter name.
    T_H, T_C, ΔE, γ, are fixed.
    Uses Gillespie or continuous simulation within each window.
    """
    deltaS_B_vals = []
    phi_values = []
    bit_bias_values = []
    param_range = np.linspace(params_range[0], params_range[1], n_points, endpoint=True)
    percentage_comp = 0.0
    for param_value in param_range:
        params = fixed_params.copy()
        params[param_name] = param_value
        try:
            phys = PhysParams(Th=params["Th"], Tc=params["Tc"], DeltaE=params["DeltaE"], gamma_hot=params["gamma_hot"], kappa_cold=1.0)
        except ValueError:
            raise ValueError(f"Invalid parameters for {params}")
        p0_in, p1_in = probs_from_delta_bias(delta_fixed)
        sta= None
        if gillespie:
            sta = getSolutionWithPhys(phys, delta_fixed, params.get("tau", 1.0), useDetSolution)
            if useDetSolution:
                phi_emp = sta["Phi"]
                bit_bias = sta["pB_out"][0] - sta["pB_out"][1]
                phi_values.append(phi_emp)
                bit_bias_values.append(bit_bias)
                deltaS_B_vals.append(uf.DeltaS_B(delta_fixed, phi_emp))
                percentage_comp += 1.0
                print(f"Completed {percentage_comp / len(param_range) * 100:.1f}%")
                continue
        else:
            sta = run_sim_continuous(N=N, T=params.get("tau", 1.0), dt=0.1, phys=phys, p0_in=p0_in, demon_init=demon_init, seed=seed)
        phi_emp = sta["outgoing"]["p1"] - sta["incoming"]["p1"]
        phi_values.append(phi_emp)
        bit_bias = sta["outgoing"]["epsilon"]
        bit_bias_values.append(bit_bias)
        deltaS_B_vals.append(uf.DeltaS_B(delta_fixed, phi_emp))
        percentage_comp += 1.0
        print(f"Completed {percentage_comp / len(param_range) * 100:.1f}%")
    
    # Plotting, 
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    line1 = ax1.plot(param_range, phi_values, marker='o', label='Φ (Information Current)', color='blue')
    ax1.set_xlabel(f'{param_name}')
    ax1.set_ylabel('Φ (Information Current)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'Information Current Φ over {param_name} (δ={delta_fixed}, N={N})')
    ax1.grid()
    
    # Now bit bias on secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(param_range, bit_bias_values, color='green', marker='x', label='Outgoing Bit Bias δ_out')
    ax2.set_ylabel('Outgoing Bit Bias δ_out', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Zero Line For The Secondary Axis
    ax2.axhline(0, color='black', linestyle='--')
    line3 = None
    if plotNegativeDelta:
        percentage_comp = 0.0
        phi_values = []
        bit_bias_values = []
        for param_value in param_range:
            params = fixed_params.copy()
            params[param_name] = param_value
            try:
                phys = PhysParams(Th=params["Th"], Tc=params["Tc"], DeltaE=params["DeltaE"], gamma_hot=params["gamma_hot"], kappa_cold=1.0)
            except ValueError:
                raise ValueError(f"Invalid parameters for {params}")
            p0_in, p1_in = probs_from_delta_bias(-delta_fixed)
            sta= None
            if gillespie:
                sta = getSolutionWithPhys(phys, -delta_fixed, params.get("tau", 1.0), useDetSolution)
                if useDetSolution:
                    phi_emp = sta["Phi"]
                    bit_bias = sta["pB_out"][0] - sta["pB_out"][1]
                    phi_values.append(phi_emp)
                    bit_bias_values.append(bit_bias)
                    percentage_comp += 1.0
                    print(f"Completed (Opposite Bias) {percentage_comp / len(param_range) * 100:.1f}%")
                    continue
            else:
                sta = run_sim_continuous(N=N, T=params.get("tau", 1.0), dt=0.1, phys=phys, p0_in=p0_in, demon_init=demon_init, seed=seed)
            phi_emp = sta["outgoing"]["p1"] - sta["incoming"]["p1"]
            phi_values.append(phi_emp)
            bit_bias = sta["outgoing"]["epsilon"]
            bit_bias_values.append(bit_bias)
            percentage_comp += 1.0
            print(f"Completed (Opposite Bias) {percentage_comp / len(param_range) * 100:.1f}%")
    # Combine legends
        line3 = ax2.plot(param_range, bit_bias_values, marker='o', label='Outgoing Bit Bias (Opposite)', color='cyan')
        # ax3.get_yaxis().set_visible(False)  # hide the third y-axis
        # Combine legends from all three axes
    if line3:
        lines = line1 + line2 + line3
    else:
        lines = line1 + line2   
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels)
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, deltaS_B_vals, marker='o', color='purple')
    plt.xlabel(f'{param_name}')
    plt.ylabel('ΔS_B (Bit Entropy Change)')
    plt.title(f'Bit Entropy Change ΔS_B over {param_name} (δ={delta_fixed}, N={N})')
    plt.grid()
    plt.show()

# -------------------------
# Tiny demo
# -------------------------
if __name__ == "__main__":
    # Physical setup
    Th, Tc = 5.6, 1.0       # temperatures (in k_B=1 units)
    DeltaE = 1.0            # energy scale
    print(f"Running simulation with parameters:")
    print(f"  Th = {Th}, Tc = {Tc}, DeltaE = {DeltaE}")
    phys = PhysParams(Th=Th, Tc=Tc, DeltaE=DeltaE, gamma_hot=2.0)
    # Verify derived biases
    print(f"sigma (hot) = {phys.sigma:.4f}, omega (cold) = {phys.omega:.4f}")

    # Incoming bit bias: set either p0_in or epsilon_in
    epsilon_in = 0.9                      # more 0s than 1s
    p0_in, p1_in = probs_from_delta_bias(epsilon_in)
    # print(f"Incoming p0={p0_in:.3f}, p1={p1_in:.3f}, eps={epsilon_in:.3f}")
    
    # sta = run_sim_gillespie_windows(N=10000, tau=1.0, phys=phys, p0_in=p0_in, demon_init="u", seed=7)
    # print("Empirical incoming, Gillespie:", sta["incoming"])
    # print("Empirical outgoing, Gillespie:", sta["outgoing"])
    # print(f"Phi_emp, Gillespie = {sta['outgoing']['p1'] - sta['incoming']['p1']:.4f}")
    # plot_phi_map_gillespie(N=3000, tau=1.0, demon_init="u", n_points=401, omega=0.5, seed=datetime.microsecond)
    # sta_cont = run_sim_continuous(N=10000, T=1.0, dt=0.1, phys=phys, p0_in=p0_in, demon_init="u", seed=7)
    
    # print("Empirical continuous incoming:", sta_cont["incoming"])
    # print("Empirical continuous outgoing:", sta_cont["outgoing"])
    # print(f"Phi_emp continuous = {sta_cont['outgoing']['p1'] - sta_cont['incoming']['p1']:.4f}")
    # plot_phi_per_parameter_slice(N=3000, demon_init="u", n_points=200, delta_fixed=1.0,
    #     fixed_params={"Th": 1.6, "Tc": 1.0, "DeltaE": 1.0, "gamma_hot": 1.0},
    #     param_name="tau", params_range=(0.01, 2000.0), seed=datetime.microsecond)
    plot_phi_per_parameter_slice(N=3000, demon_init="u", n_points=100, delta_fixed=0.192,
        fixed_params={"Th": 1.6, "Tc": 1.0, "gamma_hot": 1.0,  "DeltaE": 2.5},
        param_name="tau", params_range=(0.01, 15.0), seed=datetime.microsecond, gillespie=True, plotNegativeDelta=True, useDetSolution=True)
    # sta_cont = run_sim_continuous(N=10000, T=2.0, dt=0.01, phys=phys, p0_in=p0_in, demon_init="u", seed=7,cumulative=True)
    # print("Empirical continuous (cumulative) incoming:", sta_cont["incoming"])
    # print("Empirical continuous (cumulative) outgoing:", sta_cont["outgoing"])
    # print(f"Phi_emp continuous (cumulative) = {sta_cont['outgoing']['p1'] - sta_cont['incoming']['p1']:.4f}")
