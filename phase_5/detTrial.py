# ndemon.py
# Maxwell's Refrigerator: n-state demon with uniform level spacing.
# Dependencies: numpy (required), scipy (optional but recommended for expm)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np

try:
    from scipy.linalg import expm  # best/robust
except Exception:
    expm = None  # we'll fall back to a simple eigen-decomposition if needed


# ==========
# Utilities
# ==========

def shannon_entropy_bernoulli(p1: float) -> float:
    """Shannon entropy (nats) of Bernoulli(p1). Handles edge cases safely."""
    p1 = min(max(p1, 1e-15), 1 - 1e-15)
    p0 = 1.0 - p1
    return -(p0 * math.log(p0) + p1 * math.log(p1))


def column_stochastic_fix(R: np.ndarray) -> None:
    """Make each column of R sum to 0 by adjusting the diagonal in place."""
    # R[i,j] is rate from j -> i (column-sum-zero convention)
    np.fill_diagonal(R, 0.0)
    col_sums = R.sum(axis=0)
    for j in range(R.shape[1]):
        R[j, j] = -col_sums[j]


def safe_expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential with scipy if available; fallback to eig (small/medium)."""
    if expm is not None:
        return expm(A)
    # Fallback: eigen-decomposition (works well for diagonalizable A)
    w, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    return V @ np.diag(np.exp(w)) @ Vinv


# ============================
# Physical parameters helpers
# ============================

@dataclass(frozen=True)
class Thermo:
    DeltaE: float           # energy spacing between consecutive demon levels
    Th: float               # hot bath temperature
    Tc: float               # cold bath temperature
    kB: float = 1.0         # Boltzmann constant (set 1 by default)

    @property
    def beta_h(self) -> float:
        return 1.0 / (self.kB * self.Th)

    @property
    def beta_c(self) -> float:
        return 1.0 / (self.kB * self.Tc)

    @property
    def sigma(self) -> float:
        # tanh(beta_h * DeltaE / 2)
        return math.tanh(0.5 * self.beta_h * self.DeltaE)

    @property
    def omega(self) -> float:
        # tanh(beta_c * DeltaE / 2)
        return math.tanh(0.5 * self.beta_c * self.DeltaE)

    @property
    def epsilon(self) -> float:
        # (omega - sigma) / (1 - omega*sigma) = tanh((beta_c - beta_h) * DeltaE / 2)
        s, w = self.sigma, self.omega
        return (w - s) / (1.0 - w * s)


# =========================================
# Build the 2n×2n generator R for n levels
# =========================================

def build_R(n: int, gamma: float, sigma: float, omega: float) -> np.ndarray:
    """
    Build the joint generator R (size 2n x 2n) for demon levels k=0..n-1 and bit b in {0,1}.
    State ordering: (0,0),(1,0),...,(n-1,0) | (0,1),(1,1),...,(n-1,1).
    Off-diagonals are nonnegative rates; columns sum to zero.
    Intrinsic (hot): (k,b) <-> (k±1,b) with bias sigma, speed gamma.
    Cooperative (cold): (k,0) <-> (k+1,1) with bias omega, base rate 1.
    """
    assert n >= 2, "Use n >= 2."
    d = 2 * n
    R = np.zeros((d, d), dtype=float)

    # helpers to index states
    def idx(k: int, b: int) -> int:
        return k + (0 if b == 0 else n)

    # Intrinsic hot-bath flips (nearest neighbor in k, fixed bit)
    up_rate = gamma * (1.0 - sigma)  # k -> k+1
    dn_rate = gamma * (1.0 + sigma)  # k -> k-1

    for b in (0, 1):
        # k -> k+1
        for k in range(n - 1):
            i_to = idx(k + 1, b)
            j_from = idx(k, b)
            R[i_to, j_from] += up_rate  # from (k,b) to (k+1,b)
        # k -> k-1
        for k in range(1, n):
            i_to = idx(k - 1, b)
            j_from = idx(k, b)
            R[i_to, j_from] += dn_rate  # from (k,b) to (k-1,b)

    # Cooperative cold-bath flips: (k,0) <-> (k+1,1)
    # forward: (k,0)->(k+1,1) has rate (1 - omega)
    # reverse: (k+1,1)->(k,0) has rate (1 + omega)
    fwd = 1.0 - omega
    rev = 1.0 + omega
    for k in range(n - 1):
        # forward
        R[idx(k + 1, 1), idx(k, 0)] += fwd
        # reverse
        R[idx(k, 0), idx(k + 1, 1)] += rev

    # Fix diagonals to make each column sum to zero
    column_stochastic_fix(R)
    return R


# ========================================
# One-window evolution and steady solution
# ========================================

def build_injector_M(n: int, delta: float) -> np.ndarray:
    """
    Injector M in R^{2n x n}.
    Given demon start distribution pD_start (n,1), the joint start vector is p_joint_in = M @ pD_start.
    Bits are i.i.d. with p0=(1+delta)/2, p1=(1-delta)/2.
    """
    p0 = 0.5 * (1.0 + delta)
    p1 = 1.0 - p0
    M = np.zeros((2 * n, n), dtype=float)
    # top block (bit 0)
    M[0:n, 0:n] = p0 * np.eye(n)
    # bottom block (bit 1)
    M[n:2 * n, 0:n] = p1 * np.eye(n)
    return M


def projectors(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (P_D, P_B).
    P_D ∈ R^{n x 2n} marginalizes joint to demon.
    P_B ∈ R^{2 x 2n} marginalizes joint to bit.
    """
    P_D = np.hstack([np.eye(n), np.eye(n)])
    ones = np.ones((1, n))
    P_B = np.vstack([np.hstack([ones, np.zeros((1, n))]),
                     np.hstack([np.zeros((1, n)), ones])])
    return P_D, P_B


def one_window_map_T(R: np.ndarray, tau: float, M: np.ndarray, P_D: np.ndarray) -> np.ndarray:
    """Return T = P_D @ exp(R*tau) @ M (n x n)."""
    U = safe_expm(R * tau)
    return P_D @ U @ M


def periodic_steady_state(T: np.ndarray, tol: float = 1e-13, maxit: int = 100000) -> np.ndarray:
    """
    Right Perron vector of a positive stochastic map T (n x n).
    Power iteration with normalization; returns a probability vector (sum=1).
    """
    n = T.shape[0]
    p = np.ones(n) / n
    for _ in range(maxit):
        p_next = T @ p
        s = p_next.sum()
        if s <= 0:
            raise RuntimeError("Non-positive vector in iteration; check R/M/T.")
        p_next /= s
        if np.linalg.norm(p_next - p, 1) < tol:
            return p_next
        p = p_next
    raise RuntimeError("Steady-state iteration did not converge.")


def outgoing_bit_stats(R: np.ndarray, tau: float, M: np.ndarray, P_B: np.ndarray, pD_start: np.ndarray) -> Tuple[float, float]:
    """Return (p0_out, p1_out)."""
    U = safe_expm(R * tau)
    p_joint_in = M @ pD_start
    p_joint_out = U @ p_joint_in
    pB_out = P_B @ p_joint_out
    s = pB_out.sum()
    if s <= 0:
        raise RuntimeError("Outgoing bit vector not positive; check inputs.")
    pB_out /= s
    return float(pB_out[0]), float(pB_out[1])


# ==========================
# Fast-demon (gamma -> inf)
# ==========================

def phi_fast_demon(delta: float, sigma: float, omega: float, tau: float) -> float:
    """
    Analytic n-independent fast-demon formula:
      Phi = 0.5 * (delta - epsilon) * (1 - exp(-(1 - sigma*omega)*tau))
    where epsilon = (omega - sigma) / (1 - omega*sigma).
    """
    eps = (omega - sigma) / (1.0 - omega * sigma)
    factor = 1.0 - math.exp(-(1.0 - sigma * omega) * tau)
    return 0.5 * (delta - eps) * factor


# =======================
# High-level convenience
# =======================

@dataclass
class RunResult:
    pD_ss: np.ndarray
    pB_in: Tuple[float, float]
    pB_out: Tuple[float, float]
    Phi: float
    Q_c_to_h: float
    DeltaS_bits: float
    epsilon: float


def run_ndemon(
    n: int,
    gamma: float,
    thermo: Thermo,
    delta: float,
    tau: float
) -> RunResult:
    """
    Full pipeline:
      - build R
      - T = P_D exp(R*tau) M
      - find periodic steady demon pD_ss
      - compute outgoing bit stats and thermodynamic outputs
    """
    sigma, omega, eps = thermo.sigma, thermo.omega, thermo.epsilon
    R = build_R(n, gamma=gamma, sigma=sigma, omega=omega)
    M = build_injector_M(n, delta=delta)
    P_D, P_B = projectors(n)

    T = one_window_map_T(R, tau, M, P_D)
    pD_ss = periodic_steady_state(T)

    p0_in, p1_in = (0.5 * (1 + delta), 0.5 * (1 - delta))
    p0_out, p1_out = outgoing_bit_stats(R, tau, M, P_B, pD_ss)

    Phi = p1_out - p1_in                       # net new 1's per window
    Q_c_to_h = Phi * thermo.DeltaE             # heat from cold to hot per window
    S_in = shannon_entropy_bernoulli(p1_in)
    S_out = shannon_entropy_bernoulli(p1_out)
    DeltaS_bits = S_out - S_in

    return RunResult(
        pD_ss=pD_ss,
        pB_in=(p0_in, p1_in),
        pB_out=(p0_out, p1_out),
        Phi=Phi,
        Q_c_to_h=Q_c_to_h,
        DeltaS_bits=DeltaS_bits,
        epsilon=eps,
    )


# ===========
# Example run
# ===========

def plot_ndemon_vs_output_bias(n_min, n_max, step, tau):
    import matplotlib.pyplot as plt

    ns = list(range(n_min, n_max + 1, step))
    biases = []
    fast_biases = []
    for n in ns:
        thermo = Thermo(DeltaE=1/(n-1), Th=5.6, Tc=1.0)
        res = run_ndemon(n=n, gamma=1.0, thermo=thermo, delta=1.0, tau=tau)
        bias_out = res.pB_out[0] - res.pB_out[1]
        fast_phi = phi_fast_demon(delta=1.0, sigma=thermo.sigma, omega=thermo.omega, tau=tau)
        # phi = (p1_out - p1_in) => bias_out = phi - p1_in +p0_out 
        fast_bias = (-fast_phi + res.pB_out[0])  # since p0_in=1.0, p1_in=0.0
        fast_biases.append(fast_bias)
        biases.append(bias_out)

    plt.figure(figsize=(8, 5))
    plt.plot(ns, biases, marker='o')
    plt.xlabel('Demon Levels (n)')
    plt.ylabel('Output Bit Bias (p0 - p1)')
    plt.title('n-State Demon Output Bit Bias vs Number of Levels')
    plt.grid(True)
    # Also plot fast-demon limit
    plt.plot(ns, fast_biases, marker='x', linestyle='--', label='Fast-Demon Limit')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Example: n=3 levels, moderate rates, clear temperature gap
    # n = 2
    # thermo = Thermo(DeltaE=1/(n-1), Th=5.6, Tc=1.0)  # kB=1 units
    # gamma = 1.0
    # delta = 1.0        # incoming bit bias (more zeros than ones)
    # tau = 1.0

    # res = run_ndemon(n=n, gamma=gamma, thermo=thermo, delta=delta, tau=tau)
    # print("== n-state demon example ==")
    # print(f"n={n}, gamma={gamma}, tau={tau}")
    # print(f"T_h={thermo.Th}, T_c={thermo.Tc}, DeltaE={thermo.DeltaE}")
    # print(f"sigma={thermo.sigma:.6f}, omega={thermo.omega:.6f}, epsilon={res.epsilon:.6f}")
    # print(f"bit_in  (p0,p1)=({res.pB_in[0]:.6f}, {res.pB_in[1]:.6f})")
    # print(f"bit_out (p0,p1)=({res.pB_out[0]:.6f}, {res.pB_out[1]:.6f})")
    # print(f"Output bias: {res.pB_out[0] -res.pB_out[1]:.6f}")
    # print(f"Phi (Δ1 per window) = {res.Phi:.6e}")
    # print(f"Q_c->h per window   = {res.Q_c_to_h:.6e}")
    # print(f"ΔS_bits (nats)      = {res.DeltaS_bits:.6e}")
    # print(f"demon steady state  = {res.pD_ss}")

    # # Fast-demon analytical check (independent of n)
    # phi_fd = phi_fast_demon(delta=delta, sigma=thermo.sigma, omega=thermo.omega, tau=tau)
    # print(f"Phi_fast_demon      = {phi_fd:.6e}  (sanity check)")
    plot_ndemon_vs_output_bias(n_min=2, n_max=200, step=5, tau=30.0)