# ndemon.py
# Maxwell's Refrigerator: n-state demon with uniform level spacing.
# Dependencies: numpy (required), scipy (optional but recommended for expm)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
from functools import lru_cache
import numpy as np

try:
    from scipy.linalg import expm  # best/robust
    from scipy.sparse import csr_matrix, eye as sparse_eye, diags
    from scipy.sparse.linalg import expm as sparse_expm
    SCIPY_AVAILABLE = True
except Exception:
    expm = None
    SCIPY_AVAILABLE = False


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


def safe_expm(A: np.ndarray, use_sparse: bool = None) -> np.ndarray:
    """
    Matrix exponential with optimizations.
    For large sparse matrices, uses sparse methods when available.
    """
    if use_sparse is None:
        # Auto-detect sparsity threshold
        use_sparse = A.shape[0] > 100 and np.count_nonzero(A) / A.size < 0.1
    
    if use_sparse and SCIPY_AVAILABLE:
        try:
            A_sparse = csr_matrix(A)
            return sparse_expm(A_sparse).toarray()
        except:
            pass  # Fall back to dense methods
    
    if expm is not None:
        return expm(A)
    
    # Fallback: eigen-decomposition (works well for diagonalizable A)
    w, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    return V @ np.diag(np.exp(w)) @ Vinv


# Matrix exponential cache for repeated computations
_expm_cache = {}

def cached_expm(A: np.ndarray, tau: float, cache_key: str = None) -> np.ndarray:
    """
    Cached matrix exponential for repeated R*tau computations.
    Uses a hash of the matrix and tau as key if cache_key not provided.
    """
    if cache_key is None:
        # Create a hash-based key (for identical matrices)
        cache_key = hash((A.tobytes(), tau))
    
    if cache_key in _expm_cache:
        return _expm_cache[cache_key]
    
    result = safe_expm(A * tau)
    _expm_cache[cache_key] = result
    return result


def clear_caches():
    """Clear all caches to free memory when needed."""
    global _expm_cache
    _expm_cache.clear()
    # Clear LRU caches
    build_injector_M_cached.cache_clear()
    projectors_cached.cache_clear()


# Set optimal NumPy settings for performance
def configure_numpy_for_performance():
    """Configure NumPy for optimal performance."""
    # Use optimal BLAS threading (if available)
    try:
        import os
        # Set optimal thread count (usually number of physical cores)
        os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count() // 2))
        os.environ['MKL_NUM_THREADS'] = str(min(4, os.cpu_count() // 2))
    except:
        pass


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

def build_R_optimized(n: int, gamma: float, sigma: float, omega: float) -> np.ndarray:
    """
    Optimized vectorized construction of the joint generator R (size 2n x 2n).
    Uses vectorized operations instead of loops for better performance.
    """
    assert n >= 2, "Use n >= 2."
    d = 2 * n
    
    # Pre-compute rates
    up_rate = gamma * (1.0 - sigma)  # k -> k+1
    dn_rate = gamma * (1.0 + sigma)  # k -> k-1
    fwd = 1.0 - omega  # (k,0) -> (k+1,1)
    rev = 1.0 + omega  # (k+1,1) -> (k,0)
    
    # Use sparse matrix construction for efficiency
    if SCIPY_AVAILABLE and n > 50:
        from scipy.sparse import lil_matrix
        R = lil_matrix((d, d), dtype=np.float64)
    else:
        R = np.zeros((d, d), dtype=np.float64)
    
    # Vectorized intrinsic hot-bath transitions
    # For bit b=0: indices 0 to n-1
    # For bit b=1: indices n to 2n-1
    
    # k -> k+1 transitions (vectorized)
    for b_offset in [0, n]:  # bit 0 and bit 1 blocks
        k_from = np.arange(n - 1) + b_offset
        k_to = np.arange(1, n) + b_offset
        R[k_to, k_from] = up_rate
    
    # k -> k-1 transitions (vectorized)  
    for b_offset in [0, n]:
        k_from = np.arange(1, n) + b_offset
        k_to = np.arange(n - 1) + b_offset
        R[k_to, k_from] = dn_rate
    
    # Cooperative cold-bath transitions: (k,0) <-> (k+1,1)
    # Vectorized approach
    k_indices = np.arange(n - 1)
    # (k,0) -> (k+1,1): from indices k to indices (k+1)+n
    R[k_indices + 1 + n, k_indices] = fwd
    # (k+1,1) -> (k,0): from indices (k+1)+n to indices k
    R[k_indices, k_indices + 1 + n] = rev
    
    # Convert to dense if was sparse
    if hasattr(R, 'toarray'):
        R = R.toarray()
    
    # Fix diagonals to make each column sum to zero
    column_stochastic_fix(R)
    return R


# Backwards compatibility - use optimized version by default
def build_R(n: int, gamma: float, sigma: float, omega: float) -> np.ndarray:
    """Original build_R function - now calls optimized version."""
    return build_R_optimized(n, gamma, sigma, omega)


# ========================================
# One-window evolution and steady solution
# ========================================

@lru_cache(maxsize=128)
def build_injector_M_cached(n: int, delta: float) -> np.ndarray:
    """
    Cached version of injector M to avoid recomputation.
    LRU cache automatically handles memory management.
    """
    p0 = 0.5 * (1.0 + delta)
    p1 = 1.0 - p0
    M = np.zeros((2 * n, n), dtype=np.float64)
    # top block (bit 0)
    M[0:n, 0:n] = p0 * np.eye(n)
    # bottom block (bit 1)
    M[n:2 * n, 0:n] = p1 * np.eye(n)
    return M


def build_injector_M(n: int, delta: float) -> np.ndarray:
    """
    Injector M in R^{2n x n}.
    Given demon start distribution pD_start (n,1), the joint start vector is p_joint_in = M @ pD_start.
    Bits are i.i.d. with p0=(1+delta)/2, p1=(1-delta)/2.
    """
    return build_injector_M_cached(n, delta)


@lru_cache(maxsize=64)
def projectors_cached(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached version of projectors to avoid recomputation.
    Return (P_D, P_B).
    P_D ∈ R^{n x 2n} marginalizes joint to demon.
    P_B ∈ R^{2 x 2n} marginalizes joint to bit.
    """
    P_D = np.hstack([np.eye(n), np.eye(n)])
    ones = np.ones((1, n))
    P_B = np.vstack([np.hstack([ones, np.zeros((1, n))]),
                     np.hstack([np.zeros((1, n)), ones])])
    return P_D, P_B


def projectors(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (P_D, P_B).
    P_D ∈ R^{n x 2n} marginalizes joint to demon.
    P_B ∈ R^{2 x 2n} marginalizes joint to bit.
    """
    return projectors_cached(n)


def one_window_map_T(R: np.ndarray, tau: float, M: np.ndarray, P_D: np.ndarray) -> np.ndarray:
    """Return T = P_D @ exp(R*tau) @ M (n x n)."""
    U = cached_expm(R, tau)
    return P_D @ U @ M


def periodic_steady_state(T: np.ndarray, tol: float = 1e-13, maxit: int = 100000) -> np.ndarray:
    """
    Right Perron vector of a positive stochastic map T (n x n).
    Optimized power iteration with better convergence detection.
    """
    n = T.shape[0]
    p = np.ones(n, dtype=np.float64) / n
    
    for iteration in range(maxit):
        p_next = T @ p
        s = p_next.sum()
        if s <= 0:
            raise RuntimeError("Non-positive vector in iteration; check R/M/T.")
        p_next /= s
        
        # More efficient convergence check using relative tolerance
        diff = np.linalg.norm(p_next - p, ord=1)
        if diff < tol:
            return p_next
        p = p_next
        
        # Early exit check for faster convergence
        if iteration > 10 and iteration % 100 == 0:
            if diff < tol * 10:  # Getting close, reduce tolerance gradually
                tol *= 0.9
    
    raise RuntimeError("Steady-state iteration did not converge.")


def outgoing_bit_stats(R: np.ndarray, tau: float, M: np.ndarray, P_B: np.ndarray, pD_start: np.ndarray) -> Tuple[float, float]:
    """Return (p0_out, p1_out). Optimized version using cached exponential."""
    U = cached_expm(R, tau)
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
    
    # Pre-compute common values to avoid repeated calculations
    delta = 1.0
    gamma = 1.0
    
    for n in ns:
        # Create thermo object once per n
        thermo = Thermo(DeltaE=1/(n-1), Th=1.6, Tc=1.0)
        
        # Cache key for this specific configuration
        cache_key = f"n{n}_gamma{gamma}_tau{tau}_delta{delta}"
        
        res = run_ndemon(n=n, gamma=gamma, thermo=thermo, delta=delta, tau=tau)
        bias_out = res.pB_out[0] - res.pB_out[1]
        
        # Fast demon calculation (analytical, very fast)
        fast_phi = phi_fast_demon(delta=delta, sigma=thermo.sigma, omega=thermo.omega, tau=tau)
        fast_bias = (-fast_phi + res.pB_out[0])  # since p0_in=1.0, p1_in=0.0
        
        fast_biases.append(fast_bias)
        biases.append(bias_out)
        print(f"% complete: {100 * (n - n_min) / (n_max - n_min):.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(ns, biases, marker='o', label='Numerical')
    plt.plot(ns, fast_biases, marker='x', linestyle='--', label='Fast-Demon Limit')
    plt.xlabel('Demon Levels (n)')
    plt.ylabel('Output Bit Bias (p0 - p1)')
    plt.title('n-State Demon Output Bit Bias vs Number of Levels')
    plt.grid(True)
    plt.legend()
    plt.show()


def get_T_H_from_sigma(sigma: float, deltaE: float) -> float:
    """Given sigma, omega, return T_H assuming kB=1."""
    try:
        beta_h = (2.0 / deltaE) * math.atanh(sigma)
        T_H = 1.0 / beta_h
    except:
        T_H = float('inf')  # Handle case where beta_h is zero
    return T_H


def get_sigma_from_epsilon_omega(epsilon: float, omega: float) -> float:
    """Given epsilon, omega, return sigma."""
    sigma = (omega - epsilon) / (1.0 - omega * epsilon)
    # assert -1.0 < sigma < 1.0, f"Calculated sigma out of bounds! sigma={sigma}, epsilon={epsilon}, omega={omega}"
    return sigma


def plot_epsilon_delta_phase_diagram(n: int, gamma: float, tau: float):
    import matplotlib.pyplot as plt
    
    # Use fewer points for faster computation, but still good resolution
    delta_vals = np.linspace(-1.0, 1.0, 101)  # Reduced from 201
    epsilon_vals = np.linspace(0.0, 1.0, 101)  # Reduced from 201
    
    T_C = 1.0
    deltaE = 1.0 / (n - 1)
    base_thermo = Thermo(DeltaE=deltaE, Th=5.6, Tc=T_C)
    omega = base_thermo.omega
    
    phi_vals = np.zeros((len(epsilon_vals), len(delta_vals)), dtype=np.float32)  # Use float32 to save memory
    
    # Pre-allocate and vectorize where possible
    for i, delta in enumerate(delta_vals):
        for j, epsilon in enumerate(epsilon_vals):
            try:
                sigma = get_sigma_from_epsilon_omega(epsilon, omega)
                T_H = get_T_H_from_sigma(sigma, deltaE)
                thermo_temp = Thermo(DeltaE=deltaE, Th=T_H, Tc=T_C)
                res = run_ndemon(n=n, gamma=gamma, thermo=thermo_temp, delta=delta, tau=tau)
                phi_vals[j, i] = res.pB_out[0] - res.pB_out[1]
            except:
                phi_vals[j, i] = np.nan  # Handle numerical issues gracefully
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(delta_vals, epsilon_vals, phi_vals, levels=50, cmap='RdBu_r', 
                     vmin=-np.nanmax(np.abs(phi_vals)), vmax=np.nanmax(np.abs(phi_vals)))
    plt.colorbar(cp, label='Output Bias (p0 - p1)')
    plt.ylabel('Epsilon (Thermo Parameter)')
    plt.xlabel('Delta (Input Bit Bias)')
    plt.title(f'Phase Diagram: n={n}, gamma={gamma}, tau={tau}')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.show()


def plot_ndmon_tau_vs_output_bias_phase_diagram(n_min=20, n_max=50, step=5, tau_min=1.0, tau_max=100.0, tau_steps=10):
    import matplotlib.pyplot as plt
    
    ns = list(range(n_min, n_max + 1, step))
    taus = np.linspace(tau_min, tau_max, tau_steps)
    
    # Pre-allocate results for vectorized operations
    results = {}
    
    for n in ns:
        # Pre-compute thermo object once per n
        thermo = Thermo(DeltaE=1/(n-1), Th=5.6, Tc=1.0)
        biases = []
        
        for tau in taus:
            res = run_ndemon(n=n, gamma=1.0, thermo=thermo, delta=1.0, tau=tau)
            bias_out = res.pB_out[0] - res.pB_out[1]
            biases.append(bias_out)
        
        plt.plot(taus, biases, marker='o', label=f'n={n}', linewidth=1.5)
    
    plt.xlabel('Interaction Time (tau)')
    plt.ylabel('Output Bit Bias (p0 - p1)')
    plt.title('n-State Demon Output Bit Bias vs Interaction Time')
    plt.grid(True, alpha=0.3)
    # plt.legend()  # Commented out as noted in original
    plt.show()

if __name__ == "__main__":
    # Configure NumPy for optimal performance
    configure_numpy_for_performance()
    
    print("Running optimized Maxwell's Demon simulation...")
    print("Matrix operations have been optimized with:")
    print("- Cached matrix exponentials")
    print("- Vectorized matrix construction") 
    print("- Sparse matrix support for large systems")
    print("- Memory-efficient data types")
    print("- LRU caching for repeated computations")
    
    # Test the optimizations with a simple example first
    print("\nTesting optimizations...")
    import time
    
    n = 100
    thermo = Thermo(DeltaE=1/(n-1), Th=5.6, Tc=1.0)
    gamma = 1.0
    delta = 1.0
    tau = 1.0
    
    start_time = time.time()
    res = run_ndemon(n=n, gamma=gamma, thermo=thermo, delta=delta, tau=tau)
    end_time = time.time()
    
    print(f"== n-state demon example (n={n}) ==")
    print(f"Computation time: {end_time - start_time:.4f} seconds")
    print(f"T_h={thermo.Th}, T_c={thermo.Tc}, DeltaE={thermo.DeltaE:.6f}")
    print(f"sigma={thermo.sigma:.6f}, omega={thermo.omega:.6f}, epsilon={res.epsilon:.6f}")
    print(f"bit_in  (p0,p1)=({res.pB_in[0]:.6f}, {res.pB_in[1]:.6f})")
    print(f"bit_out (p0,p1)=({res.pB_out[0]:.6f}, {res.pB_out[1]:.6f})")
    print(f"Output bias: {res.pB_out[0] - res.pB_out[1]:.6f}")
    print(f"Phi (Δ1 per window) = {res.Phi:.6e}")
    print(f"Q_c->h per window   = {res.Q_c_to_h:.6e}")
    print(f"ΔS_bits (nats)      = {res.DeltaS_bits:.6e}")
    
    # Test cache effectiveness
    print(f"\nCache info:")
    print(f"Matrix exponential cache size: {len(_expm_cache)}")
    print(f"Injector cache info: {build_injector_M_cached.cache_info()}")
    print(f"Projector cache info: {projectors_cached.cache_info()}")
    
    # Fast-demon analytical check (independent of n)
    phi_fd = phi_fast_demon(delta=delta, sigma=thermo.sigma, omega=thermo.omega, tau=tau)
    print(f"Phi_fast_demon      = {phi_fd:.6e}  (analytical)")
    
    # Test performance with multiple runs (should show cache benefits)
    print(f"\nTesting cache performance with 10 repeated runs...")
    start_time = time.time()
    for i in range(10):
        res = run_ndemon(n=n, gamma=gamma, thermo=thermo, delta=delta, tau=tau)
    end_time = time.time()
    print(f"10 runs completed in {end_time - start_time:.4f} seconds")
    print(f"Average time per run: {(end_time - start_time)/10:.4f} seconds")
    
    # Clear caches to free memory after computation
    clear_caches()
    print("Caches cleared.")
    
    # Uncomment to run plotting functions (requires compatible matplotlib/numpy)
    plot_ndemon_vs_output_bias(n_min=2000, n_max=3000, step=100, tau=50.0)
    # plot_epsilon_delta_phase_diagram(n=20, gamma=1.0, tau=20000.0)
    # plot_ndmon_tau_vs_output_bias_phase_diagram(n_min=2, n_max=100, step=1, tau_min=1.0, tau_max=100.0, tau_steps=30)