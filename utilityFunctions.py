import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
# -----------------------------
# Utilities: state indexing
# Order joint states as [u0, d0, u1, d1] to match the supplemental
# We'll treat probability vectors as column vectors (shape (4,1))
# and use the convention dp/dt = R p, with each column of R summing to 0.
# -----------------------------
kB = 1 # Boltzmann constant in our units (energy/temperature)
U0, D0, U1, D1 = 0, 1, 2, 3  # states in rate matrix


# PD projects joint end -> demon end (2x4)
PD = np.array([
    [1, 0, 1, 0],  # pu = u0 + u1
    [0, 1, 0, 1],  # pd = d0 + d1
], dtype=float)

# PB projects joint end -> bit end (2x4)
PB = np.array([
    [1, 1, 0, 0],  # p0 = u0 + d0
    [0, 0, 1, 1],  # p1 = u1 + d1
], dtype=float)
def normalize(v):
    s = v.sum()
    return v / s if s > 0 else v
def sample_categorical(prob_col, rng):
    """Sample index i according to probabilities prob_col (1D)."""
    # Numerical safety:
    pc = np.clip(prob_col, 0.0, 1.0)
    s = pc.sum()
    if s <= 0:
        raise ValueError("Invalid probability column in sampling.")
    pc = pc / s
    return rng.choice(len(pc), p=pc)
def temps_to_sigma_omega(Tc, Th, DeltaE, k_B=kB):
    """Return (sigma, omega, epsilon) from temperatures in Kelvin and ΔE (J)."""
    beta_h = 1.0 / (k_B * Th)
    beta_c = 1.0 / (k_B * Tc)
    sigma = np.tanh(0.5 * beta_h * DeltaE)
    omega = np.tanh(0.5 * beta_c * DeltaE)
    epsilon = (omega - sigma) / (1.0 - omega * sigma)  # equals tanh(0.5*(beta_c - beta_h)*ΔE)
    return float(sigma), float(omega), float(epsilon)
def sigma_omega_to_temps(sigma, omega, DeltaE, k_B=kB):
    """Invert to (Tc, Th) from (sigma, omega) and ΔE."""
    # Clip to avoid ±1 where arctanh diverges
    s = np.clip(sigma, -0.999999999, 0.999999999)
    w = np.clip(omega, -0.999999999, 0.999999999)
    Th = DeltaE / (2.0 * k_B * np.arctanh(s))
    Tc = DeltaE / (2.0 * k_B * np.arctanh(w))
    return float(Tc), float(Th)
def build_R(gamma, sigma, omega):
    """
    Construct the 4x4 generator R for joint states [u0, d0, u1, d1].
    Off-diagonals: rates FROM column-state TO row-state.
    Diagonals set so each column sums to zero.
    
    Intrinsic (hot): d<->u for each bit:
      d -> u: gamma*(1 - sigma)
      u -> d: gamma*(1 + sigma)
    Cooperative (cold): 0d <-> 1u:
      d0 -> u1: (1 - omega)
      u1 -> d0: (1 + omega)
    """
    R = np.zeros((4,4), dtype=float)

    # Intrinsic transitions for bit=0
    R[D0, U0] = gamma*(1 + sigma)  # u0 -> d0
    R[U0, D0] = gamma*(1 - sigma)  # d0 -> u0

    # Intrinsic transitions for bit=1
    R[D1, U1] = gamma*(1 + sigma)  # u1 -> d1
    R[U1, D1] = gamma*(1 - sigma)  # d1 -> u1

    # Cooperative transitions (cold bath): 0d <-> 1u
    R[D0, U1] = (1 + omega)        # u1 -> d0
    R[U1, D0] = (1 - omega)        # d0 -> u1

    # Diagonals: columns sum to zero
    for j in range(4):
        R[j, j] = -np.sum(R[:, j]) + R[j, j]
    return R
def steady_demon_via_iteration(T, tol=1e-12, maxit=100000):
    p = np.array([0.5, 0.5], dtype=float)
    for _ in range(maxit):
        p_next = T @ p
        s = p_next.sum()
        if s <= 0:
            raise RuntimeError("T produced non-positive vector; check R/M/PD.")
        p_next /= s
        if np.linalg.norm(p_next - p, 1) < tol:
            return p_next
        p = p_next
    raise RuntimeError("Steady state iteration did not converge.")

def project_bit(p_joint):
    """Return bit marginal (p0, p1) from joint p=[u0,d0,u1,d1]."""
    p0 = p_joint[U0] + p_joint[D0]
    p1 = p_joint[U1] + p_joint[D1]
    return np.array([p0, p1])

def project_demon(p_joint):
    """Return demon marginal (pu, pd) from joint p=[u0,d0,u1,d1]."""
    pu = p_joint[U0] + p_joint[U1]
    pd = p_joint[D0] + p_joint[D1]
    return np.array([pu, pd])
def deterministic_solution(gamma, sigma, omega, tau, delta):
    """
    Given parameters and incoming bit bias delta = p0 - p1,
    compute:
      - steady demon distribution at interval start,
      - outgoing bit distribution,
      - Phi (net production of 1s per bit),
      - Q_{c->h} per bit (in units of DeltaE; you can multiply later),
      - T matrix (2x2) mapping demon start -> demon end,
      - P_tau = exp(R*tau) for reuse.
    """
    # Incoming bit distribution:
    p0 = (1 + delta)/2
    p1 = (1 - delta)/2
    assert 0 <= p0 <= 1 and 0 <= p1 <= 1, "delta must be in [-1,1]"

    R = build_R(gamma, sigma, omega)

    print("Rate matrix R:\n", R)
    P_tau = expm(R * tau)

    # M maps demon marginal at start to joint start (uncorrelated with bit):
    # p_joint_start = M @ pD_start, where pD_start = [pu, pd]
    # M is 4x2: rows joint states; cols demon states (u,d)
    # For bit=0: weight p0 on u0 and d0; for bit=1: weight p1 on u1 and d1.
    M = np.array([
        [p0, 0.0],   # u0 from demon u
        [0.0, p0],   # d0 from demon d
        [p1, 0.0],   # u1 from demon u
        [0.0, p1],   # d1 from demon d
    ], dtype=float)

    # PD projects joint end -> demon end (2x4)
    PD = np.array([
        [1, 0, 1, 0],  # pu = u0 + u1
        [0, 1, 0, 1],  # pd = d0 + d1
    ], dtype=float)

    # PB projects joint end -> bit end (2x4)
    PB = np.array([
        [1, 1, 0, 0],  # p0 = u0 + d0
        [0, 0, 1, 1],  # p1 = u1 + d1
    ], dtype=float)

    # One-interval demon map: pD_end = T @ pD_start
    T = PD @ (P_tau @ M)  # 2x2

    # Find periodic steady state pD_start: T p = p, normalized
    vals, vecs = eig(T)
    # Pick eigenvector with eigenvalue closest to 1
    idx = np.argmin(np.abs(vals - 1))
    # pD_start = np.real(vecs[:, idx])
    # pD_start = normalize(np.maximum(pD_start, 0))
    pD_start = steady_demon_via_iteration(T)
    # Outgoing bit distribution:
    p_joint_start = M @ pD_start          # 4,
    p_joint_end = P_tau @ p_joint_start   # 4,
    pB_out = PB @ p_joint_end             # 2,
    pB_out = normalize(pB_out)

    # Phi = p1_out - p1_in
    Phi = pB_out[1] - p1

    # Heat per bit in units of DeltaE: Q_c->h = Phi * DeltaE  (we return Phi only)
    return {
        "pD_start": pD_start,       # [pu, pd]
        "pB_out": pB_out,           # [p0_out, p1_out]
        "Phi": Phi,
        "T": T,
        "P_tau": P_tau,
        "R": R,
        "p_in": np.array([p0, p1]),
    }
def monte_carlo_tape(N, gamma, sigma, omega, tau, delta, seed=0):
    """
    Simulate N bits of the tape explicitly.
    - Incoming bits sampled iid with p1 = (1-delta)/2.
    - Demon state persists across intervals.
    - Within each interval, evolve joint state by sampling from P_tau columns.
    Returns summary statistics and history arrays for optional inspection.
    """
    rng = np.random.default_rng(seed)

    # Precompute:
    R = build_R(gamma, sigma, omega)
    P_tau = expm(R * tau)

    p0 = (1 + delta)/2
    p1 = (1 - delta)/2

    # Initialize demon; choose e.g. u with its hot-bath equilibrium if you like
    # but any initialization works; transient dies out.
    demon_state = rng.choice([0,1])  # 0=u, 1=d

    outgoing_bits = []
    incoming_bits = []

    # For convergence tracking:
    running_phi = []  # empirical Phi_n over first n bits

    count_out_1 = 0

    for n in range(1, N+1):
        # Sample incoming bit:
        b_in = rng.choice([0,1], p=[p0, p1])
        incoming_bits.append(b_in)

        # Build column index (joint start):
        # joint state index j from (demon_state, b_in)
        if demon_state == 0 and b_in == 0: j = U0
        elif demon_state == 1 and b_in == 0: j = D0
        elif demon_state == 0 and b_in == 1: j = U1
        else: j = D1

        # Sample end joint state i ~ column j of P_tau
        i = sample_categorical(P_tau[:, j], rng)

        # Outgoing bit:
        b_out = 0 if i in [U0, D0] else 1
        outgoing_bits.append(b_out)

        # Demon state for next interval:
        demon_state = 0 if i in [U0, U1] else 1

        # Update running Phi estimate:
        count_out_1 += (1 if b_out == 1 else 0)
        # Empirical p1_out so far:
        p1_out_emp = count_out_1 / n
        Phi_emp = p1_out_emp - p1
        running_phi.append(Phi_emp)

    outgoing_bits = np.array(outgoing_bits, dtype=int)
    incoming_bits = np.array(incoming_bits, dtype=int)

    # Final empirical stats:
    p1_out_emp = outgoing_bits.mean()
    Phi_emp = p1_out_emp - p1

    return {
        "Phi_emp": Phi_emp,
        "p1_out_emp": p1_out_emp,
        "running_phi": np.array(running_phi),
        "incoming_bits": incoming_bits,
        "outgoing_bits": outgoing_bits,
        "P_tau": P_tau,
        "R": R,
    }
def convergence_study(gamma, sigma, omega, tau, delta, N_list=(10**2,10**3, 3*10**3, 10**4, 3*10**4, 10**5), seed=0):
    """
    For several N, run Monte Carlo and compare Phi_emp to deterministic Phi.
    Returns arrays of N, Phi_emp, and absolute error |Phi_emp - Phi_det|.
    """
    det = deterministic_solution(gamma, sigma, omega, tau, delta)
    Phi_det = det["Phi"]

    Ns = []
    Phi_emp_list = []
    err_list = []
    for i, N in enumerate(N_list):
        run = monte_carlo_tape(N, gamma, sigma, omega, tau, delta, seed=seed+i)
        Phi_emp = run["Phi_emp"]
        Ns.append(N)
        Phi_emp_list.append(Phi_emp)
        err_list.append(abs(Phi_emp - Phi_det))

    return {
        "N": np.array(Ns),
        "Phi_emp": np.array(Phi_emp_list),
        "Phi_det": Phi_det,
        "abs_err": np.array(err_list),
        "det": det,
    }
def bit_bias_from_p(p0=None, p1=None, delta=None):
    """Convert between (p0,p1) and δ = p0 - p1."""
    if delta is None:
        assert p0 is not None and p1 is not None
        return float(p0 - p1)
    else:
        p0 = 0.5 * (1.0 + delta)
        p1 = 0.5 * (1.0 - delta)
        return float(p0), float(p1)
    
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

def delta_out_from_phi(delta_in, Phi):
    """Eq. (6): δ' = δ - 2Φ (since Φ = p1' - p1)."""
    return float(delta_in - 2.0 * Phi)

def DeltaS_B(delta_in, Phi):
    """Eq. (9): ΔS_B = S(δ') - S(δ)."""
    delta_out = delta_out_from_phi(delta_in, Phi)
    return S_of_delta(delta_out) - S_of_delta(delta_in)

def Q_c_to_h(Phi, DeltaE):
    """Eq. (7): Q_{c→h} = Φ * ΔE (per bit/interval)."""
    return float(Phi * DeltaE)    
