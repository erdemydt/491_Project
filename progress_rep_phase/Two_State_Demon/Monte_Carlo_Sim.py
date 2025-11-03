import numpy as np
import matplotlib.pyplot as plt
import Constants as const
from dataclasses import dataclass
from numba import njit
import Utils as utils
@dataclass
class PhysParams:
    T_H: float  # Hot reservoir temperature
    T_C: float  # Cold reservoir temperature
    DeltaE: float  # Energy difference
    sigma: float  
    omega: float
    tau: float = 10.0  # Time duration for each state
    k_B: float = const.k_B  # Boltzmann constant
STATE_IDX = {0: '0U', 2: '0D', 1: '1U', 3: '1D'}

def build_R_batch(sigmas, omegas, gammas, dtype=np.float64):
    """Vectorized batch: sigmas, omegas, gammas broadcast to shape (N,)."""
    sigmas = np.asarray(sigmas, dtype=dtype)
    omegas = np.asarray(omegas, dtype=dtype)
    gammas = np.asarray(gammas, dtype=dtype)
    a = gammas * (1.0 - sigmas)
    b = gammas * (1.0 + sigmas)
    c = 1.0 - omegas
    d = 1.0 + omegas

    N = np.broadcast(a, b, c, d).shape[0]

    R = np.zeros((N, 4, 4), dtype=dtype)

    # Intrinsic
    R[:, 0, 1] = a
    R[:, 1, 0] = b
    R[:, 2, 3] = a
    R[:, 3, 2] = b
    # Cooperative
    R[:, 2, 1] = c
    R[:, 1, 2] = d

    # Diagonals per column
    R[:, np.arange(4), np.arange(4)] = -R.sum(axis=1)
    return R

@njit
def gillespie_step(state, R, tau):
    """Perform a single Gillespie step."""
    rates = R[:, state]
    rtot = -rates[state]
 

    
    if rtot <= 0:
        return state
    outgoing_rates = rates.copy()
    outgoing_rates[state] = 0.0
    
    dt = -np.log(np.random.rand()) / rtot
    if dt > tau:
        return state
    r = np.random.rand() * rtot
    cumulative = 0.0
    for next_state in range(4):
        cumulative += outgoing_rates[next_state]
        if r < cumulative:
            return next_state
    return state  # Should not reach here
@njit
def build_tape(delta, N):
    """Build a bit tape with bias delta."""
    tape = np.zeros(N, dtype=np.int32)
    for i in range(N):
        tape[i] = 1 if np.random.rand() >delta else 0
    return tape

def monte_carlo_sim(params: PhysParams, N_cycles: int, delta: float, initial_demon_state: str = "D") -> dict:
    """Run Monte Carlo simulation."""
    # Prepare rate matrix   
    sigma = params.sigma
    omega = params.omega
    gamma = const.gamma
    R = build_R_batch([sigma], [omega], [gamma])[0]
    # Initialize tape
    tape = build_tape(delta, N_cycles)   
    out_tape = np.zeros(N_cycles, dtype=np.int32)
    # Initial state
    initial_state = tape[0]*2 + (1 if initial_demon_state == "D" else 0)
    state = initial_state
    for i in range(N_cycles):
        # Evolve state
        state = gillespie_step(state, R, params.tau)
        # Record output tape
        out_tape[i] = 1 if state // 2 == 1 else 0
        # Update state for next cycle
        if i < N_cycles - 1:
            next_bit = tape[i + 1]
            demon_part = state % 2
            state = next_bit * 2 + demon_part
    return {"states": state, "tape": tape, "out_tape": out_tape}

def get_bias(tape):
    num_ones = np.sum(tape)
    num_zeros = len(tape) - num_ones
    return (num_zeros - num_ones) / len(tape)

def build_params_grid(T_C,N_points=50):
    omega = utils.get_sigma_omega_from_T_H_C(const.T_H, T_C, k_B=const.k_B, deltaE=const.DeltaE)[1]
    epsilon_vals = np.linspace(-0.99, 0.99, N_points)
    sigma_vals = np.array([utils.get_sigma_from_epsilon_omega(epsilon, omega) for epsilon in epsilon_vals])
    bias_vals = np.linspace(-0.99, 0.99, N_points)
    return sigma_vals, np.array([omega]*N_points), epsilon_vals, bias_vals

def get_phi_from_bias(bias_in, bias_out):
    """Calculate phi from input and output biases."""
    return 0.5 * (bias_out - bias_in)
def build_phase_diagram(T_C, N_points=50):
    sigma_vals, omega_vals, epsilon_vals, bias_vals = build_params_grid(T_C, N_points)
    phi_vals = []

    for bias, i in zip(bias_vals, range(len(bias_vals))):
        phi_vals.append([])
        in_biases = []
        biases = []
        for sigma, omega, epsilon in zip(sigma_vals, omega_vals, epsilon_vals):
            params = PhysParams(
                T_H=const.T_H,
                T_C=T_C,
                DeltaE=const.DeltaE,
                sigma=sigma,
                omega=omega
            )
            sim_result = monte_carlo_sim(params, N_cycles=2000, delta=bias)
            bias_out = get_bias(sim_result["out_tape"])
            phi = get_phi_from_bias(bias, bias_out)
            in_biases.append(bias)
            biases.append(epsilon)
            phi_vals[i].append(phi)
        print(f"Completed percentage: {(i+1)/len(bias_vals)*100:.2f}%")
  
    return epsilon_vals, in_biases, np.array(biases), np.array(phi_vals)

def plot_phase_diagram(epsilon_vals, in_biases, phi_vals):
    plt.figure(figsize=(8, 6))
    plt.contourf( phi_vals, levels=50, cmap='RdBu_r')
    plt.colorbar(label='Phi')
    plt.xlabel('Epsilon')
    plt.ylabel('Input Bias')
    plt.title('Phase Diagram')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    T_C = const.T_C
    N_points = 500
    epsilon_vals, in_biases, biases, phi_vals = build_phase_diagram(T_C, N_points)
    
    # rotate phi_vals -90 degrees for correct orientation
    phi_vals = np.rot90(phi_vals)
    phi_vals = np.rot90(phi_vals)
    phi_vals = np.rot90(phi_vals)

    plot_phase_diagram(epsilon_vals, in_biases, phi_vals)
