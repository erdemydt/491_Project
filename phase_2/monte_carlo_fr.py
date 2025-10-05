import numpy as np
import scipy.linalg as la
from dataclasses import dataclass
from datetime import datetime
import random
import matplotlib.pyplot as plt
# Physical parameters for the simulation
k_B = 1.0  # Boltzmann constant in chosen units

@dataclass
class PhysParams:
    Th: float        # Hot bath temperature
    Tc: float        # Cold bath temperature
    gamma: float  # Coupling rate to hot bath
    bias: float      # Tape bias 
    DeltaE: float = 1.0   # Energy scale of the demon
    kappa_cold: float = 1.0  # Coupling rate to cold bath
    tau : float = 1.0        # Interaction time per bit 
    def temps_to_sigma_omega(self, Th: float, Tc: float, DeltaE: float, k_B: float = 1.0) -> tuple:
        """Convert temperatures to dimensionless biases sigma and omega."""
        
        sigma = np.tanh(DeltaE / (2 * k_B * Th))
        omega = np.tanh(DeltaE / (2 * k_B * Tc))
        if (sigma <0 or omega <0):
            omegaNeg = omega <0
            sigmaNeg = sigma <0
            print("Warning: negative sigma & omega detected!")
            if omegaNeg:
                print(f"  omega = {omega:.4f} < 0 (Tc={Tc}, DeltaE={DeltaE})")
            if sigmaNeg:
                print(f"  sigma = {sigma:.4f} < 0 (Th={Th}, DeltaE={DeltaE})")
        epsilon = (omega - sigma) / (1 - sigma * omega)
        return sigma, omega, epsilon
    def probs_from_bias(self, bias: float) -> tuple:
        """Convert tape bias to probabilities p0 and p1."""
        p1 = (1 - bias) / 2
        p0 = (1 + bias) / 2
        return p0, p1
    def __post_init__(self):
        self.sigma, self.omega, self.epsilon = self.temps_to_sigma_omega(
            self.Th, self.Tc, self.DeltaE, k_B
        )
        self.p0_in, self.p1_in = self.probs_from_bias(self.bias)

@dataclass
class MonteCarloSimParams:
    N: int               # Number of bits to simulate
    phys: PhysParams     # Physical parameters
    p0_in: float        # Probability of incoming bit being 0
    demon_init: str     # Initial state of the demon ("u" or "d")
    seed: int = datetime.now().timestamp()  # Random seed
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.p0_in = self.phys.p0_in
        self.p1_in = self.phys.p1_in
        
@dataclass
class MonteCarloStats:
    incoming: dict      # Statistics of incoming bits
    outgoing: dict      # Statistics of outgoing bits
    demon: dict         # Statistics of demon states
    running_phi: list   # Running empirical phi values
    Phi_emp: float      # Final empirical phi value
    def __post_init__(self):
        self.incoming = {"0": 0, "1": 0}
        self.outgoing = {"0": 0, "1": 0}
        self.demon = {"u": 0, "d": 0}
        self.running_phi = []
        self.Phi_emp = 0.0
# Allowed transitions are: 0d <-> 1u (cold bath), 0u <-> 0d (hot bath), 1u <-> 1d (hot bath)    
def get_rates(phys: PhysParams) -> dict:
    """Calculate transition rates based on physical parameters. Only Allowed Transitions.
    Args:
        phys (PhysParams): Physical parameters of the system.
    Returns:
        dict: Dictionary of transition rates.
    """
    rates = {
        "0d_to_1u": phys.kappa_cold * (1 - phys.omega),  # Cold bath induced transition
        "1u_to_0d": phys.kappa_cold * (1 + phys.omega),  # Cold bath induced transition
        "0u_to_0d": phys.gamma * (1 + phys.sigma) ,  # Hot bath induced transition
        "0d_to_0u": phys.gamma * (1 - phys.sigma) ,  # Hot bath induced transition
        "1u_to_1d": phys.gamma * (1 + phys.sigma) ,  # Hot bath induced transition
        "1d_to_1u": phys.gamma * (1 - phys.sigma),  # Hot bath induced transition

    }
    return rates

def get_prob_from_outgoing_rates(rates: dict, dt: float) -> dict:
    """Convert transition rates to probabilities over a small time step dt."""
    rtot = sum(rates.values())
    if rtot <= 0:
        return {"stay": 1.0}
    stay = np.exp(-rtot * dt)
    dist ={"stay": stay}
    comp = 1 - stay
    for key, rate in rates.items():
        dist[key] = dist.get(key, 0.0) + (rate / rtot) * comp
        
    # Normalize to ensure sum to 1
    total_prob = sum(dist.values())
    for key in dist:
        dist[key] /= total_prob if total_prob > 0 else 1.0 / len(dist)
    return dist
def get_probs_for_joint_state(joint_state: str, rates: dict, dt: float) -> dict:
    """Get transition probabilities for a given joint state of demon and bit."""
    if joint_state == "0u":
        rates = {
            "0u_to_0d": rates["0u_to_0d"],
        }
        return get_prob_from_outgoing_rates(rates, dt)
    elif joint_state == "0d":
        rates = {
            "0d_to_0u": rates["0d_to_0u"],
            "0d_to_1u": rates["0d_to_1u"],
        }
        return get_prob_from_outgoing_rates(rates, dt)
    elif joint_state == "1u":
        rates = {
            "1u_to_1d": rates["1u_to_1d"],
            "1u_to_0d": rates["1u_to_0d"],
        }
        return get_prob_from_outgoing_rates(rates, dt)
    elif joint_state == "1d":
        rates = {
            "1d_to_1u": rates["1d_to_1u"],
        }
        return get_prob_from_outgoing_rates(rates, dt)



def plot_probs_over_time(probs: dict, joint_state: str, dt: float, total_time: float):
    """Plot transition probabilities over time for a given joint state."""
    times = np.arange(0, total_time, dt)
    prob_matrix = {key: [] for key in probs.keys()}
    
    for t in times:
        current_probs = get_probs_for_joint_state(joint_state, rates, t)
        for key in probs.keys():
            prob_matrix[key].append(current_probs.get(key, 0.0))
    
    plt.figure(figsize=(10, 6))
    for key, values in prob_matrix.items():
        plt.plot(times, values, label=key)
    
    plt.title(f"Transition Probabilities from State {joint_state}")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()


def generate_tape(p0_in: float, N: int) -> np.array:
    """Generate a whole tape based on the incoming probability p0_in."""
    tape = []
    for _i in range(N):
        tape.append("0" if random.random() < p0_in else "1")
    return np.array(tape)
def evolve_one_step(joint_state: str, rates: dict, dt: float) -> str:
    """Evolve the joint state of demon and bit over a small time step dt based on transition rates."""
    probs = get_probs_for_joint_state(joint_state, rates, dt)
    rand_val = random.random()
    cumulative_prob = 0.0
    
    for transition, prob in probs.items():
        cumulative_prob += prob
        if rand_val < cumulative_prob:
            if transition == "stay":
                return joint_state
            else:
                return transition.split("_to_")[1]
    return joint_state  # Fallback to staying in the same state
def evolve_one_joint_state(joint_state: str, rates: dict, dt: float, tau: float) -> str:
    """Evolve the joint state of demon and bit over time dt in total time tau based on transition rates."""
    num_steps = int(tau / dt)
    total_time = 0.0
    num_of_state_changes = 0

    for _ in range(num_steps):
        out_state = evolve_one_step(joint_state, rates, dt)
        total_time += dt
        if out_state != joint_state:
            print(f"State changed to {out_state} at time {total_time:.4f} from {joint_state}.")
            joint_state = out_state
            num_of_state_changes += 1
            total_time = 0.0
        # if total_time >= tau:
            # print("Warning: total_time exceeded tau without state change.")
            # raise ValueError("Total time exceeded tau without state change.")
            
    print(f"Number of state changes during evolution: {num_of_state_changes}")
    raise SystemExit("Exiting after state change count.")
    return joint_state

def run_sim(phys : PhysParams, N: int, p0_in: float, demon_init: str, dt: float, seed: int = None) -> MonteCarloStats:
    """Run the Monte Carlo simulation for N bits."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    rates = get_rates(phys)
    tape = generate_tape(p0_in, N)
    stats = MonteCarloStats(incoming={}, outgoing={}, demon={}, running_phi=[], Phi_emp=0.0)
    
    demon_state = demon_init
    phi_count = 0
    
    for i in range(N):
        bit_in = tape[i]
        stats.incoming[bit_in] += 1
        
        joint_state = bit_in + demon_state
        joint_state = evolve_one_joint_state(joint_state, rates, dt, phys.tau)
        
        bit_out = joint_state[0]
        demon_state = joint_state[1]
        
        stats.outgoing[bit_out] += 1
        stats.demon[demon_state] += 1

        phi_count = stats.outgoing["1"] - stats.incoming["1"]

        stats.running_phi.append(phi_count / (i + 1))
    
    stats.Phi_emp = phi_count / N
    return stats
# Example usage
if __name__ == "__main__":
    # Define physical parameters
    Th, Tc = 2.0, 1.6       # temperatures (in k_B=1 units)
    DeltaE = 1.0            # energy scale
    gamma = 3.0             # hot bath coupling rate
    kappa_cold = 1.0        # cold bath coupling rate
    bias = 0.9              # tape bias
    tau = 1.0               # interaction time per bit
    
    phys = PhysParams(Th=Th, Tc=Tc, DeltaE=DeltaE, gamma=gamma, bias=bias, kappa_cold=kappa_cold, tau=tau)
    
    print(f"Physical parameters:")
    print(f"  Th={Th}, Tc={Tc}, ΔE={DeltaE}")
    print(f"  σ (hot bias) = {phys.sigma:.4f}")
    print(f"  ω (cold bias) = {phys.omega:.4f}")
    print(f"  γ (hot rate) = {phys.gamma}")
    print(f"  κ (cold rate) = {phys.kappa_cold}")
    print(f"  ε (temp bias) = {phys.epsilon:.4f}")
    

    # Time step for simulation
    dt = 0.0001
    # Run the Monte Carlo simulation
    stats_sim = run_sim(phys=phys, N=1000, p0_in=phys.p0_in, demon_init="u", dt=dt, seed=7)
    print("Empirical incoming:", stats_sim.incoming)
    print("Empirical outgoing:", stats_sim.outgoing)
    print("Demon occupancy:", stats_sim.demon)
    print(f"Empirical Phi: {stats_sim.Phi_emp:.4f}")
    print(f"Final running Phi: {stats_sim.running_phi[-1]:.4f}")


    
