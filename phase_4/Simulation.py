from Demon import Demon, PhysParams, T_H, T_C, DELTAE, GAMMA
from Tape import Tape
import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, demon: Demon, tape: Tape, tau: float):
        self.demon = demon
        self.init_tape_arr = tape.tape_arr
        self.tau = tau
        self.N = tape.N
        self.init_tape = tape
        self.total_energy_transferred = 0.0  # Total energy transferred to hot reservoir
    def run_gillespie_window_for_joint_state(self, joint_state: str) -> tuple[str, float]:
        """Run the Gillespie algorithm for a single interaction window given the initial joint state.
        Args:
            joint_state (str): Initial joint state in the form 'dX_bY' where X is the demon state and Y is the bit state.
        Returns:
            final_state (str): Final joint state after the interaction window.
            time_elapsed (float): Total time elapsed during the interaction window.
        """
        time_elapsed = 0.0
        current_state = joint_state
        while time_elapsed < self.tau:
            rates = self.demon.get_rates_for_full_state(current_state)
            total_rate = sum(rates.values())
            if total_rate == 0:
                break  # No transitions possible
            # Time to next event
            dt = np.random.exponential(1 / total_rate)
            if time_elapsed + dt > self.tau:
                break  # Next event exceeds interaction window
            time_elapsed += dt
            # Choose which event occurs
            rand = np.random.uniform(0, total_rate)
            cumulative_rate = 0.0
            for transition, rate in rates.items():
                cumulative_rate += rate
                if rand < cumulative_rate:
                    current_state = transition.split('->')[1]
                    break
                

        return current_state, time_elapsed
    def run_full_simulation(self) -> tuple[Tape, Tape]:
        """Run the full simulation over the entire tape.
        Returns:
            final_tape (Tape): The final state of the tape after the simulation.
            initial_tape (Tape): The initial state of the tape before the simulation.
        """
        final_tape = Tape(N=self.N, p0=self.init_tape.p0, tape_arr=self.init_tape_arr.copy())
        # Run the simulation for each bit in the tape
        current_demon_state = self.demon.current_state
        demon_states_sequence = [current_demon_state]
        for i in range(self.N):
            joint_state = f'{final_tape.tape_arr[i]}_{current_demon_state}'
            final_state, _ = self.run_gillespie_window_for_joint_state(joint_state)
            current_demon_state = final_state.split('_')[1]
            final_tape.tape_arr[i] = final_state.split('_')[0]
            demon_states_sequence.append(current_demon_state)
        # get the number of highest demon state visits
        # demon_highest_state_visits = sum(1 for state in demon_states_sequence if state == f'd{self.demon.n - 1}')
        # raise SystemExit("Exiting after reporting highest state visits.")
        return final_tape, self.init_tape, demon_states_sequence
    
    def compute_statistics(self, final_tape: Tape) -> dict:
        """Compute statistics comparing the initial and final tape states.
        Args:
            final_tape (Tape): The final state of the tape after the simulation.
        Returns:
            stats (dict): Dictionary containing statistics such as incoming and outgoing distributions, biases, and entropy changes.
        """ 
        
        initial_counts = {state: np.sum(self.init_tape.tape_arr == state) for state in self.init_tape.states}
        final_counts = {state: np.sum(final_tape.tape_arr == state) for state in final_tape.states}
        initial_distribution = {state: count / self.N for state, count in initial_counts.items()}
        final_distribution = {state: count / self.N for state, count in final_counts.items()}
        initial_entropy = self.init_tape.get_entropy()
        final_entropy = final_tape.get_entropy()
        delta_s_b = final_entropy - initial_entropy
        bias_in = -initial_distribution['1'] + initial_distribution['0']
        bias_out = -final_distribution['1'] + final_distribution['0']
        phi = final_distribution['1'] - initial_distribution['1']
        total_energy_transferred = 0.0

        delta_e = self.demon.get_delta_e_for_state_n(0)
        total_energy_transferred += phi*final_tape.N * delta_e
        stats = {
            "incoming": {
                "distribution": initial_distribution,
                "p0": initial_distribution['0'],
                "p1": initial_distribution['1'],
                "entropy": initial_entropy,
                "bias": bias_in
            },
            "outgoing": {
                "distribution": final_distribution,
                "p0": final_distribution['0'],
                "p1": final_distribution['1'],
                "entropy": final_entropy,
                "DeltaS_B": delta_s_b,
                "phi": phi,
                "Q_c": phi*delta_e,
                "bias": bias_out
            },
            "demon": {
                "final_state": self.demon.current_state,
                "pu": sum(1 for i in range(self.demon.n) if self.demon.current_state == f'd{i}') / self.demon.n
            }
        }
        return stats
def plot_demon_states(demon_states_sequence: list[str]):
    """Plot the demon states histogram."""
    # Get unique states and sort them properly
    unique_states = sorted(set(demon_states_sequence), key=lambda x: int(x[1:]))
    
    # Count occurrences of each state
    state_counts = {state: demon_states_sequence.count(state) for state in unique_states}
    
    plt.figure(figsize=(10, 4))
    plt.bar(state_counts.keys(), state_counts.values(), width=0.8)
    if len(state_counts) > 40:
        # Reduce x-ticks for better readability
        step = len(state_counts) // 20
        plt.xticks(list(state_counts.keys())[::step])
    plt.xlabel('Demon States')
    plt.ylabel('Frequency')
    plt.title('Demon State Distribution Over Time')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_phi_vs_demon_n(max_n: int, tape_params: dict = None, tau: float = 1.0, 
                        min_n: int = 2, phys_params: PhysParams = None):
    """Plot phi vs demon n."""
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 0.5}
    if phys_params is None:
        phys_params = PhysParams(Th=T_H, Tc=T_C, DeltaE=DELTAE, gamma=GAMMA)
    
    assert max_n >= 2 and min_n >= 2, "max_n and min_n should be at least 2"
    
    phi_values = []
    n_values = list(range(min_n, max_n + 1))
    
    for i, n in enumerate(n_values):
        phys_params_n = PhysParams( DeltaE=phys_params.DeltaE, gamma=phys_params.gamma,demon_n=n, sigma_k=phys_params.sigma, omega_k=phys_params.omega)
        demon = Demon(n=n, phys_params=phys_params)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, demon_states_sequence = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        # plot_demon_states(demon_states_sequence=demon_states_sequence)
        phi_values.append(stats["outgoing"]["phi"])
        print(f"Progress: {(i+1)/len(n_values)*100:.1f}%", end='\r')
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, phi_values, marker='o', linewidth=2, markersize=6, color='steelblue')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='φ = 0.5')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.xlabel('Number of Demon States (n), Lowest Initial State', fontsize=12)
    plt.ylabel('Bit Flip Fraction (φ)', fontsize=12)
    plt.title(f'Bit Flip Fraction vs Number of Demon States\nτ={tau}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    return n_values, phi_values

def build_phase_diagram_tau_vs_demon_n(max_n: int, tape_params: dict = None, tau_values: list = None,min_n: int = 2):
    """Build a phase diagram of bias_out over demon n and tau."""
    if tau_values is None:
        tau_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    bias_out_matrix = []
    n_values = list(range(min_n, max_n + 1))
    for tau in tau_values:
        bias_out_row = []
        for n in n_values:
            demon = Demon(n=n)
            tape = Tape(**tape_params)
            sim = Simulation(demon=demon, tape=tape, tau=tau)
            final_tape, _, _ = sim.run_full_simulation()
            stats = sim.compute_statistics(final_tape)
            bias_out_row.append(stats["outgoing"]["bias"])
        bias_out_matrix.append(bias_out_row)
    phys_params = demon.phys_params
    plt.figure(figsize=(10, 6))
    # Anything close within 1e-5 of zero is set to zero for better color scaling
    bias_out_matrix = np.array(bias_out_matrix)
    bias_out_matrix[np.abs(bias_out_matrix) < 1e-3] = 0.0
    min_bias, max_bias = np.min(bias_out_matrix), np.max(bias_out_matrix)
   # Symmetric color scale
    min_bias += 0.01
    # Define extent to properly label axes
    min_tau, max_tau = min(tau_values), max(tau_values)
    plt.imshow(bias_out_matrix, extent=[min_n, max_n, min_tau, max_tau], aspect='auto', origin='lower', cmap='viridis', vmin=min_bias, vmax=max_bias)
    plt.colorbar(label='Bias Out')
    plt.xlabel('Number of Demon States (n)')
    plt.ylabel('Interaction Time (tau)')
    plt.title(f'Phase Diagram of Bias Out for ΔE={phys_params.DeltaE}, Th={phys_params.Th}, Tc={phys_params.Tc} , bias in= {tape_params["p0"]*2-1}')
    plt.tight_layout()
    plt.show()

def plot_phi_vs_tau(tau_values: list, tape_params: dict = None, demon_n: int = 2, phys_params: PhysParams = None):
    """Plot phi vs tau for a fixed demon n."""
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 0.5}
    if phys_params is None:
        phys_params = PhysParams(Th=T_H, Tc=T_C, DeltaE=DELTAE, gamma=GAMMA)
    
    phi_values = []
    for i, tau in enumerate(tau_values):
        demon = Demon(n=demon_n, phys_params=phys_params)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        phi_values.append(stats["outgoing"]["phi"])
        print(f"Progress: {(i+1)/len(tau_values)*100:.1f}%", end='\r')
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, phi_values, marker='o', linewidth=2, markersize=6, color='steelblue', label='Simulation')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='φ = 0.5')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.xlabel('Interaction Time (τ)', fontsize=12)
    plt.ylabel('Bit Flip Fraction (φ)', fontsize=12)
    plt.title(f'Bit Flip Fraction vs Interaction Time\nn={demon_n}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}, ', fontsize=11)
    plt.text(0.02, 0.98, 
                f'DeltaE={phys_params.DeltaE}',
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    return tau_values, phi_values

def plot_total_energy_vs_demon_n(min_n: int, max_n: int, tape_params: dict = None, tau: float = 1.0):
    """Plot total energy transferred to cold reservoir vs demon n."""
    total_energy_values = []
    bias_vals = []
    assert max_n >= 2 and min_n >= 2, "max_n should be at least 2 and min_n should be at least 2"
    total_delta_E_values = []
    n_values = list(range(min_n, max_n + 1))
    for n in n_values:
        demon = Demon(n=n)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        total_delta_E_values.append(demon.get_total_delta_e())
        total_energy_values.append(stats["outgoing"]["Q_c"])
        bias_vals.append(stats["outgoing"]["bias"])
        print(f"{n/len(n_values)*100:.2f}% completed")
    phys_params = demon.phys_params
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, total_energy_values, marker='o')

    plt.xlabel('Number of Demon States (n)')
    plt.ylabel('Total Energy Transferred to Cold Reservoir (Q_c)')
    plt.title(f'Total Energy Transferred to Cold Reservoir vs Number of Demon States (n) for ΔE={phys_params.DeltaE}, Th={phys_params.Th}, Tc={phys_params.Tc} , bias in= {tape_params["p0"]*2-1}')
    plt.grid(True)
    plt.tight_layout()
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, bias_vals, marker='o', color='cyan')
    plt.xlabel('Number of Demon States (n)')
    plt.grid(True)
    plt.plot(n_values, np.full_like(n_values, 0), linestyle='--', color='black')
    plt.ylabel('Bias Out')
    plt.xticks(n_values[::max(1, len(n_values)//10)])
    plt.title(f'Bias Out vs Number of Demon States (n) for ΔE={phys_params.DeltaE}, Th={phys_params.Th}, Tc={phys_params.Tc} , bias in= {tape_params["p0"]*2-1}')
    plt.tight_layout()
    plt.show()


def plot_output_vs_N(N_values: list, output: str = 'phi', tape_params: dict = None, 
                     demon_n: int = 2, tau: float = 1.0, phys_params: PhysParams = None):
    """Plot selected output parameter vs tape length N.
    
    Args:
        N_values (list): List of tape lengths to test
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (p0 will be used, N will be overridden)
        demon_n (int): Number of demon states
        tau (float): Interaction time
        phys_params (PhysParams): Physical parameters
    """
    if tape_params is None:
        tape_params = {"p0": 0.5}
    if phys_params is None:
        phys_params = PhysParams(Th=T_H, Tc=T_C, DeltaE=DELTAE, gamma=GAMMA)
    
    output_map = {'phi': 'phi', 'bias_out': 'bias', 'Q_c': 'Q_c', 'delta_S_b': 'DeltaS_B'}
    output_labels = {'phi': 'Bit Flip Fraction (φ)', 'bias_out': 'Bias Out (p₀ - p₁)', 
                     'Q_c': 'Energy to Cold (Q_c)', 'delta_S_b': 'Entropy Change (ΔS_B)'}
    
    y_values = []
    for i, N in enumerate(N_values):
        demon = Demon(n=demon_n, phys_params=phys_params)
        tape = Tape(N=int(N), p0=tape_params['p0'])
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        y_values.append(stats["outgoing"][output_map[output]])
        print(f"Progress: {(i+1)/len(N_values)*100:.1f}%", end='\r')
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, y_values, marker='o', linewidth=2, markersize=6, color='steelblue')
    if output in ['phi', 'bias_out', 'delta_S_b']:
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Tape Length (N)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='φ = 0.5')
    
    plt.title(f'{output_labels[output]} vs Tape Length\nn={demon_n}, τ={tau}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}, p₀={tape_params["p0"]:.2f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return N_values, y_values


def plot_output_vs_gamma(gamma_values: list, output: str = 'phi', tape_params: dict = None,
                         demon_n: int = 2, tau: float = 1.0, phys_params: PhysParams = None):
    """Plot selected output parameter vs transition rate gamma.
    
    Args:
        gamma_values (list): List of gamma values to test
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of demon states
        tau (float): Interaction time
        phys_params (PhysParams): Physical parameters (will use Th, Tc, DeltaE)
    """
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 0.5}
    if phys_params is None:
        phys_params = PhysParams(Th=T_H, Tc=T_C, DeltaE=DELTAE, gamma=GAMMA)
    
    output_map = {'phi': 'phi', 'bias_out': 'bias', 'Q_c': 'Q_c', 'delta_S_b': 'DeltaS_B'}
    output_labels = {'phi': 'Bit Flip Fraction (φ)', 'bias_out': 'Bias Out (p₀ - p₁)', 
                     'Q_c': 'Energy to Cold (Q_c)', 'delta_S_b': 'Entropy Change (ΔS_B)'}
    
    y_values = []
    for i, gamma in enumerate(gamma_values):
        temp_phys = PhysParams(Th=phys_params.Th, Tc=phys_params.Tc, 
                               DeltaE=phys_params.DeltaE, gamma=gamma)
        demon = Demon(n=demon_n, phys_params=temp_phys)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        y_values.append(stats["outgoing"][output_map[output]])
        print(f"Progress: {(i+1)/len(gamma_values)*100:.1f}%", end='\r')
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, y_values, marker='o', linewidth=2, markersize=6, color='darkgreen', label='Simulation')
    
    # Add theoretical limit for phi when gamma -> infinity
    if output == 'phi':
        # Calculate theoretical phi using the formula:
        # Φ = (δ - ε/2)[1 - e^(-(1-σω)τ)]
        # where ε = (ω - σ)/(1 - ωσ) and δ = p0 - 0.5
        sigma = phys_params.sigma
        omega = phys_params.omega
        p0 = tape_params['p0']
        
        # δ (bias in, but as a fraction from 0.5)
        delta = 2*p0 -1
        
        # ε (epsilon)
        epsilon = (omega - sigma) / (1 - omega * sigma)
        
        # Theoretical φ in the limit gamma -> infinity
        phi_theory = ((delta - epsilon) / 2) * (1 - np.exp(-(1 - sigma * omega) * tau))
        
        plt.axhline(y=phi_theory, color='red', linestyle='--', linewidth=2, 
                   label=f'Theory (γ→∞): φ = {phi_theory:.4f}')
        plt.text(0.02, 0.98, 
                f'σ = {sigma:.4f}\nω = {omega:.4f}\nε = {epsilon:.4f}\nδ = {delta:.4f}\nτ = {tau:.2f}',
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    if output in ['phi', 'bias_out', 'delta_S_b']:
        plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.xlabel('Transition Rate (γ)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    plt.title(f'{output_labels[output]} vs Transition Rate\nn={demon_n}, τ={tau}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}, σ={phys_params.sigma:.2f}, ω={phys_params.omega:.2f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    return gamma_values, y_values


def plot_output_vs_sigma(sigma_values: list, output: str = 'phi', tape_params: dict = None,
                         demon_n: int = 2, tau: float = 1.0, phys_params: PhysParams = None):
    """Plot selected output parameter vs hot reservoir parameter sigma.
    
    Args:
        sigma_values (list): List of sigma values to test (must be in (-1, 1))
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of demon states
        tau (float): Interaction time
        phys_params (PhysParams): Physical parameters (will use omega, DeltaE, gamma)
    """
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 0.5}
    if phys_params is None:
        phys_params = PhysParams(Th=T_H, Tc=T_C, DeltaE=DELTAE, gamma=GAMMA)
    
    output_map = {'phi': 'phi', 'bias_out': 'bias', 'Q_c': 'Q_c', 'delta_S_b': 'DeltaS_B'}
    output_labels = {'phi': 'Bit Flip Fraction (φ)', 'bias_out': 'Bias Out (p₀ - p₁)', 
                     'Q_c': 'Energy to Cold (Q_c)', 'delta_S_b': 'Entropy Change (ΔS_B)'}
    
    y_values = []
    for i, sigma in enumerate(sigma_values):
        temp_phys = PhysParams(sigma=sigma, omega=phys_params.omega,
                               DeltaE=phys_params.DeltaE, gamma=phys_params.gamma)
        demon = Demon(n=demon_n, phys_params=temp_phys)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        y_values.append(stats["outgoing"][output_map[output]])
        print(f"Progress: {(i+1)/len(sigma_values)*100:.1f}%", end='\r')
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, y_values, marker='o', linewidth=2, markersize=6, color='crimson')
    if output in ['phi', 'bias_out', 'delta_S_b']:
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Hot Reservoir Parameter (σ)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    plt.title(f'{output_labels[output]} vs σ\nn={demon_n}, τ={tau}, ω={phys_params.omega:.3f}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return sigma_values, y_values


def plot_output_vs_omega(omega_values: list, output: str = 'phi', tape_params: dict = None,
                         demon_n: int = 2, tau: float = 1.0, phys_params: PhysParams = None):
    """Plot selected output parameter vs cold reservoir parameter omega.
    
    Args:
        omega_values (list): List of omega values to test (must be in (-1, 1))
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of demon states
        tau (float): Interaction time
        phys_params (PhysParams): Physical parameters (will use sigma, DeltaE, gamma)
    """
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 0.5}
    if phys_params is None:
        phys_params = PhysParams(Th=T_H, Tc=T_C, DeltaE=DELTAE, gamma=GAMMA)
    
    output_map = {'phi': 'phi', 'bias_out': 'bias', 'Q_c': 'Q_c', 'delta_S_b': 'DeltaS_B'}
    output_labels = {'phi': 'Bit Flip Fraction (φ)', 'bias_out': 'Bias Out (p₀ - p₁)', 
                     'Q_c': 'Energy to Cold (Q_c)', 'delta_S_b': 'Entropy Change (ΔS_B)'}
    
    y_values = []
    for i, omega in enumerate(omega_values):
        temp_phys = PhysParams(sigma=phys_params.sigma, omega=omega,
                               DeltaE=phys_params.DeltaE, gamma=phys_params.gamma)
        demon = Demon(n=demon_n, phys_params=temp_phys)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        y_values.append(stats["outgoing"][output_map[output]])
        print(f"Progress: {(i+1)/len(omega_values)*100:.1f}%", end='\r')
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(omega_values, y_values, marker='o', linewidth=2, markersize=6, color='darkorange')
    if output in ['phi', 'bias_out', 'delta_S_b']:
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Cold Reservoir Parameter (ω)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    plt.title(f'{output_labels[output]} vs ω\nn={demon_n}, τ={tau}, σ={phys_params.sigma:.3f}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return omega_values, y_values


def plot_entropy_change_vs_phi(phi_values: list = None, p0: float = 0.5):
    """Plot entropy change ΔS_B vs phi analytically.
    
    This function plots the relationship between bit entropy change and bit flip fraction
    based on the analytical formulas:
    - S(δ) = -(1-δ)/2 * ln((1-δ)/2) - (1+δ)/2 * ln((1+δ)/2)
    - Φ = p'₁ - p₁ = (δ - δ')/2
    - ΔS_B = S(δ') - S(δ)
    
    Args:
        phi_values (list): List of phi values to plot. If None, uses linspace from -1 to 1
        p0 (float): Initial probability of bit being 0 (default: 0.5)
    """
    if phi_values is None:
        phi_values = np.linspace(-1, 1, 500)
    
    def entropy_from_bias(delta):
        """Calculate entropy S(δ) from bias δ.
        S(δ) = -(1-δ)/2 * ln((1-δ)/2) - (1+δ)/2 * ln((1+δ)/2)
        """
        # Avoid log(0) by clipping delta to valid range
        delta = np.clip(delta, -0.9999, 0.9999)
        
        term1 = (1 - delta) / 2
        term2 = (1 + delta) / 2
        
        # Handle edge cases where terms are very small
        s = 0.0
        if term1 > 1e-10:
            s -= term1 * np.log(term1)
        if term2 > 1e-10:
            s -= term2 * np.log(term2)
        
        return s
    
    # Initial bias: δ = p₀ - p₁ = p₀ - (1 - p₀) = 2p₀ - 1
    delta_initial = 2 * p0 - 1
    
    # Calculate initial entropy
    S_initial = entropy_from_bias(delta_initial)
    
    # For each phi, calculate final bias and entropy change
    delta_S_B_values = []
    valid_phi_values = []
    
    for phi in phi_values:
        # Φ = (δ - δ')/2  =>  δ' = δ - 2Φ
        delta_final = delta_initial - 2 * phi
        
        # Only include valid final bias values
        if -1 < delta_final < 1:
            S_final = entropy_from_bias(delta_final)
            delta_S_B = S_final - S_initial
            delta_S_B_values.append(delta_S_B)
            valid_phi_values.append(phi)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(valid_phi_values, delta_S_B_values, linewidth=2, color='steelblue', label='ΔS_B(Φ)')
    plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Mark the maximum entropy change point
    max_idx = np.argmax(delta_S_B_values)
    max_phi = valid_phi_values[max_idx]
    max_delta_S = delta_S_B_values[max_idx]
    plt.plot(max_phi, max_delta_S, 'r*', markersize=15, label=f'Max: Φ={max_phi:.3f}, ΔS_B={max_delta_S:.3f}')
    
    plt.xlabel('Bit Flip Fraction (Φ)', fontsize=12)
    plt.ylabel('Per Bit Entropy Change (ΔS_B)', fontsize=12)
    plt.title(f'Per Bit Entropy Change vs Bit Flip Fraction\nInitial: p₀={p0:.2f}, δ={delta_initial:.3f}, S(δ)={S_initial:.4f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    return valid_phi_values, delta_S_B_values

def plot_delta_E_vs_output(delta_E_values: list, output: str = 'phi', tape_params: dict = None,
                         demon_n: int = 2, tau: float = 1.0, phys_params_in: PhysParams = None):
    """Plot selected output parameter vs energy gap DeltaE.
    
    Args:
        delta_E_values (list): List of DeltaE values to test
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of demon states
        tau (float): Interaction time
        phys_params (PhysParams): Physical parameters (will use sigma, omega, gamma)
    """
    if tape_params is None:
        tape_params = {"N": 1000, "p0": 0.5}
    if phys_params_in is None:
        phys_params_in = PhysParams(Th=T_H, Tc=T_C, gamma=GAMMA, DeltaE=1.0)
    
    output_map = {'phi': 'phi', 'bias_out': 'bias', 'Q_c': 'Q_c', 'delta_S_b': 'DeltaS_B'}
    output_labels = {'phi': 'Bit Flip Fraction (φ)', 'bias_out': 'Bias Out (p₀ - p₁)', 
                     'Q_c': 'Energy to Cold (Q_c)', 'delta_S_b': 'Entropy Change (ΔS_B)'}
    
    y_values = []
    for i, delta_E in enumerate(delta_E_values):
        phys_params = PhysParams(Th=phys_params_in.Th, Tc=phys_params_in.Tc,
                                 DeltaE=delta_E, gamma=phys_params_in.gamma)
        demon = Demon(n=demon_n, phys_params=phys_params)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        y_values.append(stats["outgoing"][output_map['phi']])
        print(f"Progress: {(i+1)/len(delta_E_values)*100:.1f}%", end='\r')
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(delta_E_values, y_values, marker='o', linewidth=2, markersize=6, color='purple')
    if output in ['phi', 'bias_out', 'delta_S_b']:
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Energy Gap (ΔE)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    plt.title(f'{output_labels[output]} vs Energy Gap\nn={demon_n}, τ={tau}, σ={phys_params_in.sigma:.3f}, ω={phys_params_in.omega:.3f}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return delta_E_values, y_values

def plot_q_c_vs_demon_n(min_n: int, max_n: int, tape_params: dict = None, tau: float = 1.0, phys_params: PhysParams = None,plot_phi: bool = False):
    """Plot total energy transferred to cold reservoir vs demon n."""
    total_energy_values = []
    bias_vals = []
    assert max_n >= 2 and min_n >= 2, "max_n should be at least 2 and min_n should be at least 2"
    n_values = list(range(min_n, max_n + 1))
    phi_values = []
    for n in n_values:
        demon = Demon(n=n, phys_params=PhysParams( DeltaE=phys_params.DeltaE*(n-1), Th=phys_params.Th, Tc=phys_params.Tc,  gamma=phys_params.gamma))
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        total_energy_values.append(stats["outgoing"]["Q_c"])
        bias_vals.append(stats["outgoing"]["bias"])
        phi_values.append(stats["outgoing"]["phi"])
        print(f"{(n-n_values[0])/len(n_values)*100:.2f}% completed")
    plt.figure(figsize=(8, 5))
    if plot_phi:
        plt.plot(n_values, phi_values, marker='o', color='orange', label='Bit Flip Fraction (φ)')
        plt.legend(loc='lower right')
        plt.plot(n_values, .5*np.ones_like(n_values), linestyle='--', color='red', alpha=0.7)
        plt.ylabel('Bit Flip Fraction (φ)')
        plt.twinx()

    plt.plot(n_values, total_energy_values, marker='o', color='blue', label='Energy to Cold Reservoir per Bit (Q_c)')
    plt.legend()
    plt.xlabel('Number of Demon States (n)')
    plt.ylabel('Energy Transferred to Cold Reservoir, Per Incoming Bit (Q_c)')
    plt.title(f'Energy Transferred to Cold Reservoir, Per Incoming Bit  vs Number of Demon States (n) for ΔE={phys_params.DeltaE}, Th={phys_params.Th:.2f}, Tc={phys_params.Tc:.2f} , bias in= {tape_params["p0"]*2-1}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example 1: Analytical entropy change vs phi
    
    # Example 2: Phi vs tau
    # plot_phi_vs_tau(
    #     tau_values=np.linspace(0.1, 10.0, 100).tolist(),
    #     tape_params={"N": 5000, "p0": 1.0},
    #     phys_params=PhysParams(sigma=0.3, omega=0.46, DeltaE=.00025, gamma=1.0),
    #     demon_n=5
    # )
    # plot_phi_vs_tau(
    #     tau_values=np.linspace(0.1, 20.0, 10).tolist(),
    #     tape_params={"N": 100000, "p0": 1.0},
    #     phys_params=PhysParams(sigma=0.2, omega=0.7, DeltaE=.05, gamma=1.0),
    #     demon_n=2
    # )
    plot_q_c_vs_demon_n(
        min_n=2,
        max_n=90,
        tape_params={"N": 5000, "p0": 1.0},
        tau=40.0,
        phys_params=PhysParams(sigma=0.2, omega=0.7, DeltaE=.1, gamma=1.0),
        plot_phi=True
    )

    # plot_output_vs_N(
    #     N_values=np.linspace(100, 10000, 50).tolist(),
    #     output='phi',
    #     tape_params={"p0": 1.0},
    #     demon_n=20,
    #     tau=10.0,
    #     phys_params=PhysParams(sigma=0.2, omega=0.7, DeltaE=1, gamma=1.0)
    # )
 
    # Example 3: Phi vs demon n
    # plot_phi_vs_demon_n(
    #     max_n=100,
    #     tape_params={"N": 10000, "p0": 1.0},
    #     phys_params=PhysParams(sigma=0.2, omega=0.7, DeltaE=1.0, gamma=1.0),
    #     tau=10.0
    # )