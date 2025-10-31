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
                "Q_c": total_energy_transferred,
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

def plot_bias_out_vs_demon_n(max_n: int, tape_params: dict = None, tau: float = 1.0):
    """Plot bias_out vs demon n."""
    bias_out_values = []
    assert max_n >= 2, "max_n should be at least 2"
    n_values = list(range(2, max_n + 1))
    for n in n_values:
        demon = Demon(n=n)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        bias_out_values.append(stats["outgoing"]["bias"])
    phys_params = demon.phys_params
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, bias_out_values, marker='o')
    plt.xlabel('Number of Demon States (n)')
    plt.ylabel('Bias Out ')
    plt.title(f'Bias Out vs Number of Demon States (n) for ΔE={phys_params.DeltaE}, Th={phys_params.Th}, Tc={phys_params.Tc} , bias in= {tape_params["p0"]*2-1}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

def plot_bias_vs_tau(tau_values: list, tape_params: dict = None, demon_n: int = 2):
    """Plot bias_out vs tau for a fixed demon n."""
    bias_out_values = []
    for tau in tau_values:
        demon = Demon(n=demon_n)
        tape = Tape(**tape_params)
        sim = Simulation(demon=demon, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        bias_out_values.append(stats["outgoing"]["bias"])
    phys_params = demon.phys_params
    plt.figure(figsize=(8, 5))
    plt.plot(tau_values, bias_out_values, marker='o', label='Bias Out')
    plt.xlabel('Interaction Time (tau)')
    plt.ylabel('Bias Out ')
    plt.plot(tau_values, [0]*len(tau_values), 'r--', label='Bias Out = 0')
    plt.title(f'Bias Out vs Interaction Time (tau) for n={demon_n}, ΔE={phys_params.DeltaE}, Th={phys_params.Th}, Tc={phys_params.Tc} , bias in= {tape_params["p0"]*2-1}, N={tape_params["N"]}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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


if __name__ == "__main__":
    plot_total_energy_vs_demon_n(min_n=2, max_n=20, tape_params={"N": 50000, "p0": 1.0}, tau=20.0)