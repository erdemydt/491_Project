from Demon import Demon, PhysParams, T_H, T_C, DELTAE, GAMMA
from Tape import Tape
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class StackedDemonSimulation:
    """Simulation with K demons stacked on top of each other.
    
    Each bit interacts sequentially with K demons before becoming the output bit.
    
    Attributes:
        demons (List[Demon]): List of K demons
        K (int): Number of stacked demons
        tau (float): Interaction time per demon
        N (int): Number of bits on tape
        init_tape (Tape): Initial tape
    """
    
    def __init__(self, demons: List[Demon], tape: Tape, tau: float):
        """Initialize stacked demon simulation.
        
        Args:
            demons (List[Demon]): List of K demons (all should have same n)
            tape (Tape): Initial tape
            tau (float): Interaction time per demon
        """
        self.demons = demons
        self.K = len(demons)
        self.tau = tau
        self.init_tape = tape
        self.init_tape_arr = tape.tape_arr.copy()
        self.N = tape.N
        
        # Verify all demons have the same number of states
        demon_n_values = [demon.n for demon in demons]
        if len(set(demon_n_values)) > 1:
            raise ValueError(f"All demons must have the same number of states. Got: {demon_n_values}")
        
        self.demon_n = demons[0].n
    
    def run_gillespie_window_for_joint_state(self, demon: Demon, joint_state: str) -> Tuple[str, float]:
        """Run the Gillespie algorithm for a single interaction window with one demon.
        
        Args:
            demon (Demon): The demon to interact with
            joint_state (str): Initial joint state in the form 'bitValue_dX'
        
        Returns:
            final_state (str): Final joint state after the interaction window
            time_elapsed (float): Total time elapsed during the interaction window
        """
        time_elapsed = 0.0
        current_state = joint_state
        
        while time_elapsed < self.tau:
            rates = demon.get_rates_for_full_state(current_state)
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
    
    def process_bit_through_demons(self, bit_value: str, demon_states: List[str]) -> Tuple[str, List[str]]:
        """Process a single bit through all K demons sequentially.
        
        Args:
            bit_value (str): Initial bit value ('0' or '1')
            demon_states (List[str]): Current states of all K demons
        
        Returns:
            final_bit (str): Final bit value after passing through all demons
            final_demon_states (List[str]): Final states of all K demons
        """
        current_bit = bit_value
        new_demon_states = []
        
        for k in range(self.K):
            # Form joint state for this demon
            joint_state = f'{current_bit}_{demon_states[k]}'
            
            # Run interaction with this demon
            final_state, _ = self.run_gillespie_window_for_joint_state(self.demons[k], joint_state)
            
            # Extract new bit and demon state
            current_bit = final_state.split('_')[0]
            new_demon_state = final_state.split('_')[1]
            new_demon_states.append(new_demon_state)
        
        return current_bit, new_demon_states
    
    def run_full_simulation(self) -> Tuple[Tape, Tape, List[List[str]]]:
        """Run the full simulation over the entire tape with K stacked demons.
        
        Returns:
            final_tape (Tape): The final state of the tape after the simulation
            initial_tape (Tape): The initial state of the tape before the simulation
            demon_states_history (List[List[str]]): History of demon states for each demon
        """
        final_tape = Tape(N=self.N, p0=self.init_tape.p0, tape_arr=self.init_tape_arr.copy())
        
        # Initialize demon states (all start at their initial states)
        current_demon_states = [demon.current_state for demon in self.demons]
        
        # Track demon state history for each demon
        demon_states_history = [[] for _ in range(self.K)]
        for k in range(self.K):
            demon_states_history[k].append(current_demon_states[k])
        
        # Process each bit through all K demons
        for i in range(self.N):
            initial_bit = final_tape.tape_arr[i]
            
            # Process bit through all demons
            final_bit, current_demon_states = self.process_bit_through_demons(
                initial_bit, current_demon_states
            )
            
            # Update tape with final bit
            final_tape.tape_arr[i] = final_bit
            
            # Record demon states
            for k in range(self.K):
                demon_states_history[k].append(current_demon_states[k])
        
        return final_tape, self.init_tape, demon_states_history
    
    def compute_statistics(self, final_tape: Tape) -> Dict:
        """Compute statistics comparing the initial and final tape states.
        
        Args:
            final_tape (Tape): The final state of the tape after the simulation
        
        Returns:
            stats (dict): Dictionary containing statistics
        """
        initial_counts = {state: np.sum(self.init_tape.tape_arr == state) 
                         for state in self.init_tape.states}
        final_counts = {state: np.sum(final_tape.tape_arr == state) 
                       for state in final_tape.states}
        
        initial_distribution = {state: count / self.N for state, count in initial_counts.items()}
        final_distribution = {state: count / self.N for state, count in final_counts.items()}
        
        initial_entropy = self.init_tape.get_entropy()
        final_entropy = final_tape.get_entropy()
        
        delta_s_b = final_entropy - initial_entropy
        bias_in = initial_distribution['0'] - initial_distribution['1']
        bias_out = final_distribution['0'] - final_distribution['1']
        phi = final_distribution['1'] - initial_distribution['1']
        
        # Energy transferred (using first demon's energy scale)
        delta_e = self.demons[0].get_delta_e_for_state_n(0)
        print(f"Delta E for state n=0: {delta_e}")

        total_energy_transferred = phi * delta_e
        
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
                "bias": bias_out
            },
            "phi": phi,
            "bias": bias_out - bias_in,
            "Q_c": total_energy_transferred,
            "N": self.N,
            "K": self.K
        }
        
        return stats


def plot_output_vs_K(K_values: List[int], output: str = 'phi', tape_params: Dict = None,
                     demon_n: int = 2, tau: float = 1.0, phys_params: PhysParams = None):
    """Plot selected output parameter vs number of stacked demons K.
    
    Args:
        K_values (List[int]): List of K values (number of stacked demons) to test
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of states per demon
        tau (float): Interaction time per demon
        phys_params (PhysParams): Physical parameters
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8, DeltaE=DELTAE, gamma=GAMMA)
    
    output_map = {
        'phi': 'phi',
        'bias_out': 'bias',
        'Q_c': 'Q_c',
        'delta_S_b': 'DeltaS_B'
    }
    
    output_labels = {
        'phi': 'Bit Flip Fraction (φ)',
        'bias_out': 'Bias Out (p₀ - p₁)',
        'Q_c': 'Energy to Cold (Q_c)',
        'delta_S_b': 'Entropy Change (ΔS_B)'
    }
    
    y_values = []
    
    for i, K in enumerate(K_values):
        print(f"Progress: {i+1}/{len(K_values)} - K={K}", end='\r')
        
        # Create K demons
        demons = [Demon(n=demon_n, phys_params=phys_params, init_state='d0') 
                 for _ in range(K)]
        
        # Create tape
        tape = Tape(N=tape_params["N"], p0=tape_params["p0"])
        
        # Run simulation
        sim = StackedDemonSimulation(demons=demons, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        
        # Compute statistics
        stats = sim.compute_statistics(final_tape)
        
        # Extract the desired output
        if output == 'phi':
            y_values.append(stats['phi'])
        elif output == 'bias_out':
            y_values.append(stats['outgoing']['bias'])
        elif output == 'Q_c':
            y_values.append(stats['Q_c'])
        elif output == 'delta_S_b':
            y_values.append(stats['outgoing']['DeltaS_B'])
    
    print()  # New line after progress
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, y_values, marker='o', linewidth=2, markersize=6, color='steelblue')
    
    # Add reference lines
    if output in ['phi', 'bias_out', 'delta_S_b']:
        plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    if output == 'phi':
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='φ = 0.5')
    
    plt.xlabel('Number of Stacked Demons (K)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    
    # Title with parameters
    title = f'{output_labels[output]} vs Number of Stacked Demons\n'
    title += f'n={demon_n}, τ={tau}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}, '
    title += f'N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}'
    
    # Add delta_e_mode and preserve_mode info
    title += f'\nΔE mode: {phys_params.delta_e_mode}, Preserve: {phys_params.preserve_mode}'
    
    plt.title(title, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if output == 'phi':
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return y_values

def plot_output_vs_tau(tau_vals: List[float], output: str = 'phi', tape_params: Dict = None,
                     demon_n: int = 2, K: int = 5, phys_params: PhysParams = None):
    """Plot selected output parameter vs interaction time tau.
    
    Args:
        tau_vals (List[float]): List of tau values (interaction times) to test
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of states per demon
        K (int): Number of stacked demons
        phys_params (PhysParams): Physical parameters
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8, DeltaE=DELTAE, gamma=GAMMA)
    
    output_map = {
        'phi': 'phi',
        'bias_out': 'bias',
        'Q_c': 'Q_c',
        'delta_S_b': 'DeltaS_B'
    }
    
    output_labels = {
        'phi': 'Bit Flip Fraction (φ)',
        'bias_out': 'Bias Out (p₀ - p₁)',
        'Q_c': 'Energy to Cold (Q_c)',
        'delta_S_b': 'Entropy Change (ΔS_B)'
    }
    
    y_values = []
    
    for i, tau in enumerate(tau_vals):
        print(f"Progress: {i+1}/{len(tau_vals)} - tau={tau}", end='\r')
        
        # Create K demons
        demons = [Demon(n=demon_n, phys_params=phys_params, init_state='d0') 
                 for _ in range(K)]
        
        # Create tape
        tape = Tape(N=tape_params["N"], p0=tape_params["p0"])
        
        # Run simulation
        sim = StackedDemonSimulation(demons=demons, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        
        # Compute statistics
        stats = sim.compute_statistics(final_tape)
        
        # Extract the desired output
        if output == 'phi':
            y_values.append(stats['phi'])
        elif output == 'bias_out':
            y_values.append(stats['outgoing']['bias'])
        elif output == 'Q_c':
            y_values.append(stats['Q_c'])
        elif output == 'delta_S_b':
            y_values.append(stats['outgoing']['DeltaS_B'])
    
    print()  # New line after progress
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(tau_vals, y_values, marker='o', linewidth=2, markersize=6, color='seagreen')
    
    # Add reference lines
    if output in ['phi', 'bias_out', 'delta_S_b']:
        plt.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    if output == 'phi':
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='φ = 0.5')
    
    plt.xlabel('Interaction Time (τ)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    
    # Title with parameters
    title = f'{output_labels[output]} vs Interaction Time (τ)\n'
    title += f'n={demon_n}, K={K}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}, '
    title += f'N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}'
    
    # Add delta_e_mode and preserve_mode info
    title += f'\nΔE mode: {phys_params.delta_e_mode}, Preserve: {phys_params.preserve_mode}'
    
    plt.title(title, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if output == 'phi':
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return y_values
    
    
    
def plot_q_c_vs_demon_n(min_n: int = 2, max_n: int = 50, tape_params: Dict = None,
                        tau: float = 1.0, phys_params: PhysParams = None, plot_phi: bool = False):
    """Plot total energy transferred to cold reservoir Q_c vs number of demon states n.
    
    Args:
        min_n (int): Minimum number of demon states
        max_n (int): Maximum number of demon states
        tape_params (dict): Tape parameters (N, p0)
        tau (float): Interaction time per demon
        phys_params (PhysParams): Physical parameters
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(sigma=0.3, omega=0.8, DeltaE=DELTAE, gamma=GAMMA)
    
    n_values = list(range(min_n, max_n + 1))
    total_energy_values = []
    phi_values = []
    
    for i, n in enumerate(n_values):
        print(f"Progress: {i+1}/{len(n_values)} - n={n}", end='\r')
        
        demon = Demon(n=n, phys_params=PhysParams( DeltaE=phys_params.DeltaE, sigma=phys_params.sigma, omega=phys_params.omega, gamma=phys_params.gamma,preserve_mode='temperatures'))
        
        tape = Tape(N=tape_params["N"], p0=tape_params["p0"])
        
        sim = StackedDemonSimulation(demons=[demon], tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        
        stats = sim.compute_statistics(final_tape)
        
        total_energy_values.append(stats['Q_c']*stats['phi'])
        if plot_phi:
            phi_values.append(stats['phi'])
    
    print()  # New line after progress
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    if plot_phi:
        plt.plot(n_values, phi_values, marker='o', linewidth=2, markersize=6, color='orange', label='φ (Bit Flip Fraction)')
        plt.ylabel('Bit Flip Fraction (φ)', fontsize=12)
        plt.twinx()
    plt.plot(n_values, total_energy_values, marker='o', linewidth=2, markersize=6, color='seagreen')
    plt.xlabel('Number of Demon States (n)', fontsize=12)
    plt.ylabel('Total Energy Transferred to Cold Reservoir (Q_c)', fontsize=12)
    plt.title('Total Energy Transferred to Cold Reservoir (Q_c) vs Number of Demon States (n)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # Example 1: Plot phi vs K with per-state DeltaE mode
    print("Example 1: Per-state DeltaE mode, preserving sigma/omega")
    # plot_output_vs_K(
    #     K_values=list(range(1, 31)),
    #     output='phi',
    #     tape_params={"N": 10000, "p0": 1.0},
    #     demon_n=5,
    #     tau=1.0,
    #     phys_params=PhysParams(
    #         sigma=0.4, 
    #         omega=0.7, 
    #         DeltaE=1.0, 
    #         gamma=1.0,
    #         delta_e_mode='total',
    #         demon_n=5,
    #         preserve_mode='sigma_omega'
    #     )
    # )
    plot_q_c_vs_demon_n(
        min_n=15,
        max_n=30,
        tape_params={"N": 3000, "p0": 1.0},
        tau=50.0,
        phys_params=PhysParams(sigma=0.2, omega=0.7, DeltaE=1, gamma=1.0),
        plot_phi=True
    )
    

