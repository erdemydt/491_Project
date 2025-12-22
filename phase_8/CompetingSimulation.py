import numpy as np
import matplotlib.pyplot as plt
from CompetingDemon import CompetingDemon, PhysParams
from Tape import Tape
from typing import List, Tuple, Dict


class CompetingDemonSimulation:
    """Simulation where K identical demons compete to interact with each bit.
    
    For each bit, all K demons draw a dt from exponential distributions based
    on their current states. The demon with the smallest dt wins and interacts
    with the bit. This allows demons in different states to have different
    probabilities of interaction.
    
    Attributes:
        demons (List[CompetingDemon]): List of K identical demons
        K (int): Number of competing demons
        tau (float): Total interaction time per bit
        N (int): Number of bits on tape
        init_tape (Tape): Initial tape
    """
    
    def __init__(self, demons: List[CompetingDemon], tape: Tape, tau: float):
        """Initialize competing demon simulation.
        
        Args:
            demons (List[CompetingDemon]): List of K demons (all should have same n and params)
            tape (Tape): Initial tape
            tau (float): Total interaction time per bit
        """
        self.demons = demons
        self.K = len(demons)
        self.tau = tau
        self.init_tape = tape
        self.init_tape_arr = tape.tape_arr.copy()
        self.N = tape.N
        
        # Verify all demons have the same configuration
        demon_n_values = [demon.n for demon in demons]
        if len(set(demon_n_values)) > 1:
            raise ValueError(f"All demons must have the same number of states. Got: {demon_n_values}")
        
        self.demon_n = demons[0].n
        
        # Track interaction statistics
        self.interaction_counts = np.zeros(self.K, dtype=int)  # How many times each demon won
    
    def select_winning_demon(self, bit_value: str, demon_states: List[str]) -> Tuple[int, float]:
        """Select which demon interacts based on competitive sampling.
        
        For each demon, we draw a time-to-next-event dt from exponential distribution
        based on the total rate from that demon's current joint state.
        The demon with the smallest dt wins.
        
        Args:
            bit_value (str): Current bit value ('0' or '1')
            demon_states (List[str]): Current states of all K demons
        
        Returns:
            winning_demon_idx (int): Index of the demon that wins
            winning_dt (float): The dt value for the winning demon
        """
        dts = []
        
        for k in range(self.K):
            # Form joint state for this demon
            joint_state = f'{bit_value}_{demon_states[k]}'
            
            # Get total rate for this demon
            rates = self.demons[k].get_rates_for_joint_state(joint_state)
            total_rate = sum(rates.values())
            
            if total_rate > 0:
                # Draw time-to-next-event from exponential distribution
                dt = np.random.exponential(1.0 / total_rate)
            else:
                # No transitions possible, set dt to infinity
                dt = np.inf
            
            dts.append(dt)
        
        # Find demon with smallest dt
        winning_idx = int(np.argmin(dts))
        winning_dt = dts[winning_idx]
        
        return winning_idx, winning_dt
    
    def run_gillespie_window_for_winning_demon(self, demon: CompetingDemon, 
                                               joint_state: str) -> Tuple[str, float]:
        """Run Gillespie algorithm for the winning demon's interaction.
        
        Args:
            demon (CompetingDemon): The winning demon
            joint_state (str): Initial joint state in the form 'bitValue_dX'
        
        Returns:
            final_state (str): Final joint state after the interaction window
            time_elapsed (float): Total time elapsed during the interaction window
        """
        time_elapsed = 0.0
        current_state = joint_state
        
        while time_elapsed < self.tau:
            rates_dict = demon.get_rates_for_joint_state(current_state)
            
            if not rates_dict:
                break
            
            transitions = list(rates_dict.keys())
            rates = np.array([rates_dict[t] for t in transitions])
            total_rate = np.sum(rates)
            
            if total_rate == 0:
                break
            
            # Time to next event
            dt = np.random.exponential(1.0 / total_rate)
            
            if time_elapsed + dt > self.tau:
                break  # Next event exceeds interaction window
            
            time_elapsed += dt
            
            # Choose which event occurs
            probabilities = rates / total_rate
            transition_idx = np.random.choice(len(transitions), p=probabilities)
            chosen_transition = transitions[transition_idx]
            
            # Extract final state
            current_state = chosen_transition.split('->')[1]
        
        return current_state, time_elapsed
    
    def process_bit_with_competing_demons(self, bit_value: str, 
                                         demon_states: List[str]) -> Tuple[str, List[str], int]:
        """Process a single bit with K competing demons.
        
        The demon with the shortest time-to-next-event wins and interacts with the bit.
        
        Args:
            bit_value (str): Initial bit value ('0' or '1')
            demon_states (List[str]): Current states of all K demons
        
        Returns:
            final_bit (str): Final bit value after interaction
            final_demon_states (List[str]): Final states of all K demons
            winning_demon_idx (int): Index of the demon that won the interaction
        """
        # Select winning demon based on competitive sampling
        winning_idx, _ = self.select_winning_demon(bit_value, demon_states)
        
        # Form joint state for the winning demon
        joint_state = f'{bit_value}_{demon_states[winning_idx]}'
        
        # Run interaction with winning demon
        final_state, _ = self.run_gillespie_window_for_winning_demon(
            self.demons[winning_idx], joint_state
        )
        
        # Extract new bit and demon state
        final_bit = final_state.split('_')[0]
        new_demon_state = final_state.split('_')[1]
        
        # Update demon states (only winning demon changes)
        final_demon_states = demon_states.copy()
        final_demon_states[winning_idx] = new_demon_state
        
        return final_bit, final_demon_states, winning_idx
    
    def run_full_simulation(self) -> Tuple[Tape, Tape, List[List[str]], np.ndarray]:
        """Run the full simulation over the entire tape with K competing demons.
        
        Returns:
            final_tape (Tape): The final state of the tape after the simulation
            initial_tape (Tape): The initial state of the tape before the simulation
            demon_states_history (List[List[str]]): History of demon states for each demon
            interaction_counts (np.ndarray): Number of times each demon won
        """
        final_tape = Tape(N=self.N, p0=self.init_tape.p0, tape_arr=self.init_tape_arr.copy())
        
        # Initialize demon states (all start at their initial states)
        current_demon_states = [demon.current_state for demon in self.demons]
        
        # Track demon state history for each demon
        demon_states_history = [[] for _ in range(self.K)]
        for k in range(self.K):
            demon_states_history[k].append(current_demon_states[k])
        
        # Reset interaction counts
        self.interaction_counts = np.zeros(self.K, dtype=int)
        
        # Process each bit with competing demons
        for i in range(self.N):
            initial_bit = final_tape.tape_arr[i]
            
            # Process bit with competing demons
            final_bit, current_demon_states, winning_idx = \
                self.process_bit_with_competing_demons(initial_bit, current_demon_states)
            
            # Update tape with final bit
            final_tape.tape_arr[i] = final_bit
            
            # Track winning demon
            self.interaction_counts[winning_idx] += 1
            
            # Record demon states
            for k in range(self.K):
                demon_states_history[k].append(current_demon_states[k])
        
        return final_tape, self.init_tape, demon_states_history, self.interaction_counts
    
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
        delta_e = self.demons[0].phys_params.DeltaE
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
            "K": self.K,
            "interaction_counts": self.interaction_counts,
            "interaction_fractions": self.interaction_counts / self.N
        }
        
        return stats


def plot_output_vs_K(K_values: List[int], output: str = 'phi', tape_params: Dict = None,
                     demon_n: int = 2, tau: float = 1.0, phys_params: PhysParams = None):
    """Plot selected output parameter vs number of competing demons K.
    
    Args:
        K_values (List[int]): List of K values (number of competing demons) to test
        output (str): Output to plot - 'phi', 'bias_out', 'Q_c', or 'delta_S_b'
        tape_params (dict): Tape parameters (N, p0)
        demon_n (int): Number of states per demon
        tau (float): Interaction time per bit
        phys_params (PhysParams): Physical parameters
    """
    if tape_params is None:
        tape_params = {"N": 5000, "p0": 1.0}
    
    if phys_params is None:
        phys_params = PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    
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
        demons = [CompetingDemon(n=demon_n, phys_params=phys_params, 
                                init_state='d0', demon_id=k) for k in range(K)]
        
        # Create tape
        tape = Tape(N=tape_params["N"], p0=tape_params["p0"])
        
        # Run simulation
        sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=tau)
        final_tape, _, _, _ = sim.run_full_simulation()
        
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
    
    plt.xlabel('Number of Competing Demons (K)', fontsize=12)
    plt.ylabel(output_labels[output], fontsize=12)
    
    # Title with parameters
    title = f'{output_labels[output]} vs Number of Competing Demons\n'
    title += f'n={demon_n}, τ={tau}, σ={phys_params.sigma:.3f}, ω={phys_params.omega:.3f}, '
    title += f'N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}'
    title += f'\n(Competitive interaction: shortest dt wins)'
    
    plt.title(title, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if output == 'phi':
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return y_values


def plot_interaction_distribution(interaction_counts: np.ndarray, K: int, 
                                  title: str = "Demon Interaction Distribution"):
    """Plot the distribution of interactions across K demons.
    
    Args:
        interaction_counts (np.ndarray): Count of interactions for each demon
        K (int): Number of demons
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    demon_labels = [f'Demon {k}' for k in range(K)]
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    
    plt.bar(demon_labels, interaction_counts, color=colors, alpha=0.7)
    plt.xlabel('Demon ID', fontsize=12)
    plt.ylabel('Number of Interactions', fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, count in enumerate(interaction_counts):
        plt.text(i, count, f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def run_single_simulation_demo():
    """Run a single simulation and display results."""
    print("=" * 70)
    print("COMPETING DEMONS SIMULATION DEMO")
    print("=" * 70)
    
    # Setup parameters
    phys_params = PhysParams(
        DeltaE=1.0,
        gamma=1.0,
        Th=1.6,
        Tc=1.0
    )
    
    tape_params = {'N': 5000, 'p0': 1.0}
    K = 5
    demon_n = 3
    tau = 1.0
    
    print(f"\nPhysical Parameters:")
    print(f"  ΔE: {phys_params.DeltaE}")
    print(f"  T_hot: {phys_params.Th}")
    print(f"  T_cold: {phys_params.Tc}")
    print(f"  γ: {phys_params.gamma}")
    print(f"  σ: {phys_params.sigma:.4f}")
    print(f"  ω: {phys_params.omega:.4f}")
    
    print(f"\nSimulation Parameters:")
    print(f"  Number of demons (K): {K}")
    print(f"  Demon states (n): {demon_n}")
    print(f"  Interaction time (τ): {tau}")
    print(f"  Tape length (N): {tape_params['N']}")
    print(f"  Initial p₀: {tape_params['p0']}")
    
    print(f"\nInteraction mechanism: Competitive sampling")
    print(f"  - Each demon draws dt from exponential(1/total_rate)")
    print(f"  - Demon with smallest dt wins and interacts with bit")
    
    # Create components
    demons = [CompetingDemon(n=demon_n, phys_params=phys_params, 
                            init_state='d0', demon_id=k) for k in range(K)]
    tape = Tape(N=tape_params['N'], p0=tape_params['p0'])
    sim = CompetingDemonSimulation(demons=demons, tape=tape, tau=tau)
    
    print("\nRunning simulation...")
    final_tape, initial_tape, demon_states_history, interaction_counts = \
        sim.run_full_simulation()
    
    # Compute statistics
    stats = sim.compute_statistics(final_tape)
    
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    
    print(f"\nBit-level statistics:")
    print(f"  Bit flip fraction (φ): {stats['phi']:.4f}")
    print(f"  Bias in: {stats['incoming']['bias']:.4f}")
    print(f"  Bias out: {stats['outgoing']['bias']:.4f}")
    print(f"  Bias change: {stats['bias']:+.4f}")
    
    print(f"\nEntropy:")
    print(f"  S_in: {stats['incoming']['entropy']:.4f}")
    print(f"  S_out: {stats['outgoing']['entropy']:.4f}")
    print(f"  ΔS: {stats['outgoing']['DeltaS_B']:+.4f}")
    
    print(f"\nEnergy:")
    print(f"  Q_c: {stats['Q_c']:.4f}")
    
    print(f"\nInteraction distribution:")
    for k in range(K):
        count = interaction_counts[k]
        fraction = count / tape_params['N']
        print(f"  Demon {k}: {count} interactions ({fraction:.2%})")
    
    print("\n" + "=" * 70)
    
    # Plot interaction distribution
    plot_interaction_distribution(
        interaction_counts, K,
        title=f"Demon Interaction Distribution (K={K}, n={demon_n}, τ={tau})"
    )


if __name__ == "__main__":
    # Run demo
    run_single_simulation_demo()
    
    # Uncomment to run phi vs K plot
    # print("\nGenerating φ vs K plot...")
    # plot_output_vs_K(
    #     K_values=list(range(1, 21)),
    #     output='phi',
    #     tape_params={"N": 5000, "p0": 1.0},
    #     demon_n=3,
    #     tau=1.0,
    #     phys_params=PhysParams(DeltaE=1.0, gamma=1.0, Th=1.6, Tc=1.0)
    # )
