import numpy as np
import matplotlib.pyplot as plt
from StackedDemon import StackedDemon, StackedPhysParams
from StackedTape import StackedTape

class StackedSimulation:
    """Gillespie simulation for stacked demon interacting with bit pairs."""
    
    def __init__(self, demon: StackedDemon, tape: StackedTape, tau: float):
        """Initialize the simulation.
        
        Args:
            demon (StackedDemon): The demon with 3 states
            tape (StackedTape): Input tape with bits
            tau (float): Interaction time per bit pair
        """
        self.demon = demon
        self.input_tape = tape
        self.tau = tau
        self.current_demon_state = demon.current_state
        
        # We'll process pairs, so tape length in pairs
        self.N_pairs = tape.get_num_pairs()
        
        # Initialize output tape (copy of input)
        self.output_tape = StackedTape(tape.N, tape.p0, tape_arr=tape.tape_arr.copy())
    
    def run_gillespie_window_for_joint_state(self, joint_state: str) -> tuple[str, float]:
        """Run Gillespie algorithm for one interaction window.
        
        Args:
            joint_state (str): Current joint state like '00_d0'
            
        Returns:
            tuple[str, float]: (final_joint_state, total_time_elapsed)
        """
        current_state = joint_state
        time_elapsed = 0.0
        
        while time_elapsed < self.tau:
            # Get all possible transitions and rates
            rates_dict = self.demon.get_rates_for_joint_state(current_state)
            
            if not rates_dict:
                # No transitions possible, stay in current state
                break
            
            transitions = list(rates_dict.keys())
            rates = np.array([rates_dict[t] for t in transitions])
            total_rate = np.sum(rates)
            
            if total_rate == 0:
                break
            
            # Sample time to next transition
            dt = np.random.exponential(1.0 / total_rate)
            
            # Check if we exceed tau
            if time_elapsed + dt > self.tau:
                break
            
            # Choose which transition occurs
            probabilities = rates / total_rate
            transition_idx = np.random.choice(len(transitions), p=probabilities)
            chosen_transition = transitions[transition_idx]
            
            # Extract final state from transition string
            # Format: 'XX_dY->ZZ_dW'
            final_state = chosen_transition.split('->')[1]
            
            # Update state and time
            current_state = final_state
            time_elapsed += dt
        
        return current_state, time_elapsed
    
    def run_full_simulation(self) -> tuple[StackedTape, StackedTape, list]:
        """Run the full simulation over all bit pairs.
        
        Returns:
            tuple: (input_tape, output_tape, demon_state_sequence)
        """
        demon_state_sequence = []
        
        # Process tape in pairs (indices 0-1, 2-3, 4-5, ...)
        for pair_idx in range(self.N_pairs):
            bit_idx = pair_idx * 2  # Convert pair index to bit index
            
            # Get input bit pair
            input_pair = self.input_tape.get_pair_at_index(bit_idx)
            
            # Form joint state
            joint_state = f'{input_pair}_{self.current_demon_state}'
            
            # Run Gillespie for this interaction window
            final_joint_state, _ = self.run_gillespie_window_for_joint_state(joint_state)
            
            # Extract output pair and new demon state
            output_pair, new_demon_state = final_joint_state.split('_')
            
            # Update output tape
            self.output_tape.set_pair_at_index(bit_idx, output_pair)
            
            # Update demon state for next interaction
            self.current_demon_state = new_demon_state
            demon_state_sequence.append(new_demon_state)
        
        return self.input_tape, self.output_tape, demon_state_sequence
    
    def compute_statistics(self, input_tape: StackedTape, output_tape: StackedTape) -> dict:
        """Compute statistics comparing input and output tapes.
        
        Args:
            input_tape (StackedTape): Input tape
            output_tape (StackedTape): Output tape
            
        Returns:
            dict: Statistics including phi, bias, entropy changes, etc.
        """
        # Get distributions
        input_dist = input_tape.get_initial_distribution()
        output_dist = output_tape.get_initial_distribution()
        
        p0_in, p1_in = input_dist
        p0_out, p1_out = output_dist
        
        # Bit flip fraction
        flips = np.sum(input_tape.tape_arr != output_tape.tape_arr)
        phi = flips / input_tape.N
        
        # Bias change
        bias_in = p0_in - p1_in
        bias_out = p0_out - p1_out
        bias_change = bias_out - bias_in
        
        # Entropy change
        S_in = input_tape.get_entropy()
        S_out = output_tape.get_entropy()
        delta_S = S_out - S_in
        
        # Pair-based statistics
        pair_dist_in = input_tape.get_pair_distribution()
        pair_dist_out = output_tape.get_pair_distribution()
        
        # Count pair flips (how many pairs changed)
        pair_flips = 0
        for i in range(0, input_tape.N - 1, 2):
            if input_tape.get_pair_at_index(i) != output_tape.get_pair_at_index(i):
                pair_flips += 1
        
        pair_flip_fraction = pair_flips / input_tape.get_num_pairs()
        
        return {
            'phi': phi,  # Individual bit flip fraction
            'pair_flip_fraction': pair_flip_fraction,  # Pair change fraction
            'bias_in': bias_in,
            'bias_out': bias_out,
            'bias_change': bias_change,
            'p0_in': p0_in,
            'p1_in': p1_in,
            'p0_out': p0_out,
            'p1_out': p1_out,
            'S_in': S_in,
            'S_out': S_out,
            'delta_S': delta_S,
            'pair_dist_in': pair_dist_in,
            'pair_dist_out': pair_dist_out,
            'num_bit_flips': flips,
            'num_pair_changes': pair_flips,
            'N': input_tape.N,
            'N_pairs': input_tape.get_num_pairs()
        }


def plot_demon_states(demon_states_sequence: list[str], title: str = "Demon State Distribution"):
    """Plot the demon states histogram."""
    unique_states = ['d0', 'd1', 'd2']
    state_counts = {state: demon_states_sequence.count(state) for state in unique_states}
    
    plt.figure(figsize=(8, 5))
    colors = ['blue', 'orange', 'red']
    plt.bar(state_counts.keys(), state_counts.values(), width=0.6, color=colors, alpha=0.7)
    plt.xlabel('Demon State', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_phi_vs_tau(tau_values: list, tape_params: dict = None, 
                    phys_params: StackedPhysParams = None):
    """Plot bit flip fraction vs interaction time tau.
    
    Args:
        tau_values (list): List of tau values to test
        tape_params (dict): Tape parameters {'N': int, 'p0': float}
        phys_params (StackedPhysParams): Physical parameters
    """
    if tape_params is None:
        tape_params = {'N': 1000, 'p0': 0.9}
    if phys_params is None:
        phys_params = StackedPhysParams(DeltaE_1=0.5, DeltaE_2=0.5, gamma=1.0, Th=2.0, Tc=1.0)
    
    phi_values = []
    pair_flip_values = []
    
    for i, tau in enumerate(tau_values):
        print(f"Progress: {i+1}/{len(tau_values)} (tau={tau:.2f})", end='\r')
        
        tape = StackedTape(N=tape_params['N'], p0=tape_params['p0'])
        demon = StackedDemon(phys_params=phys_params)
        sim = StackedSimulation(demon, tape, tau)
        
        input_tape, output_tape, demon_seq = sim.run_full_simulation()
        stats = sim.compute_statistics(input_tape, output_tape)
        
        phi_values.append(stats['phi'])
        pair_flip_values.append(stats['pair_flip_fraction'])
    
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, phi_values, marker='o', linewidth=2, markersize=6, 
             color='steelblue', label='Bit Flip Fraction (φ)')
    plt.plot(tau_values, pair_flip_values, marker='s', linewidth=2, markersize=6,
             color='crimson', label='Pair Change Fraction')
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Interaction Time (τ)', fontsize=12)
    plt.ylabel('Flip Fraction', fontsize=12)
    plt.title(f'Flip Fractions vs Interaction Time\n' +
              f'ΔE₁={phys_params.DeltaE_1}, ΔE₂={phys_params.DeltaE_2}, ' +
              f'Th={phys_params.Th}, Tc={phys_params.Tc}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}',
              fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def run_single_simulation_demo():
    """Run a single simulation and display results."""
    print("=" * 70)
    print("STACKED DEMON SIMULATION DEMO")
    print("=" * 70)
    
    # Setup parameters
    phys_params = StackedPhysParams(
        DeltaE_1=1.,  # Energy gap low->medium
        DeltaE_2=1.,  # Energy gap medium->high
        gamma=1.0,
        Th=2.47,
        Tc=.58
    )
    
    tape_params = {'N': 4000, 'p0': 1.}  # 4000 bits = 2000 pairs
    tau = 10.0
    
    print(f"\nPhysical Parameters:")
    print(f"  ΔE₁ (d0->d1): {phys_params.DeltaE_1}")
    print(f"  ΔE₂ (d1->d2): {phys_params.DeltaE_2}")
    print(f"  Total ΔE: {phys_params.DeltaE_1 + phys_params.DeltaE_2}")
    print(f"  T_hot: {phys_params.Th}")
    print(f"  T_cold: {phys_params.Tc}")
    print(f"  γ: {phys_params.gamma}")
    print(f"  τ: {tau}")
    
    print(f"\nTape Parameters:")
    print(f"  N: {tape_params['N']} bits ({tape_params['N']//2} pairs)")
    print(f"  p₀: {tape_params['p0']}")
    
    # Create components
    tape = StackedTape(N=tape_params['N'], p0=tape_params['p0'])
    demon = StackedDemon(phys_params=phys_params)
    sim = StackedSimulation(demon, tape, tau)
    
    print("\nRunning simulation...")
    input_tape, output_tape, demon_seq = sim.run_full_simulation()
    
    print("\nAnalyzing input tape:")
    input_tape.analyze_pairs(verbose=True)
    
    print("\nAnalyzing output tape:")
    output_tape.analyze_pairs(verbose=True)
    
    # Compute and display statistics
    stats = sim.compute_statistics(input_tape, output_tape)
    
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"\nBit-level statistics:")
    print(f"  Bit flip fraction (φ): {stats['phi']:.4f}")
    print(f"  Number of bit flips: {stats['num_bit_flips']}/{stats['N']}")
    print(f"  Bias in: {stats['bias_in']:.4f}")
    print(f"  Bias out: {stats['bias_out']:.4f}")
    print(f"  Bias change: {stats['bias_change']:+.4f}")
    
    print(f"\nPair-level statistics:")
    print(f"  Pair change fraction: {stats['pair_flip_fraction']:.4f}")
    print(f"  Number of pair changes: {stats['num_pair_changes']}/{stats['N_pairs']}")
    
    print(f"\nEntropy:")
    print(f"  S_in: {stats['S_in']:.4f}")
    print(f"  S_out: {stats['S_out']:.4f}")
    print(f"  ΔS: {stats['delta_S']:+.4f}")
    
    print(f"\nDemon state statistics:")
    demon_counts = {state: demon_seq.count(state) for state in ['d0', 'd1', 'd2']}
    total = len(demon_seq)
    print(f"  d0 (low): {demon_counts['d0']} ({demon_counts['d0']/total:.2%})")
    print(f"  d1 (medium): {demon_counts['d1']} ({demon_counts['d1']/total:.2%})")
    print(f"  d2 (high): {demon_counts['d2']} ({demon_counts['d2']/total:.2%})")
    
    print("=" * 70)
    
    # Plot demon state distribution
    plot_demon_states(demon_seq, 
                     title=f"Demon State Distribution (τ={tau}, N_pairs={stats['N_pairs']})")

def plot_phi_vs_tau(tau_values: list, tape_params: dict = None, 
                    phys_params: StackedPhysParams = None):
    """Plot bit flip fraction vs interaction time tau.
    
    Args:
        tau_values (list): List of tau values to test
        tape_params (dict): Tape parameters {'N': int, 'p0': float}
        phys_params (StackedPhysParams): Physical parameters
    """
    if tape_params is None:
        tape_params = {'N': 1000, 'p0': 0.9}
    if phys_params is None:
        phys_params = StackedPhysParams(DeltaE_1=0.5, DeltaE_2=0.5, gamma=1.0, Th=2.0, Tc=1.0)
    
    phi_values = []
    pair_flip_values = []
    
    for i, tau in enumerate(tau_values):
        print(f"Progress: {i+1}/{len(tau_values)} (tau={tau:.2f})", end='\r')
        
        tape = StackedTape(N=tape_params['N'], p0=tape_params['p0'])
        demon = StackedDemon(phys_params=phys_params)
        sim = StackedSimulation(demon, tape, tau)
        
        input_tape, output_tape, demon_seq = sim.run_full_simulation()
        stats = sim.compute_statistics(input_tape, output_tape)
        
        phi_values.append(stats['phi'])
        pair_flip_values.append(stats['pair_flip_fraction'])
    
    print()
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, phi_values, marker='o', linewidth=2, markersize=6, 
             color='steelblue', label='Bit Flip Fraction (φ)')
    plt.plot(tau_values, pair_flip_values, marker='s', linewidth=2, markersize=6,
             color='crimson', label='Pair Change Fraction')
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Interaction Time (τ)', fontsize=12)
    plt.ylabel('Flip Fraction', fontsize=12)
    plt.title(f'Flip Fractions vs Interaction Time\n' +
              f'ΔE₁={phys_params.DeltaE_1}, ΔE₂={phys_params.DeltaE_2}, ' +
              f'Th={phys_params.Th}, Tc={phys_params.Tc}, N={tape_params["N"]}, p₀={tape_params["p0"]:.2f}',
              fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run demo
    run_single_simulation_demo()
    
    # Uncomment to run phi vs tau plot
    # print("\nGenerating φ vs τ plot...")
    # plot_phi_vs_tau(
    #     tau_values=np.linspace(0.1, 20.0, 20).tolist(),
    #     tape_params={'N': 1000, 'p0': 0.9},
    #     phys_params=StackedPhysParams(DeltaE_1=0.5, DeltaE_2=0.5, gamma=1.0, Th=2.0, Tc=1.0)
    # )
