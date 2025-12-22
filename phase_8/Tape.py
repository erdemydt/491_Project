import numpy as np
import matplotlib.pyplot as plt


class Tape:
    """Class representing a tape of bits.
    
    Attributes:
        N (int): Number of bits on the tape.
        p0 (float): Probability of a bit being 0.
        states (list): List of bit states ('0' and '1').
        state_indices (dict): Mapping from state to index.
        tape_arr (np.ndarray): Array representing the tape bits.
    """
    def __init__(self, N: int, p0: float, seed: int = 7, tape_arr: np.ndarray = None):
        self.N = N
        self.p0 = p0
        self.states = ['0', '1']
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.probabilities = [p0, 1 - p0]
        
        if tape_arr is not None:
            self.tape_arr = tape_arr
        else:
            np.random.seed(seed)
            self.tape_arr = self.build_tape_array(N, p0)

    def build_tape_array(self, N: int, p0: float) -> np.ndarray:
        """Build the tape array with bits 0 and 1 based on probability p0."""
        return np.random.choice(['0', '1'], size=N, p=[p0, 1 - p0])
    
    def get_initial_distribution(self) -> np.ndarray:
        """Get the initial probability distribution of the tape."""
        self.probabilities = [np.sum(self.tape_arr == state) / self.N for state in self.states] 
        return np.array(self.probabilities)
    
    def get_entropy(self) -> float:
        """Calculate the entropy of the tape."""
        p0, p1 = self.probabilities
        terms = []
        if p0 > 0: 
            terms.append(-p0 * np.log(p0))
        if p1 > 0: 
            terms.append(-p1 * np.log(p1))
        return float(sum(terms))
    
    def plot_distribution(self):
        """Plot the full tape array"""
        plt.figure(figsize=(10, 2))
        float_tape = np.array(self.tape_arr, dtype=float)
        plt.imshow(float_tape.reshape(1, -1), cmap='binary', aspect='auto')
        plt.yticks([])
        plt.xticks(range(self.N))
        plt.title('Tape Bit Distribution')
        plt.xlabel('Bit Position')
        plt.show()
