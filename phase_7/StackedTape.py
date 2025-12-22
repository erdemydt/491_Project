import numpy as np
import matplotlib.pyplot as plt

class StackedTape:
    """Tape that works with 2-bit pairs.
    
    The tape is still stored as individual bits, but we process them in pairs.
    """
    
    def __init__(self, N: int, p0: float, seed: int = 7, tape_arr: np.ndarray = None):
        """Initialize tape with N bits.
        
        Args:
            N (int): Number of bits (should be even for clean pair processing)
            p0 (float): Probability of individual bit being 0
            seed (int): Random seed
            tape_arr (np.ndarray): Pre-existing tape array (optional)
        """
        if N % 2 != 0:
            print(f"Warning: N={N} is odd. Adding 1 to make it even for pair processing.")
            N += 1
        
        self.N = N
        self.p0 = p0
        self.states = ['0', '1']
        self.pair_states = ['00', '01', '10', '11']
        
        if tape_arr is not None:
            self.tape_arr = tape_arr
        else:
            np.random.seed(seed)
            self.tape_arr = np.random.choice(['0', '1'], size=N, p=[p0, 1 - p0])
        
        self.probabilities = [p0, 1 - p0]
    
    def get_pair_at_index(self, idx: int) -> str:
        """Get the 2-bit pair starting at index idx.
        
        Args:
            idx (int): Starting index (should be even)
            
        Returns:
            str: Two-bit string like '00', '01', '10', or '11'
        """
        if idx < 0 or idx + 1 >= self.N:
            raise ValueError(f"Index {idx} out of bounds for pair access")
        return self.tape_arr[idx] + self.tape_arr[idx + 1]
    
    def set_pair_at_index(self, idx: int, pair: str):
        """Set the 2-bit pair starting at index idx.
        
        Args:
            idx (int): Starting index (should be even)
            pair (str): Two-bit string like '00', '01', '10', or '11'
        """
        if idx < 0 or idx + 1 >= self.N:
            raise ValueError(f"Index {idx} out of bounds for pair access")
        if len(pair) != 2 or pair not in self.pair_states:
            raise ValueError(f"Invalid pair: {pair}")
        
        self.tape_arr[idx] = pair[0]
        self.tape_arr[idx + 1] = pair[1]
    
    def get_num_pairs(self) -> int:
        """Get the number of 2-bit pairs in the tape."""
        return self.N // 2
    
    def get_pair_distribution(self) -> dict:
        """Get the distribution of 2-bit pairs in the tape.
        
        Returns:
            dict: Counts and probabilities for each pair type
        """
        counts = {pair: 0 for pair in self.pair_states}
        num_pairs = self.get_num_pairs()
        
        for i in range(0, self.N - 1, 2):
            pair = self.get_pair_at_index(i)
            counts[pair] += 1
        
        probabilities = {pair: count / num_pairs for pair, count in counts.items()}
        
        return {
            'counts': counts,
            'probabilities': probabilities,
            'num_pairs': num_pairs
        }
    
    def get_initial_distribution(self) -> np.ndarray:
        """Get the initial probability distribution of individual bits."""
        self.probabilities = [np.sum(self.tape_arr == state) / self.N for state in self.states]
        return np.array(self.probabilities)
    
    def get_entropy(self) -> float:
        """Calculate the Shannon entropy of the tape based on individual bits."""
        p0, p1 = self.probabilities
        terms = []
        if p0 > 0:
            terms.append(-p0 * np.log(p0))
        if p1 > 0:
            terms.append(-p1 * np.log(p1))
        return float(sum(terms))
    
    def get_pair_entropy(self) -> float:
        """Calculate the Shannon entropy based on 2-bit pairs."""
        pair_dist = self.get_pair_distribution()
        entropy = 0.0
        
        for prob in pair_dist['probabilities'].values():
            if prob > 0:
                entropy -= prob * np.log(prob)
        
        return entropy
    
    def plot_distribution(self):
        """Plot the full tape array."""
        plt.figure(figsize=(12, 3))
        
        # Plot individual bits
        plt.subplot(2, 1, 1)
        float_tape = np.array(self.tape_arr, dtype=float)
        plt.imshow(float_tape.reshape(1, -1), cmap='binary', aspect='auto')
        plt.yticks([])
        plt.title('Tape Bit Distribution (Individual Bits)')
        plt.xlabel('Bit Position')
        
        # Plot pair distribution
        plt.subplot(2, 1, 2)
        pair_dist = self.get_pair_distribution()
        pairs = list(pair_dist['probabilities'].keys())
        probs = list(pair_dist['probabilities'].values())
        
        plt.bar(pairs, probs, color='steelblue')
        plt.ylabel('Probability')
        plt.xlabel('Bit Pair')
        plt.title('2-Bit Pair Distribution')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_pairs(self, verbose: bool = True) -> dict:
        """Comprehensive analysis of 2-bit pairs in the tape.
        
        Args:
            verbose (bool): Whether to print results
            
        Returns:
            dict: Analysis results
        """
        pair_dist = self.get_pair_distribution()
        
        # Calculate expected probabilities (if bits were independent)
        p0 = np.sum(self.tape_arr == '0') / self.N
        p1 = 1 - p0
        
        expected_probs = {
            '00': p0 * p0,
            '01': p0 * p1,
            '10': p1 * p0,
            '11': p1 * p1
        }
        
        # Calculate deviations
        deviations = {}
        for pair in self.pair_states:
            actual = pair_dist['probabilities'][pair]
            expected = expected_probs[pair]
            deviations[pair] = actual - expected
        
        results = {
            'pair_distribution': pair_dist,
            'expected_probabilities': expected_probs,
            'deviations': deviations,
            'individual_bit_probs': {'0': p0, '1': p1},
            'bit_entropy': self.get_entropy(),
            'pair_entropy': self.get_pair_entropy()
        }
        
        if verbose:
            print("=" * 60)
            print("STACKED TAPE PAIR ANALYSIS")
            print("=" * 60)
            print(f"\nTape length: {self.N} bits ({self.get_num_pairs()} pairs)")
            print(f"\nIndividual bit probabilities:")
            print(f"  P(0) = {p0:.4f}")
            print(f"  P(1) = {p1:.4f}")
            print(f"  Bit entropy: {results['bit_entropy']:.4f}")
            
            print(f"\nPair distribution:")
            for pair in self.pair_states:
                actual = pair_dist['probabilities'][pair]
                expected = expected_probs[pair]
                deviation = deviations[pair]
                print(f"  {pair}: {actual:.4f} (expected: {expected:.4f}, deviation: {deviation:+.4f})")
            
            print(f"\nPair entropy: {results['pair_entropy']:.4f}")
            print(f"Expected pair entropy (if independent): {2 * results['bit_entropy']:.4f}")
            print("=" * 60)
        
        return results
