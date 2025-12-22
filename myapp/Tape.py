import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict


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


class SmartTape(Tape):
    """Enhanced Tape class with support for correlated bits.
    
    This class extends the basic Tape to support different correlation patterns
    and provides methods to analyze correlations in the tape.
    
    Attributes:
        N (int): Number of bits on the tape.
        p0 (float): Probability of a bit being 0.
        correlation_type (str): Type of correlation ('none', 'markov', 'block', 'periodic').
        correlation_strength (float): Strength of correlation (0 = uncorrelated, 1 = maximum correlation).
        tape_arr (np.ndarray): Array representing the tape bits.
    """
    
    def __init__(self, N: int, p0: float, seed: int = 7, tape_arr: np.ndarray = None,
                 correlation_type: str = 'none', correlation_strength: float = 0.0,
                 block_size: Optional[int] = None, period: Optional[int] = None):
        """Initialize a SmartTape with optional correlations.
        
        Args:
            N (int): Number of bits on the tape.
            p0 (float): Probability of a bit being 0.
            seed (int): Random seed for reproducibility.
            tape_arr (np.ndarray): Pre-existing tape array (overrides generation).
            correlation_type (str): Type of correlation to introduce:
                - 'none': Independent bits (default)
                - 'markov': First-order Markov chain (neighboring bits correlated)
                - 'block': Blocks of identical bits
                - 'periodic': Periodic pattern with noise
            correlation_strength (float): Strength of correlation (0 to 1).
            block_size (int): Average block size for 'block' correlation.
            period (int): Period length for 'periodic' correlation.
        """
        self.correlation_type = correlation_type
        self.correlation_strength = correlation_strength
        self.block_size = block_size
        self.period = period
        
        # Initialize parent class without building the tape yet
        self.N = N
        self.p0 = p0
        self.states = ['0', '1']
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.probabilities = [p0, 1 - p0]
        
        if tape_arr is not None:
            self.tape_arr = tape_arr
        else:
            np.random.seed(seed)
            self.tape_arr = self._build_correlated_tape()
    
    def _build_correlated_tape(self) -> np.ndarray:
        """Build tape array with specified correlation pattern."""
        if self.correlation_type == 'none':
            return self.build_tape_array(self.N, self.p0)
        
        elif self.correlation_type == 'markov':
            return self._build_markov_tape()
        
        elif self.correlation_type == 'block':
            return self._build_block_tape()
        
        elif self.correlation_type == 'periodic':
            return self._build_periodic_tape()
        
        else:
            raise ValueError(f"Unknown correlation_type: {self.correlation_type}")
    
    def _build_markov_tape(self) -> np.ndarray:
        """Build tape using first-order Markov chain.
        
        correlation_strength controls the probability of staying in the same state:
        - 0.0: completely independent (like 'none')
        - 1.0: maximum correlation (tendency to stay in same state)
        """
        tape = np.empty(self.N, dtype=str)
        
        # Start with initial bit based on p0
        tape[0] = '0' if np.random.random() < self.p0 else '1'
        
        # Transition probabilities
        # P(next=current | current) = base_prob + correlation_strength * (1 - base_prob)
        p_stay_0 = self.p0 + self.correlation_strength * (1 - self.p0)
        p_stay_1 = (1 - self.p0) + self.correlation_strength * self.p0
        
        for i in range(1, self.N):
            if tape[i-1] == '0':
                tape[i] = '0' if np.random.random() < p_stay_0 else '1'
            else:
                tape[i] = '1' if np.random.random() < p_stay_1 else '0'
        
        return tape
    
    def _build_block_tape(self) -> np.ndarray:
        """Build tape with blocks of identical bits.
        
        correlation_strength controls block sizes:
        - 0.0: average block size of 1 (uncorrelated)
        - 1.0: very large blocks
        """
        if self.block_size is None:
            # Default block size based on correlation strength
            avg_block_size = 1 + int(self.correlation_strength * 20)
        else:
            avg_block_size = self.block_size
        
        tape = []
        while len(tape) < self.N:
            # Choose bit value based on p0
            bit_value = '0' if np.random.random() < self.p0 else '1'
            
            # Determine block size (exponential distribution)
            if self.correlation_strength == 0:
                block_length = 1
            else:
                block_length = max(1, int(np.random.exponential(avg_block_size)))
            
            # Add block
            tape.extend([bit_value] * block_length)
        
        return np.array(tape[:self.N])
    
    def _build_periodic_tape(self) -> np.ndarray:
        """Build tape with periodic pattern and noise.
        
        correlation_strength controls how strongly the pattern is preserved:
        - 0.0: completely random (no periodicity)
        - 1.0: perfect periodic pattern
        """
        if self.period is None:
            self.period = max(2, int(10 * (1 - self.correlation_strength) + 2))
        
        # Create base periodic pattern
        pattern = np.random.choice(['0', '1'], size=self.period, 
                                   p=[self.p0, 1 - self.p0])
        
        # Tile the pattern
        n_repeats = (self.N // self.period) + 1
        tape = np.tile(pattern, n_repeats)[:self.N]
        
        # Add noise based on correlation_strength
        if self.correlation_strength < 1.0:
            noise_prob = 1.0 - self.correlation_strength
            noise_mask = np.random.random(self.N) < noise_prob
            noise_bits = np.random.choice(['0', '1'], size=self.N, 
                                         p=[self.p0, 1 - self.p0])
            tape[noise_mask] = noise_bits[noise_mask]
        
        return tape
    
    def compute_autocorrelation(self, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute autocorrelation function of the tape.
        
        Args:
            max_lag (int): Maximum lag to compute (default: N//10)
        
        Returns:
            lags (np.ndarray): Array of lag values
            autocorr (np.ndarray): Autocorrelation values for each lag
        """
        if max_lag is None:
            max_lag = min(100, self.N // 10)
        
        # Convert to numeric (0 and 1)
        numeric_tape = (self.tape_arr == '1').astype(float)
        
        # Center the data
        centered = numeric_tape - np.mean(numeric_tape)
        
        # Compute autocorrelation
        lags = np.arange(max_lag + 1)
        autocorr = np.zeros(max_lag + 1)
        
        variance = np.var(numeric_tape)
        if variance > 0:
            for lag in lags:
                if lag == 0:
                    autocorr[lag] = 1.0
                else:
                    autocorr[lag] = np.mean(centered[:-lag] * centered[lag:]) / variance
        
        return lags, autocorr
    
    def compute_nearest_neighbor_correlation(self) -> float:
        """Compute correlation between nearest neighboring bits.
        
        Returns:
            float: Pearson correlation coefficient between adjacent bits
        """
        numeric_tape = (self.tape_arr == '1').astype(float)
        
        if self.N < 2:
            return 0.0
        
        # Compute correlation between tape[i] and tape[i+1]
        corr = np.corrcoef(numeric_tape[:-1], numeric_tape[1:])[0, 1]
        
        # Handle NaN (occurs when variance is zero)
        return 0.0 if np.isnan(corr) else corr
    
    def compute_block_statistics(self) -> Dict[str, float]:
        """Compute statistics about runs/blocks of identical bits.
        
        Returns:
            dict: Statistics including mean block length, max block length, etc.
        """
        if self.N == 0:
            return {'mean_block_length': 0, 'max_block_length': 0, 'n_blocks': 0}
        
        # Find runs of identical bits
        block_lengths = []
        current_length = 1
        
        for i in range(1, self.N):
            if self.tape_arr[i] == self.tape_arr[i-1]:
                current_length += 1
            else:
                block_lengths.append(current_length)
                current_length = 1
        block_lengths.append(current_length)
        
        return {
            'mean_block_length': np.mean(block_lengths),
            'max_block_length': np.max(block_lengths),
            'std_block_length': np.std(block_lengths),
            'n_blocks': len(block_lengths),
            'block_lengths': block_lengths
        }
    
    def compute_mutual_information(self, lag: int = 1) -> float:
        """Compute mutual information between bits separated by lag.
        
        Args:
            lag (int): Separation between bits
        
        Returns:
            float: Mutual information in nats
        """
        if lag >= self.N:
            return 0.0
        
        # Get pairs of bits
        bits_1 = self.tape_arr[:-lag]
        bits_2 = self.tape_arr[lag:]
        
        # Compute joint probabilities
        p_00 = np.mean((bits_1 == '0') & (bits_2 == '0'))
        p_01 = np.mean((bits_1 == '0') & (bits_2 == '1'))
        p_10 = np.mean((bits_1 == '1') & (bits_2 == '0'))
        p_11 = np.mean((bits_1 == '1') & (bits_2 == '1'))
        
        # Marginal probabilities
        p_0 = p_00 + p_01
        p_1 = p_10 + p_11
        p_0_lag = p_00 + p_10
        p_1_lag = p_01 + p_11
        
        # Compute mutual information
        mi = 0.0
        for p_joint, p_marg1, p_marg2 in [(p_00, p_0, p_0_lag),
                                           (p_01, p_0, p_1_lag),
                                           (p_10, p_1, p_0_lag),
                                           (p_11, p_1, p_1_lag)]:
            if p_joint > 0 and p_marg1 > 0 and p_marg2 > 0:
                mi += p_joint * np.log(p_joint / (p_marg1 * p_marg2))
        
        return mi
    
    def get_correlation_summary(self) -> Dict:
        """Get comprehensive correlation summary of the tape.
        
        Returns:
            dict: Dictionary containing various correlation metrics
        """
        lags, autocorr = self.compute_autocorrelation(max_lag=20)
        nn_corr = self.compute_nearest_neighbor_correlation()
        block_stats = self.compute_block_statistics()
        mi = self.compute_mutual_information(lag=1)
        
        return {
            'nearest_neighbor_correlation': nn_corr,
            'autocorr_lag1': autocorr[1] if len(autocorr) > 1 else 0.0,
            'autocorr_lag5': autocorr[5] if len(autocorr) > 5 else 0.0,
            'mutual_information_lag1': mi,
            'mean_block_length': block_stats['mean_block_length'],
            'max_block_length': block_stats['max_block_length'],
            'n_blocks': block_stats['n_blocks'],
            'autocorrelation_full': (lags, autocorr),
            'block_stats': block_stats
        }
    
    def plot_correlation_analysis(self, max_lag: int = 50):
        """Plot comprehensive correlation analysis of the tape.
        
        Args:
            max_lag (int): Maximum lag for autocorrelation plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Tape visualization
        ax = axes[0, 0]
        float_tape = np.array(self.tape_arr, dtype=float)
        display_len = min(200, self.N)
        ax.imshow(float_tape[:display_len].reshape(1, -1), cmap='binary', aspect='auto')
        ax.set_yticks([])
        ax.set_xlabel('Bit Position')
        ax.set_title(f'Tape Visualization (first {display_len} bits)')
        
        # 2. Autocorrelation function
        ax = axes[0, 1]
        lags, autocorr = self.compute_autocorrelation(max_lag=max_lag)
        ax.plot(lags, autocorr, marker='o', markersize=3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation Function')
        ax.grid(True, alpha=0.3)
        
        # 3. Block length distribution
        ax = axes[1, 0]
        block_stats = self.compute_block_statistics()
        block_lengths = block_stats['block_lengths']
        ax.hist(block_lengths, bins=min(50, max(block_lengths)), 
                color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Block Length')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Block Length Distribution\n(Mean: {block_stats["mean_block_length"]:.2f})')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary = self.get_correlation_summary()
        
        stats_text = f"""Correlation Summary
        
Correlation Type: {self.correlation_type}
Correlation Strength: {self.correlation_strength:.3f}

Key Metrics:
• Nearest Neighbor Corr: {summary['nearest_neighbor_correlation']:.4f}
• Autocorr (lag=1): {summary['autocorr_lag1']:.4f}
• Autocorr (lag=5): {summary['autocorr_lag5']:.4f}
• Mutual Info (lag=1): {summary['mutual_information_lag1']:.4f}

Block Statistics:
• Mean block length: {summary['mean_block_length']:.2f}
• Max block length: {summary['max_block_length']}
• Number of blocks: {summary['n_blocks']}

Tape Properties:
• N = {self.N}
• p₀ = {self.p0:.3f}
• Entropy = {self.get_entropy():.4f}
"""
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()

