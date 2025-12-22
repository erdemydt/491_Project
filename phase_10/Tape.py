"""
Phase 10: Two-Bit Tape Implementation

A tape designed for two-bit demon interactions, with support for:
- Half-and-half initialization (first half 0s, second half 1s, or vice versa)
- Controlled pair distributions ("00" and "11" pair fractions)
- Correlation analysis between pairs of bits
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List


class TwoBitTape:
    """A tape designed for two-bit demon interactions.
    
    The tape is processed in pairs: (bit_0, bit_1), (bit_2, bit_3), etc.
    This class provides methods to:
    - Initialize with controlled pair distributions
    - Analyze pair correlations
    - Compare initial and final tape states
    
    Attributes:
        N (int): Number of bits on the tape (should be even)
        n_pairs (int): Number of bit pairs (N // 2)
        tape_arr (np.ndarray): Array of bits ('0' and '1')
        p0 (float): Target probability of 0s
    """
    
    def __init__(self, N: int, p0: float = 0.5, seed: int = None, tape_arr: np.ndarray = None,
                 init_mode: str = 'random', pair_00_frac: float = None, pair_11_frac: float = None):
        """Initialize a TwoBitTape.
        
        Args:
            N (int): Number of bits (should be even for proper pairing)
            p0 (float): Probability of a bit being 0 (for random mode)
            seed (int): Random seed for reproducibility
            tape_arr (np.ndarray): Pre-existing tape array (overrides generation)
            init_mode (str): Initialization mode:
                - 'random': Independent random bits based on p0
                - 'half_split': First half all 0s, second half all 1s
                - 'half_split_reverse': First half all 1s, second half all 0s
                - 'alternating': Alternating 0s and 1s
                - 'pair_distribution': Control "00" and "11" pair fractions
            pair_00_frac (float): Fraction of "00" pairs (for 'pair_distribution' mode)
            pair_11_frac (float): Fraction of "11" pairs (for 'pair_distribution' mode)
        """
        if N % 2 != 0:
            print(f"Warning: N={N} is odd, will be treated as {N-1} for pair processing")
        
        self.N = N
        self.n_pairs = N // 2
        self.p0 = p0
        self.states = ['0', '1']
        self.pair_states = ['00', '01', '10', '11']
        self.init_mode = init_mode
        
        if seed is not None:
            np.random.seed(seed)
        
        if tape_arr is not None:
            self.tape_arr = tape_arr.copy()
        else:
            self.tape_arr = self._build_tape(init_mode, pair_00_frac, pair_11_frac)
        
        # Compute initial distribution
        self.probabilities = self._compute_bit_probabilities()
    
    def _build_tape(self, mode: str, pair_00_frac: float = None, 
                    pair_11_frac: float = None) -> np.ndarray:
        """Build the tape array based on the specified mode."""
        
        if mode == 'random':
            return np.random.choice(['0', '1'], size=self.N, p=[self.p0, 1 - self.p0])
        
        elif mode == 'half_split':
            # First half 0s, second half 1s
            tape = np.array(['0'] * (self.N // 2) + ['1'] * (self.N - self.N // 2))
            return tape
        
        elif mode == 'half_split_reverse':
            # First half 1s, second half 0s
            tape = np.array(['1'] * (self.N // 2) + ['0'] * (self.N - self.N // 2))
            return tape
        
        elif mode == 'alternating':
            # Alternating pattern
            tape = np.array(['0' if i % 2 == 0 else '1' for i in range(self.N)])
            return tape
        
        elif mode == 'pair_distribution':
            return self._build_pair_distribution_tape(pair_00_frac, pair_11_frac)
        
        else:
            raise ValueError(f"Unknown init_mode: {mode}")
    
    def _build_pair_distribution_tape(self, pair_00_frac: float = None, 
                                       pair_11_frac: float = None) -> np.ndarray:
        """Build tape with controlled pair distributions.
        
        Args:
            pair_00_frac: Fraction of "00" pairs (default: based on p0)
            pair_11_frac: Fraction of "11" pairs (default: based on p0)
            
        The remaining pairs are filled with "01" and "10" randomly.
        """
        # Default fractions based on independent bits with probability p0
        if pair_00_frac is None:
            pair_00_frac = self.p0 ** 2
        if pair_11_frac is None:
            pair_11_frac = (1 - self.p0) ** 2
        
        # Validate fractions
        if pair_00_frac + pair_11_frac > 1.0:
            raise ValueError(f"pair_00_frac + pair_11_frac must be <= 1, got {pair_00_frac + pair_11_frac}")
        
        remaining_frac = 1.0 - pair_00_frac - pair_11_frac
        pair_01_frac = remaining_frac / 2
        pair_10_frac = remaining_frac / 2
        
        # Calculate number of each pair type
        n_00 = int(self.n_pairs * pair_00_frac)
        n_11 = int(self.n_pairs * pair_11_frac)
        n_mixed = self.n_pairs - n_00 - n_11
        
        # Create pairs
        pairs = ['00'] * n_00 + ['11'] * n_11
        

        
        # Shuffle pairs
        np.random.shuffle(pairs)
        
        # Flatten to tape array
        tape = np.array(list(''.join(pairs)))
        # Fill remaining with 0 or 1 according to p0
        for _ in range(n_mixed*2):
            tape = np.append(tape, np.random.choice(['0', '1'], p=[self.p0, 1 - self.p0]))
        return tape
    
    def _compute_bit_probabilities(self) -> List[float]:
        """Compute the probability distribution of individual bits."""
        p0 = np.mean(self.tape_arr == '0')
        p1 = 1 - p0
        return [p0, p1]
    
    def get_pair_at(self, pair_index: int) -> str:
        """Get the bit pair at a specific pair index.
        
        Args:
            pair_index: Index of the pair (0 to n_pairs-1)
            
        Returns:
            str: The pair as a string (e.g., '01')
        """
        if pair_index < 0 or pair_index >= self.n_pairs:
            raise IndexError(f"Pair index must be in [0, {self.n_pairs-1}]")
        bit_idx = pair_index * 2
        return self.tape_arr[bit_idx] + self.tape_arr[bit_idx + 1]
    
    def set_pair_at(self, pair_index: int, pair_value: str):
        """Set the bit pair at a specific pair index.
        
        Args:
            pair_index: Index of the pair (0 to n_pairs-1)
            pair_value: The new pair value (e.g., '01')
        """
        if pair_index < 0 or pair_index >= self.n_pairs:
            raise IndexError(f"Pair index must be in [0, {self.n_pairs-1}]")
        if len(pair_value) != 2 or not all(b in '01' for b in pair_value):
            raise ValueError(f"pair_value must be a 2-char string of 0s and 1s, got '{pair_value}'")
        bit_idx = pair_index * 2
        self.tape_arr[bit_idx] = pair_value[0]
        self.tape_arr[bit_idx + 1] = pair_value[1]
    
    def get_all_pairs(self) -> List[str]:
        """Get all pairs from the tape."""
        return [self.get_pair_at(i) for i in range(self.n_pairs)]
    
    def get_pair_distribution(self) -> Dict[str, float]:
        """Get the distribution of pair types.
        
        Returns:
            dict: Maps pair type ('00', '01', '10', '11') to fraction
        """
        pairs = self.get_all_pairs()
        counts = {ps: pairs.count(ps) for ps in self.pair_states}
        return {ps: count / self.n_pairs for ps, count in counts.items()}
    
    def get_pair_counts(self) -> Dict[str, int]:
        """Get the counts of each pair type."""
        pairs = self.get_all_pairs()
        return {ps: pairs.count(ps) for ps in self.pair_states}
    
    def get_entropy(self) -> float:
        """Calculate the entropy of the tape based on bit distribution."""
        self.probabilities = self._compute_bit_probabilities()
        p0, p1 = self.probabilities
        terms = []
        if p0 > 0:
            terms.append(-p0 * np.log(p0))
        if p1 > 0:
            terms.append(-p1 * np.log(p1))
        return float(sum(terms))
    
    def get_pair_entropy(self) -> float:
        """Calculate the entropy based on pair distribution."""
        pair_dist = self.get_pair_distribution()
        entropy = 0.0
        for ps, prob in pair_dist.items():
            if prob > 0:
                entropy -= prob * np.log(prob)
        return entropy
    
    def compute_pair_correlation(self) -> float:
        """Compute correlation between first and second bits in pairs.
        
        Returns:
            float: Pearson correlation coefficient
        """
        first_bits = (self.tape_arr[::2] == '1').astype(float)
        second_bits = (self.tape_arr[1::2] == '1').astype(float)
        
        if len(first_bits) < 2:
            return 0.0
        
        corr = np.corrcoef(first_bits, second_bits)[0, 1]
        return 0.0 if np.isnan(corr) else corr
    
    def compute_neighboring_pair_correlation(self) -> float:
        """Compute correlation between consecutive pairs.
        
        Returns:
            float: Correlation between pair values (treating pairs as 0-3)
        """
        pairs = self.get_all_pairs()
        pair_values = np.array([int(p, 2) for p in pairs])  # Convert to integers 0-3
        
        if len(pair_values) < 2:
            return 0.0
        
        corr = np.corrcoef(pair_values[:-1], pair_values[1:])[0, 1]
        return 0.0 if np.isnan(corr) else corr
    
    def compute_mutual_information_pairs(self) -> float:
        """Compute mutual information between first and second bits in pairs.
        
        Returns:
            float: Mutual information in nats
        """
        pair_dist = self.get_pair_distribution()
        
        # Marginal probabilities
        p_first_0 = pair_dist['00'] + pair_dist['01']
        p_first_1 = pair_dist['10'] + pair_dist['11']
        p_second_0 = pair_dist['00'] + pair_dist['10']
        p_second_1 = pair_dist['01'] + pair_dist['11']
        
        # Compute MI
        mi = 0.0
        for (p_joint, p_m1, p_m2) in [
            (pair_dist['00'], p_first_0, p_second_0),
            (pair_dist['01'], p_first_0, p_second_1),
            (pair_dist['10'], p_first_1, p_second_0),
            (pair_dist['11'], p_first_1, p_second_1)
        ]:
            if p_joint > 0 and p_m1 > 0 and p_m2 > 0:
                mi += p_joint * np.log(p_joint / (p_m1 * p_m2))
        
        return mi
    
    def get_correlation_summary(self) -> Dict:
        """Get comprehensive correlation summary.
        
        Returns:
            dict: Various correlation metrics
        """
        return {
            'pair_correlation': self.compute_pair_correlation(),
            'neighboring_pair_correlation': self.compute_neighboring_pair_correlation(),
            'mutual_information': self.compute_mutual_information_pairs(),
            'pair_distribution': self.get_pair_distribution(),
            'bit_entropy': self.get_entropy(),
            'pair_entropy': self.get_pair_entropy(),
            'p0': self.probabilities[0],
            'p1': self.probabilities[1]
        }
    
    def copy(self) -> 'TwoBitTape':
        """Create a copy of this tape."""
        return TwoBitTape(
            N=self.N, 
            p0=self.p0, 
            tape_arr=self.tape_arr.copy(),
            init_mode='random'  # Doesn't matter since we provide tape_arr
        )
    
    def plot_tape(self, title: str = None, max_display: int = 200):
        """Visualize the tape.
        
        Args:
            title: Optional title for the plot
            max_display: Maximum number of bits to display
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 4))
        
        # Plot individual bits
        ax = axes[0]
        display_len = min(max_display, self.N)
        float_tape = (self.tape_arr[:display_len] == '1').astype(float)
        ax.imshow(float_tape.reshape(1, -1), cmap='binary', aspect='auto')
        ax.set_yticks([])
        ax.set_xlabel('Bit Position')
        ax.set_title(f'Bit Visualization (first {display_len} bits)' + (f' - {title}' if title else ''))
        
        # Plot pair distribution
        ax = axes[1]
        pair_dist = self.get_pair_distribution()
        bars = ax.bar(pair_dist.keys(), pair_dist.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_xlabel('Pair Type')
        ax.set_ylabel('Fraction')
        ax.set_title('Pair Distribution')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, pair_dist.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_analysis(self):
        """Plot comprehensive correlation analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Tape visualization
        ax = axes[0, 0]
        display_len = min(200, self.N)
        float_tape = (self.tape_arr[:display_len] == '1').astype(float)
        ax.imshow(float_tape.reshape(1, -1), cmap='binary', aspect='auto')
        ax.set_yticks([])
        ax.set_xlabel('Bit Position')
        ax.set_title(f'Tape Visualization (first {display_len} bits)')
        
        # 2. Pair distribution
        ax = axes[0, 1]
        pair_dist = self.get_pair_distribution()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(pair_dist.keys(), pair_dist.values(), color=colors)
        ax.set_xlabel('Pair Type')
        ax.set_ylabel('Fraction')
        ax.set_title('Pair Distribution')
        ax.set_ylim(0, max(pair_dist.values()) * 1.2 if max(pair_dist.values()) > 0 else 1)
        
        for bar, val in zip(bars, pair_dist.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Pair sequence visualization
        ax = axes[1, 0]
        pairs = self.get_all_pairs()
        pair_values = np.array([int(p, 2) for p in pairs])  # 0, 1, 2, 3
        display_pairs = min(100, len(pair_values))
        ax.plot(range(display_pairs), pair_values[:display_pairs], 'o-', markersize=3, linewidth=0.5)
        ax.set_xlabel('Pair Index')
        ax.set_ylabel('Pair Value (0-3)')
        ax.set_title(f'Pair Sequence (first {display_pairs} pairs)')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['00', '01', '10', '11'])
        ax.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary = self.get_correlation_summary()
        
        stats_text = f"""Tape Correlation Summary

Initialization Mode: {self.init_mode}
N = {self.N}, n_pairs = {self.n_pairs}

Bit Statistics:
• p₀ = {summary['p0']:.4f}
• p₁ = {summary['p1']:.4f}
• Bit Entropy = {summary['bit_entropy']:.4f}

Pair Statistics:
• Pair Entropy = {summary['pair_entropy']:.4f}
• Intra-pair Correlation = {summary['pair_correlation']:.4f}
• Inter-pair Correlation = {summary['neighboring_pair_correlation']:.4f}
• Mutual Information = {summary['mutual_information']:.4f}

Pair Distribution:
• "00": {summary['pair_distribution']['00']:.4f}
• "01": {summary['pair_distribution']['01']:.4f}
• "10": {summary['pair_distribution']['10']:.4f}
• "11": {summary['pair_distribution']['11']:.4f}
"""
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()


def compare_tapes(initial_tape: TwoBitTape, final_tape: TwoBitTape, title: str = "Tape Comparison"):
    """Compare two tapes and visualize the differences.
    
    Args:
        initial_tape: The initial tape state
        final_tape: The final tape state
        title: Title for the comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # Row 1: Pair distributions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (tape, label) in enumerate([(initial_tape, 'Initial'), (final_tape, 'Final')]):
        ax = axes[0, idx]
        pair_dist = tape.get_pair_distribution()
        bars = ax.bar(pair_dist.keys(), pair_dist.values(), color=colors)
        ax.set_xlabel('Pair Type')
        ax.set_ylabel('Fraction')
        ax.set_title(f'{label} Pair Distribution')
        ax.set_ylim(0, max(max(pair_dist.values()) * 1.2, 0.5))
        for bar, val in zip(bars, pair_dist.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Row 2: Change analysis
    # Left: Pair distribution change
    ax = axes[1, 0]
    init_dist = initial_tape.get_pair_distribution()
    final_dist = final_tape.get_pair_distribution()
    
    x = np.arange(4)
    width = 0.35
    
    init_vals = [init_dist[ps] for ps in ['00', '01', '10', '11']]
    final_vals = [final_dist[ps] for ps in ['00', '01', '10', '11']]
    
    ax.bar(x - width/2, init_vals, width, label='Initial', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, final_vals, width, label='Final', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['00', '01', '10', '11'])
    ax.set_xlabel('Pair Type')
    ax.set_ylabel('Fraction')
    ax.set_title('Pair Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Summary statistics comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    init_summary = initial_tape.get_correlation_summary()
    final_summary = final_tape.get_correlation_summary()
    
    delta_p0 = final_summary['p0'] - init_summary['p0']
    delta_entropy = final_summary['bit_entropy'] - init_summary['bit_entropy']
    delta_pair_corr = final_summary['pair_correlation'] - init_summary['pair_correlation']
    delta_mi = final_summary['mutual_information'] - init_summary['mutual_information']
    
    # Compute phi (fraction of bits flipped)
    bits_flipped = np.sum(initial_tape.tape_arr != final_tape.tape_arr)
    phi = bits_flipped / initial_tape.N
    
    stats_text = f"""{title}

                    Initial      Final        Change
─────────────────────────────────────────────────────
p₀              {init_summary['p0']:.4f}       {final_summary['p0']:.4f}      {delta_p0:+.4f}
Bit Entropy     {init_summary['bit_entropy']:.4f}       {final_summary['bit_entropy']:.4f}      {delta_entropy:+.4f}
Pair Entropy    {init_summary['pair_entropy']:.4f}       {final_summary['pair_entropy']:.4f}      {final_summary['pair_entropy']-init_summary['pair_entropy']:+.4f}
Pair Corr       {init_summary['pair_correlation']:.4f}       {final_summary['pair_correlation']:.4f}      {delta_pair_corr:+.4f}
MI              {init_summary['mutual_information']:.4f}       {final_summary['mutual_information']:.4f}      {delta_mi:+.4f}

Key Metrics:
• Bits flipped: {bits_flipped} / {initial_tape.N}
• Flip fraction (φ): {phi:.4f}
• ΔS (bit entropy): {delta_entropy:+.4f}
"""
    ax.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return {
        'phi': phi,
        'bits_flipped': bits_flipped,
        'delta_p0': delta_p0,
        'delta_entropy': delta_entropy,
        'delta_pair_correlation': delta_pair_corr,
        'delta_mutual_information': delta_mi,
        'initial_summary': init_summary,
        'final_summary': final_summary
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TwoBitTape Demo")
    print("=" * 60)
    
    # Demo 1: Random tape
    print("\n1. Random tape (p0=0.7):")
    tape_random = TwoBitTape(N=1000, p0=0.7, seed=42, init_mode='random')
    print(f"   Pair distribution: {tape_random.get_pair_distribution()}")
    print(f"   Pair correlation: {tape_random.compute_pair_correlation():.4f}")
    
    # Demo 2: Half-split tape
    print("\n2. Half-split tape:")
    tape_split = TwoBitTape(N=1000, init_mode='half_split')
    print(f"   Pair distribution: {tape_split.get_pair_distribution()}")
    print(f"   p0 = {tape_split.probabilities[0]:.4f}")
    
    # Demo 3: Controlled pair distribution
    print("\n3. Controlled pair distribution (60% '00', 20% '11'):")
    tape_controlled = TwoBitTape(N=1000, init_mode='pair_distribution', 
                                  pair_00_frac=0.6, pair_11_frac=0.2, seed=42)
    print(f"   Pair distribution: {tape_controlled.get_pair_distribution()}")
    
    # Demo 4: Visualize
    print("\n4. Visualizing controlled tape...")
    tape_controlled.plot_correlation_analysis()
