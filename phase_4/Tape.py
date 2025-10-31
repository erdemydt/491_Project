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
            self.tape_arr = self.build_tape_array(N, p0)
        np.random.seed(seed)

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
        if p0 > 0: terms.append(-p0 * np.log(p0))
        if p1 > 0: terms.append(-p1 * np.log(p1))
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
        
    
    def analyze_patterns(self, pattern_length: int) -> dict:
        """Analyze n-gram patterns of specified length in the tape.
        
        Args:
            pattern_length (int): Length of patterns to analyze (e.g., 2 for '00', '01', '10', '11')
            
        Returns:
            dict: Dictionary with pattern counts, frequencies, and expected frequencies
        """
        if pattern_length < 1 or pattern_length > self.N:
            raise ValueError(f"Pattern length must be between 1 and {self.N}")
        
        # Generate all possible patterns of given length
        possible_patterns = []
        for i in range(2**pattern_length):
            pattern = format(i, f'0{pattern_length}b')
            possible_patterns.append(pattern)
        
        # Count occurrences of each pattern
        pattern_counts = {pattern: 0 for pattern in possible_patterns}
        
        # Slide through the tape to count patterns
        for i in range(self.N - pattern_length + 1):
            pattern = ''.join(self.tape_arr[i:i + pattern_length])
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
        
        # Calculate frequencies and expected frequencies
        total_patterns = self.N - pattern_length + 1
        pattern_frequencies = {pattern: count / total_patterns for pattern, count in pattern_counts.items()}
        
        # Expected frequencies based on independent probability
        p0_current = np.sum(self.tape_arr == '0') / self.N
        p1_current = 1 - p0_current
        expected_frequencies = {}
        for pattern in possible_patterns:
            prob = 1.0
            for bit in pattern:
                prob *= p0_current if bit == '0' else p1_current
            expected_frequencies[pattern] = prob
        
        return {
            'pattern_length': pattern_length,
            'counts': pattern_counts,
            'frequencies': pattern_frequencies,
            'expected_frequencies': expected_frequencies,
            'total_patterns': total_patterns
        }
    
    def comprehensive_pattern_analysis(self) -> dict:
        """Perform comprehensive pattern analysis without specific input.
        
        Analyzes various aspects of the tape including:
        - Single bit distribution
        - Bigram patterns (length 2)
        - Trigram patterns (length 3) if tape is long enough
        - Run length analysis (consecutive same bits)
        - Alternation patterns
        - Local clustering analysis
        
        Returns:
            dict: Comprehensive analysis results
        """
        results = {}
        
        # Basic statistics
        results['basic_stats'] = {
            'tape_length': self.N,
            'zeros_count': np.sum(self.tape_arr == '0'),
            'ones_count': np.sum(self.tape_arr == '1'),
            'zeros_frequency': np.sum(self.tape_arr == '0') / self.N,
            'ones_frequency': np.sum(self.tape_arr == '1') / self.N,
            'entropy': self.get_entropy()
        }
        
        # N-gram analysis (up to length 3 or tape length, whichever is smaller)
        max_pattern_length = min(4, self.N)
        results['ngram_analysis'] = {}
        for n in range(1, max_pattern_length + 1):
            if self.N >= n:
                results['ngram_analysis'][f'{n}_gram'] = self.analyze_patterns(n)
        
        # Run length analysis (consecutive same bits)
        runs = []
        current_bit = self.tape_arr[0]
        current_run_length = 1
        
        for i in range(1, self.N):
            if self.tape_arr[i] == current_bit:
                current_run_length += 1
            else:
                runs.append((current_bit, current_run_length))
                current_bit = self.tape_arr[i]
                current_run_length = 1
        runs.append((current_bit, current_run_length))
        
        # Analyze run statistics
        zero_runs = [length for bit, length in runs if bit == '0']
        one_runs = [length for bit, length in runs if bit == '1']
        
        results['run_analysis'] = {
            'total_runs': len(runs),
            'zero_runs': {
                'count': len(zero_runs),
                'lengths': zero_runs,
                'avg_length': np.mean(zero_runs) if zero_runs else 0,
                'max_length': max(zero_runs) if zero_runs else 0
            },
            'one_runs': {
                'count': len(one_runs),
                'lengths': one_runs,
                'avg_length': np.mean(one_runs) if one_runs else 0,
                'max_length': max(one_runs) if one_runs else 0
            }
        }
        
        # Alternation analysis
        alternations = 0
        for i in range(1, self.N):
            if self.tape_arr[i] != self.tape_arr[i-1]:
                alternations += 1
        
        results['alternation_analysis'] = {
            'total_alternations': alternations,
            'alternation_rate': alternations / (self.N - 1) if self.N > 1 else 0,
            'expected_alternation_rate': 2 * results['basic_stats']['zeros_frequency'] * results['basic_stats']['ones_frequency']
        }
        
        # Local clustering analysis (divide tape into segments and analyze)
        if self.N >= 10:
            segment_size = max(10, self.N // 10)
            num_segments = self.N // segment_size
            segment_entropies = []
            
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, self.N)
                segment = self.tape_arr[start_idx:end_idx]
                
                p0_seg = np.sum(segment == '0') / len(segment)
                p1_seg = 1 - p0_seg
                
                entropy_seg = 0
                if p0_seg > 0: entropy_seg -= p0_seg * np.log(p0_seg)
                if p1_seg > 0: entropy_seg -= p1_seg * np.log(p1_seg)
                
                segment_entropies.append(entropy_seg)
            
            results['local_analysis'] = {
                'segment_size': segment_size,
                'num_segments': num_segments,
                'segment_entropies': segment_entropies,
                'avg_local_entropy': np.mean(segment_entropies),
                'entropy_variance': np.var(segment_entropies),
                'global_entropy': results['basic_stats']['entropy']
            }
        
        return results
    
    def complete_pattern_analysis(self, specific_pattern_length: int = None, verbose: bool = True) -> dict:
        """Complete pattern analysis function that combines both specific and comprehensive analysis.
        
        Args:
            specific_pattern_length (int, optional): If provided, performs specific n-gram analysis
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Complete analysis results
        """
        print("=" * 60)
        print("COMPLETE TAPE PATTERN ANALYSIS")
        print("=" * 60)
        
        # Get comprehensive analysis
        comprehensive_results = self.comprehensive_pattern_analysis()
        
        # Get specific pattern analysis if requested
        specific_results = None
        if specific_pattern_length is not None:
            specific_results = self.analyze_patterns(specific_pattern_length)
        
        # Combine results
        complete_results = {
            'comprehensive_analysis': comprehensive_results,
            'specific_pattern_analysis': specific_results
        }
        
        if verbose:
            self._print_analysis_results(comprehensive_results, specific_results)
        
        return complete_results
    
    def _print_analysis_results(self, comprehensive_results: dict, specific_results: dict = None):
        """Helper function to print analysis results in a readable format."""
        
        # Basic statistics
        basic = comprehensive_results['basic_stats']
        print(f"\nBASIC STATISTICS:")
        print(f"Tape Length: {basic['tape_length']}")
        print(f"Zeros: {basic['zeros_count']} ({basic['zeros_frequency']:.3f})")
        print(f"Ones: {basic['ones_count']} ({basic['ones_frequency']:.3f})")
        print(f"Entropy: {basic['entropy']:.4f}")
        
        # N-gram analysis summary
        print(f"\nN-GRAM PATTERNS:")
        for gram_type, gram_data in comprehensive_results['ngram_analysis'].items():
            print(f"\n{gram_type.upper()} Analysis:")
            for pattern, freq in gram_data['frequencies'].items():
                expected = gram_data['expected_frequencies'][pattern]
                deviation = abs(freq - expected)
                print(f"  {pattern}: {freq:.4f} (expected: {expected:.4f}, deviation: {deviation:.4f})")
        
        # Run analysis
        runs = comprehensive_results['run_analysis']
        print(f"\nRUN LENGTH ANALYSIS:")
        print(f"Total runs: {runs['total_runs']}")
        print(f"Zero runs: {runs['zero_runs']['count']}, avg length: {runs['zero_runs']['avg_length']:.2f}")
        print(f"One runs: {runs['one_runs']['count']}, avg length: {runs['one_runs']['avg_length']:.2f}")
        
        # Alternation analysis
        alt = comprehensive_results['alternation_analysis']
        print(f"\nALTERNATION ANALYSIS:")
        print(f"Alternation rate: {alt['alternation_rate']:.4f} (expected: {alt['expected_alternation_rate']:.4f})")
        
        # Local analysis if available
        if 'local_analysis' in comprehensive_results:
            local = comprehensive_results['local_analysis']
            print(f"\nLOCAL CLUSTERING ANALYSIS:")
            print(f"Average local entropy: {local['avg_local_entropy']:.4f}")
            print(f"Global entropy: {local['global_entropy']:.4f}")
            print(f"Entropy variance: {local['entropy_variance']:.6f}")
        
        # Specific pattern analysis
        if specific_results:
            print(f"\nSPECIFIC {specific_results['pattern_length']}-GRAM DETAILED ANALYSIS:")
            print("-" * 40)
            for pattern in sorted(specific_results['counts'].keys()):
                count = specific_results['counts'][pattern]
                freq = specific_results['frequencies'][pattern]
                expected = specific_results['expected_frequencies'][pattern]
                print(f"Pattern '{pattern}': {count} occurrences ({freq:.4f} frequency, expected: {expected:.4f})")
        
        print("=" * 60)
        
