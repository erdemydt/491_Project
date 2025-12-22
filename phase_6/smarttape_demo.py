"""
Quick demo of SmartTape correlation features.
"""

from Tape import SmartTape

# Example 1: Create uncorrelated tape
print("Example 1: Uncorrelated tape")
tape_uncorr = SmartTape(N=1000, p0=0.7, correlation_type='none', seed=42)
summary = tape_uncorr.get_correlation_summary()
print(f"  Nearest neighbor correlation: {summary['nearest_neighbor_correlation']:.4f}")
print(f"  Mean block length: {summary['mean_block_length']:.2f}")

# Example 2: Create Markov correlated tape
print("\nExample 2: Markov correlated tape (strength=0.7)")
tape_markov = SmartTape(N=1000, p0=0.7, correlation_type='markov', 
                        correlation_strength=0.7, seed=42)
summary = tape_markov.get_correlation_summary()
print(f"  Nearest neighbor correlation: {summary['nearest_neighbor_correlation']:.4f}")
print(f"  Mean block length: {summary['mean_block_length']:.2f}")
print(f"  Mutual information (lag=1): {summary['mutual_information_lag1']:.4f}")

# Example 3: Create block correlated tape
print("\nExample 3: Block correlated tape (strength=0.8)")
tape_block = SmartTape(N=1000, p0=0.7, correlation_type='block', 
                       correlation_strength=0.8, seed=42)
summary = tape_block.get_correlation_summary()
print(f"  Nearest neighbor correlation: {summary['nearest_neighbor_correlation']:.4f}")
print(f"  Mean block length: {summary['mean_block_length']:.2f}")
print(f"  Max block length: {summary['max_block_length']}")

# Example 4: Visualize correlation analysis
print("\nExample 4: Plotting correlation analysis for Markov tape...")
tape_markov.plot_correlation_analysis(max_lag=50)
