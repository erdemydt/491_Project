/**
 * Default configurations and constants for simulations
 */

export const DEFAULT_PHYS_PARAMS = {
  sigma: 0.3,
  omega: 0.8,
  Th: null,
  Tc: null,
  DeltaE: 1.0,
  gamma: 1.0,
  delta_e_mode: 'per_state',
  preserve_mode: 'sigma_omega',
};

export const DEFAULT_TAPE_CONFIG = {
  N: 1000,
  p0: 0.5,
  seed: null,
  tape_type: 'standard',
  correlation_type: 'none',
  correlation_strength: 0.0,
  block_size: null,
  period: null,
};

export const DEFAULT_DEMON_CONFIG = {
  n: 2,
  K: 1,
  energy_distribution: 'uniform',
  init_state: 'd0',
};

export const DEFAULT_TAU = 1.0;

export const SWEEP_PARAMETERS = [
  { value: 'tau', label: 'τ (Interaction Time)', min: 0.01, max: 100, step: 0.1 },
  { value: 'K', label: 'K (Stacked Demons)', min: 1, max: 50, step: 1, integer: true },
  { value: 'n', label: 'n (Demon States)', min: 2, max: 100, step: 1, integer: true },
  { value: 'sigma', label: 'σ (Intrinsic Rate)', min: 0, max: 1, step: 0.01 },
  { value: 'omega', label: 'ω (Outgoing Rate)', min: 0, max: 1, step: 0.01 },
  { value: 'p0', label: 'p₀ (Initial Probability)', min: 0, max: 1, step: 0.01 },
  { value: 'DeltaE', label: 'ΔE (Energy Difference)', min: 0.01, max: 10, step: 0.1 },
  { value: 'gamma', label: 'γ (Transition Rate)', min: 0.01, max: 10, step: 0.1 },
];

export const OUTPUT_METRICS = [
  { value: 'phi', label: 'φ (Bit Flip Fraction)', description: 'Fraction of bits flipped from 0 to 1', color: '#3b82f6' },
  { value: 'Q_c', label: 'Q_c (Energy to Cold)', description: 'Total energy transferred to cold reservoir', color: '#10b981' },
  { value: 'delta_S_b', label: 'ΔS_B (Entropy Change)', description: 'Change in bit entropy', color: '#f59e0b' },
  { value: 'bias', label: 'Bias', description: 'Output bias (p₀ - p₁)', color: '#8b5cf6' },
  { value: 'entropy_out', label: 'S_out (Output Entropy)', description: 'Final tape entropy', color: '#ef4444' },
];

export const ENERGY_DISTRIBUTIONS = [
  { value: 'uniform', label: 'Uniform', description: 'Equal energy steps between states' },
  { value: 'exponential', label: 'Exponential', description: 'Smaller gaps at lower states' },
  { value: 'quadratic', label: 'Quadratic', description: 'Quadratically increasing gaps' },
];

export const CORRELATION_TYPES = [
  { value: 'none', label: 'None', description: 'Independent bits' },
  { value: 'markov', label: 'Markov', description: 'First-order Markov chain' },
  { value: 'block', label: 'Block', description: 'Blocks of identical bits' },
  { value: 'periodic', label: 'Periodic', description: 'Periodic pattern with noise' },
];

// Helper to create default simulation config
export const createDefaultConfig = () => ({
  phys_params: { ...DEFAULT_PHYS_PARAMS },
  tape_config: { ...DEFAULT_TAPE_CONFIG },
  demon_config: { ...DEFAULT_DEMON_CONFIG },
  tau: DEFAULT_TAU,
});

// Generate array of values for sweep
export const generateSweepValues = (min, max, steps, isInteger = false) => {
  const values = [];
  const step = (max - min) / (steps - 1);
  for (let i = 0; i < steps; i++) {
    const value = min + i * step;
    values.push(isInteger ? Math.round(value) : parseFloat(value.toFixed(4)));
  }
  return values;
};
