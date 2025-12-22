import { useState } from 'react';
import { 
  NumberInput, 
  SliderInput, 
  SelectInput, 
  FormSection, 
  Button 
} from '../components/forms';
import { SweepChart } from '../components/charts';
import { runSimulation, runParameterSweep } from '../services/api';
import { 
  createDefaultConfig, 
  SWEEP_PARAMETERS, 
  OUTPUT_METRICS,
  ENERGY_DISTRIBUTIONS,
  CORRELATION_TYPES,
  generateSweepValues
} from '../utils/constants';

function ResultCard({ label, value, description, color = 'blue' }) {
  const colors = {
    blue: 'border-blue-500 text-blue-400',
    green: 'border-green-500 text-green-400',
    yellow: 'border-yellow-500 text-yellow-400',
    purple: 'border-purple-500 text-purple-400',
    red: 'border-red-500 text-red-400',
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-4 border-l-4 ${colors[color]}`}>
      <div className="text-2xl font-bold">
        {typeof value === 'number' ? value.toFixed(4) : value}
      </div>
      <div className="text-gray-300 font-medium">{label}</div>
      {description && <div className="text-gray-500 text-sm mt-1">{description}</div>}
    </div>
  );
}

export default function SimulationPage() {
  const [config, setConfig] = useState(createDefaultConfig());
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Sweep mode state
  const [sweepMode, setSweepMode] = useState(false);
  const [sweepParam, setSweepParam] = useState('tau');
  const [sweepMin, setSweepMin] = useState(0.1);
  const [sweepMax, setSweepMax] = useState(10);
  const [sweepSteps, setSweepSteps] = useState(20);
  const [sweepMetric, setSweepMetric] = useState('phi');
  const [sweepResult, setSweepResult] = useState(null);

  const updateConfig = (path, value) => {
    setConfig(prev => {
      const newConfig = { ...prev };
      const keys = path.split('.');
      let obj = newConfig;
      for (let i = 0; i < keys.length - 1; i++) {
        obj[keys[i]] = { ...obj[keys[i]] };
        obj = obj[keys[i]];
      }
      obj[keys[keys.length - 1]] = value;
      return newConfig;
    });
  };

  const runSingleSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runSimulation(config);
      setResult(res);
      setSweepResult(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const runSweep = async () => {
    setLoading(true);
    setError(null);
    try {
      const paramInfo = SWEEP_PARAMETERS.find(p => p.value === sweepParam);
      const values = generateSweepValues(sweepMin, sweepMax, sweepSteps, paramInfo?.integer);
      
      const res = await runParameterSweep({
        base_config: config,
        sweep_param: sweepParam,
        sweep_values: values,
        output_metric: sweepMetric,
      });
      
      // Transform for recharts
      const chartData = res.param_values.map((x, i) => ({
        [res.param_name]: x,
        [res.output_name]: res.output_values[i],
      }));
      
      setSweepResult({ ...res, chartData });
      setResult(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (!sweepResult) return;
    
    const headers = [sweepResult.param_name, sweepResult.output_name];
    const rows = sweepResult.param_values.map((x, i) => 
      [x, sweepResult.output_values[i]].join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sweep_${sweepResult.param_name}_${sweepResult.output_name}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">⚡ Simulation</h1>
        <div className="flex items-center space-x-2">
          <span className="text-gray-400 text-sm">Mode:</span>
          <button
            onClick={() => setSweepMode(false)}
            className={`px-3 py-1 rounded-l-lg text-sm ${!sweepMode ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
          >
            Single Run
          </button>
          <button
            onClick={() => setSweepMode(true)}
            className={`px-3 py-1 rounded-r-lg text-sm ${sweepMode ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
          >
            Parameter Sweep
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Left Panel - Configuration */}
        <div className="md:col-span-1 space-y-4">
          {/* Physics Parameters */}
          <FormSection title="Physics Parameters">
            <SliderInput
              label="σ (Intrinsic Rate)"
              value={config.phys_params.sigma || 0.3}
              onChange={(v) => updateConfig('phys_params.sigma', v)}
              min={0}
              max={1}
              step={0.01}
              tooltip="Transition parameter with hot reservoir"
            />
            <SliderInput
              label="ω (Outgoing Rate)"
              value={config.phys_params.omega || 0.8}
              onChange={(v) => updateConfig('phys_params.omega', v)}
              min={0}
              max={1}
              step={0.01}
              tooltip="Transition parameter with cold reservoir"
            />
            <NumberInput
              label="ΔE (Energy Difference)"
              value={config.phys_params.DeltaE}
              onChange={(v) => updateConfig('phys_params.DeltaE', v)}
              min={0.01}
              max={10}
              step={0.1}
            />
            <NumberInput
              label="γ (Transition Rate)"
              value={config.phys_params.gamma}
              onChange={(v) => updateConfig('phys_params.gamma', v)}
              min={0.01}
              max={10}
              step={0.1}
            />
          </FormSection>

          {/* Demon Configuration */}
          <FormSection title="Demon Configuration">
            <NumberInput
              label="n (Demon States)"
              value={config.demon_config.n}
              onChange={(v) => updateConfig('demon_config.n', Math.max(2, Math.round(v)))}
              min={2}
              max={100}
              step={1}
            />
            <NumberInput
              label="K (Stacked Demons)"
              value={config.demon_config.K}
              onChange={(v) => updateConfig('demon_config.K', Math.max(1, Math.round(v)))}
              min={1}
              max={50}
              step={1}
            />
            <SelectInput
              label="Energy Distribution"
              value={config.demon_config.energy_distribution}
              onChange={(v) => updateConfig('demon_config.energy_distribution', v)}
              options={ENERGY_DISTRIBUTIONS}
            />
          </FormSection>

          {/* Tape Configuration */}
          <FormSection title="Tape Configuration">
            <NumberInput
              label="N (Tape Length)"
              value={config.tape_config.N}
              onChange={(v) => updateConfig('tape_config.N', Math.max(10, Math.round(v)))}
              min={10}
              max={100000}
              step={100}
            />
            <SliderInput
              label="p₀ (Initial Probability of 0)"
              value={config.tape_config.p0}
              onChange={(v) => updateConfig('tape_config.p0', v)}
              min={0}
              max={1}
              step={0.01}
            />
            <SelectInput
              label="Correlation Type"
              value={config.tape_config.correlation_type}
              onChange={(v) => updateConfig('tape_config.correlation_type', v)}
              options={CORRELATION_TYPES}
            />
            {config.tape_config.correlation_type !== 'none' && (
              <SliderInput
                label="Correlation Strength"
                value={config.tape_config.correlation_strength}
                onChange={(v) => updateConfig('tape_config.correlation_strength', v)}
                min={0}
                max={1}
                step={0.01}
              />
            )}
          </FormSection>

          {/* Interaction Time */}
          <FormSection title="Interaction Time">
            <NumberInput
              label="τ (Time per Demon)"
              value={config.tau}
              onChange={(v) => updateConfig('tau', Math.max(0.01, v))}
              min={0.01}
              max={100}
              step={0.1}
            />
          </FormSection>

          {/* Sweep Configuration (when in sweep mode) */}
          {sweepMode && (
            <FormSection title="Sweep Configuration">
              <SelectInput
                label="Sweep Parameter"
                value={sweepParam}
                onChange={setSweepParam}
                options={SWEEP_PARAMETERS}
              />
              <div className="grid grid-cols-2 gap-2">
                <NumberInput
                  label="Min"
                  value={sweepMin}
                  onChange={setSweepMin}
                  min={0}
                  step={0.1}
                />
                <NumberInput
                  label="Max"
                  value={sweepMax}
                  onChange={setSweepMax}
                  min={0}
                  step={0.1}
                />
              </div>
              <NumberInput
                label="Steps"
                value={sweepSteps}
                onChange={(v) => setSweepSteps(Math.max(2, Math.round(v)))}
                min={2}
                max={100}
                step={1}
              />
              <SelectInput
                label="Output Metric"
                value={sweepMetric}
                onChange={setSweepMetric}
                options={OUTPUT_METRICS}
              />
            </FormSection>
          )}

          {/* Run Button */}
          <Button
            onClick={sweepMode ? runSweep : runSingleSimulation}
            loading={loading}
            className="w-full"
          >
            {sweepMode ? 'Run Parameter Sweep' : 'Run Simulation'}
          </Button>

          {error && (
            <div className="bg-red-900/50 border border-red-500 rounded-lg p-3 text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="md:col-span-2">
          {loading && (
            <div className="flex items-center justify-center h-64 bg-gray-900 rounded-xl">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                <p className="text-gray-400">Running simulation...</p>
              </div>
            </div>
          )}

          {!loading && !result && !sweepResult && (
            <div className="flex items-center justify-center h-64 bg-gray-900 rounded-xl">
              <p className="text-gray-500">Configure parameters and run a simulation to see results</p>
            </div>
          )}

          {/* Single Simulation Results */}
          {result && !sweepResult && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-white">Results</h2>
              
              {/* Key Metrics */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <ResultCard
                  label="φ (Bit Flip Fraction)"
                  value={result.phi}
                  color="blue"
                />
                <ResultCard
                  label="Q_c (Energy to Cold)"
                  value={result.Q_c}
                  color="green"
                />
                <ResultCard
                  label="ΔS_B (Entropy Change)"
                  value={result.delta_S_b}
                  color="yellow"
                />
                <ResultCard
                  label="Bias Out"
                  value={result.bias_out}
                  color="purple"
                />
              </div>

              {/* Distribution Comparison */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-900 rounded-xl p-4">
                  <h3 className="text-lg font-medium text-white mb-3">Input Distribution</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">p₀ (probability of 0)</span>
                      <span className="text-white font-mono">{result.p0_in.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">p₁ (probability of 1)</span>
                      <span className="text-white font-mono">{result.p1_in.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Entropy</span>
                      <span className="text-white font-mono">{result.entropy_in.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Bias</span>
                      <span className="text-white font-mono">{result.bias_in.toFixed(4)}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-900 rounded-xl p-4">
                  <h3 className="text-lg font-medium text-white mb-3">Output Distribution</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">p₀ (probability of 0)</span>
                      <span className="text-white font-mono">{result.p0_out.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">p₁ (probability of 1)</span>
                      <span className="text-white font-mono">{result.p1_out.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Entropy</span>
                      <span className="text-white font-mono">{result.entropy_out.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Bias</span>
                      <span className="text-white font-mono">{result.bias_out.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Configuration Summary */}
              <div className="bg-gray-900 rounded-xl p-4">
                <h3 className="text-lg font-medium text-white mb-3">Configuration</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">N =</span>
                    <span className="text-white ml-1">{result.N}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">K =</span>
                    <span className="text-white ml-1">{result.K}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">n =</span>
                    <span className="text-white ml-1">{config.demon_config.n}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">τ =</span>
                    <span className="text-white ml-1">{config.tau}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">σ =</span>
                    <span className="text-white ml-1">{config.phys_params.sigma}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">ω =</span>
                    <span className="text-white ml-1">{config.phys_params.omega}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">ΔE =</span>
                    <span className="text-white ml-1">{config.phys_params.DeltaE}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">γ =</span>
                    <span className="text-white ml-1">{config.phys_params.gamma}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Sweep Results */}
          {sweepResult && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-white">Parameter Sweep Results</h2>
                <Button onClick={downloadCSV} variant="secondary" className="text-sm">
                  📥 Download CSV
                </Button>
              </div>
              
              <SweepChart
                data={sweepResult.chartData}
                xKey={sweepResult.param_name}
                yKey={sweepResult.output_name}
                xLabel={SWEEP_PARAMETERS.find(p => p.value === sweepResult.param_name)?.label || sweepResult.param_name}
                yLabel={OUTPUT_METRICS.find(m => m.value === sweepResult.output_name)?.label || sweepResult.output_name}
                color={OUTPUT_METRICS.find(m => m.value === sweepResult.output_name)?.color || '#3b82f6'}
                referenceLines={sweepResult.output_name === 'phi' ? [{ value: 0.5, label: 'φ = 0.5', color: '#ef4444' }] : []}
                title={`${OUTPUT_METRICS.find(m => m.value === sweepResult.output_name)?.label} vs ${SWEEP_PARAMETERS.find(p => p.value === sweepResult.param_name)?.label}`}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
