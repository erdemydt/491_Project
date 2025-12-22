import { useState, useRef, useCallback } from 'react';
import { 
  NumberInput, 
  SliderInput, 
  SelectInput, 
  FormSection, 
  Button 
} from '../components/forms';
import { Heatmap } from '../components/charts';
import { generatePhaseDiagram } from '../services/api';
import { 
  createDefaultConfig, 
  SWEEP_PARAMETERS, 
  OUTPUT_METRICS,
  generateSweepValues
} from '../utils/constants';

const COLOR_SCALES = [
  { value: 'viridis', label: 'Viridis' },
  { value: 'plasma', label: 'Plasma' },
  { value: 'coolwarm', label: 'Cool-Warm' },
];

export default function PhaseDiagramPage() {
  const [config, setConfig] = useState(createDefaultConfig());
  const [xParam, setXParam] = useState('sigma');
  const [yParam, setYParam] = useState('omega');
  const [xMin, setXMin] = useState(0.1);
  const [xMax, setXMax] = useState(0.9);
  const [xSteps, setXSteps] = useState(15);
  const [yMin, setYMin] = useState(0.1);
  const [yMax, setYMax] = useState(0.9);
  const [ySteps, setYSteps] = useState(15);
  const [outputMetric, setOutputMetric] = useState('phi');
  const [colorScale, setColorScale] = useState('viridis');
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  
  const heatmapRef = useRef(null);

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

  const runPhaseDiagram = async () => {
    setLoading(true);
    setError(null);
    setProgress(0);
    
    try {
      const xParamInfo = SWEEP_PARAMETERS.find(p => p.value === xParam);
      const yParamInfo = SWEEP_PARAMETERS.find(p => p.value === yParam);
      
      const xValues = generateSweepValues(xMin, xMax, xSteps, xParamInfo?.integer);
      const yValues = generateSweepValues(yMin, yMax, ySteps, yParamInfo?.integer);
      
      const totalPoints = xValues.length * yValues.length;
      
      // Simulate progress (actual progress would require backend changes)
      const progressInterval = setInterval(() => {
        setProgress(p => Math.min(p + 5, 95));
      }, 500);
      
      const res = await generatePhaseDiagram({
        base_config: config,
        x_param: xParam,
        y_param: yParam,
        x_values: xValues,
        y_values: yValues,
        output_metric: outputMetric,
      });
      
      clearInterval(progressInterval);
      setProgress(100);
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCellClick = useCallback((x, y, value) => {
    setSelectedPoint({ x, y, value });
  }, []);

  const downloadCSV = () => {
    if (!result) return;
    
    // Create CSV with x values as header row
    const headers = ['y\\x', ...result.x_values.map(v => v.toFixed(4))];
    const rows = result.y_values.map((yVal, yi) => {
      return [yVal.toFixed(4), ...result.grid[yi].map(v => v.toFixed(6))];
    });
    
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `phase_diagram_${xParam}_${yParam}_${outputMetric}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadSVG = () => {
    if (!heatmapRef.current) return;
    // For now, just download the data - SVG export would require additional implementation
    alert('SVG export coming soon! Use CSV for now.');
  };

  const getParamLabel = (param) => {
    return SWEEP_PARAMETERS.find(p => p.value === param)?.label || param;
  };

  const getMetricLabel = (metric) => {
    return OUTPUT_METRICS.find(m => m.value === metric)?.label || metric;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">📊 Phase Diagram</h1>
        {result && (
          <div className="flex space-x-2">
            <Button onClick={downloadCSV} variant="secondary" className="text-sm">
              📥 CSV
            </Button>
            <Button onClick={downloadSVG} variant="secondary" className="text-sm">
              🖼️ SVG
            </Button>
          </div>
        )}
      </div>

      <div className="grid md:grid-cols-4 gap-6">
        {/* Left Panel - Configuration */}
        <div className="md:col-span-1 space-y-4">
          {/* Axis Configuration */}
          <FormSection title="X-Axis Parameter">
            <SelectInput
              label="Parameter"
              value={xParam}
              onChange={setXParam}
              options={SWEEP_PARAMETERS.filter(p => p.value !== yParam)}
            />
            <div className="grid grid-cols-2 gap-2">
              <NumberInput
                label="Min"
                value={xMin}
                onChange={setXMin}
                step={0.01}
              />
              <NumberInput
                label="Max"
                value={xMax}
                onChange={setXMax}
                step={0.01}
              />
            </div>
            <NumberInput
              label="Resolution"
              value={xSteps}
              onChange={(v) => setXSteps(Math.max(2, Math.round(v)))}
              min={2}
              max={50}
              step={1}
            />
          </FormSection>

          <FormSection title="Y-Axis Parameter">
            <SelectInput
              label="Parameter"
              value={yParam}
              onChange={setYParam}
              options={SWEEP_PARAMETERS.filter(p => p.value !== xParam)}
            />
            <div className="grid grid-cols-2 gap-2">
              <NumberInput
                label="Min"
                value={yMin}
                onChange={setYMin}
                step={0.01}
              />
              <NumberInput
                label="Max"
                value={yMax}
                onChange={setYMax}
                step={0.01}
              />
            </div>
            <NumberInput
              label="Resolution"
              value={ySteps}
              onChange={(v) => setYSteps(Math.max(2, Math.round(v)))}
              min={2}
              max={50}
              step={1}
            />
          </FormSection>

          <FormSection title="Output & Display">
            <SelectInput
              label="Output Metric"
              value={outputMetric}
              onChange={setOutputMetric}
              options={OUTPUT_METRICS}
            />
            <SelectInput
              label="Color Scale"
              value={colorScale}
              onChange={setColorScale}
              options={COLOR_SCALES}
            />
          </FormSection>

          {/* Fixed Parameters */}
          <FormSection title="Fixed Parameters">
            <NumberInput
              label="τ (Interaction Time)"
              value={config.tau}
              onChange={(v) => updateConfig('tau', v)}
              min={0.01}
              max={100}
              step={0.1}
            />
            <NumberInput
              label="N (Tape Length)"
              value={config.tape_config.N}
              onChange={(v) => updateConfig('tape_config.N', Math.round(v))}
              min={100}
              max={10000}
              step={100}
            />
            <SliderInput
              label="p₀"
              value={config.tape_config.p0}
              onChange={(v) => updateConfig('tape_config.p0', v)}
              min={0}
              max={1}
            />
            <NumberInput
              label="n (Demon States)"
              value={config.demon_config.n}
              onChange={(v) => updateConfig('demon_config.n', Math.round(v))}
              min={2}
              max={20}
              step={1}
            />
            <NumberInput
              label="K (Stacked Demons)"
              value={config.demon_config.K}
              onChange={(v) => updateConfig('demon_config.K', Math.round(v))}
              min={1}
              max={20}
              step={1}
            />
          </FormSection>

          <Button
            onClick={runPhaseDiagram}
            loading={loading}
            className="w-full"
          >
            Generate Phase Diagram
          </Button>

          {loading && (
            <div className="bg-gray-900 rounded-lg p-3">
              <div className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Computing...</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {xSteps * ySteps} points • Please wait...
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-900/50 border border-red-500 rounded-lg p-3 text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Right Panel - Heatmap */}
        <div className="md:col-span-3">
          {!result && !loading && (
            <div className="flex items-center justify-center h-96 bg-gray-900 rounded-xl">
              <div className="text-center text-gray-500">
                <p className="text-4xl mb-4">📊</p>
                <p>Configure parameters and generate a phase diagram</p>
                <p className="text-sm mt-2">
                  Click on cells to see exact values
                </p>
              </div>
            </div>
          )}

          {loading && !result && (
            <div className="flex items-center justify-center h-96 bg-gray-900 rounded-xl">
              <div className="text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
                <p className="text-gray-400">Generating phase diagram...</p>
                <p className="text-sm text-gray-500 mt-2">
                  Computing {xSteps * ySteps} simulations
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4" ref={heatmapRef}>
              <Heatmap
                grid={result.grid}
                xValues={result.x_values}
                yValues={result.y_values}
                xLabel={getParamLabel(xParam)}
                yLabel={getParamLabel(yParam)}
                colorLabel={getMetricLabel(outputMetric)}
                colorScale={colorScale}
                onCellClick={handleCellClick}
                title={`${getMetricLabel(outputMetric)} Phase Diagram`}
              />

              {/* Selected Point Info */}
              {selectedPoint && (
                <div className="bg-gray-900 rounded-xl p-4">
                  <h3 className="text-lg font-medium text-white mb-2">Selected Point</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <span className="text-gray-500">{getParamLabel(xParam)}:</span>
                      <span className="text-white ml-2 font-mono">{selectedPoint.x.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">{getParamLabel(yParam)}:</span>
                      <span className="text-white ml-2 font-mono">{selectedPoint.y.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">{getMetricLabel(outputMetric)}:</span>
                      <span className="text-blue-400 ml-2 font-mono font-bold">{selectedPoint.value.toFixed(6)}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Statistics */}
              <div className="bg-gray-900 rounded-xl p-4">
                <h3 className="text-lg font-medium text-white mb-3">Statistics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-sm text-gray-500">Min</div>
                    <div className="text-lg text-white font-mono">
                      {Math.min(...result.grid.flat()).toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-sm text-gray-500">Max</div>
                    <div className="text-lg text-white font-mono">
                      {Math.max(...result.grid.flat()).toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-sm text-gray-500">Mean</div>
                    <div className="text-lg text-white font-mono">
                      {(result.grid.flat().reduce((a, b) => a + b, 0) / result.grid.flat().length).toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-sm text-gray-500">Grid Size</div>
                    <div className="text-lg text-white font-mono">
                      {result.x_values.length} × {result.y_values.length}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
