import { useState, useEffect, useMemo } from 'react';
import { 
  NumberInput, 
  SliderInput, 
  SelectInput, 
  FormSection, 
  Button 
} from '../components/forms';
import { validateDemon } from '../services/api';
import { ENERGY_DISTRIBUTIONS } from '../utils/constants';

/**
 * Visual energy level diagram component
 */
function EnergyLevelDiagram({ n, energyValues, deltaEValues, selectedState, onStateClick }) {
  const maxEnergy = Math.max(...energyValues);
  const diagramHeight = 300;
  
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <h3 className="text-lg font-medium text-white mb-4">Energy Level Diagram</h3>
      <div className="relative" style={{ height: diagramHeight }}>
        {/* Energy axis */}
        <div className="absolute left-0 top-0 bottom-0 w-8 flex flex-col justify-between text-xs text-gray-500">
          <span>{maxEnergy.toFixed(2)}</span>
          <span>E</span>
          <span>0</span>
        </div>
        
        {/* Energy levels */}
        <div className="absolute left-10 right-4 top-2 bottom-2">
          {energyValues.map((energy, i) => {
            const yPos = ((maxEnergy - energy) / maxEnergy) * (diagramHeight - 40) + 10;
            const isSelected = selectedState === i;
            
            return (
              <div key={i}>
                {/* Level line */}
                <div
                  className={`absolute left-0 right-0 h-1 rounded cursor-pointer transition-all ${
                    isSelected ? 'bg-blue-500 h-2' : 'bg-gray-500 hover:bg-gray-400'
                  }`}
                  style={{ top: yPos }}
                  onClick={() => onStateClick?.(i)}
                >
                  {/* State label */}
                  <span className={`absolute -left-2 -top-5 text-xs ${isSelected ? 'text-blue-400' : 'text-gray-400'}`}>
                    d{i}
                  </span>
                  {/* Energy value */}
                  <span className={`absolute right-0 -top-5 text-xs font-mono ${isSelected ? 'text-blue-400' : 'text-gray-400'}`}>
                    {energy.toFixed(3)}
                  </span>
                </div>
                
                {/* Delta E arrow (between levels) */}
                {i < n - 1 && (
                  <div
                    className="absolute right-1/4 flex flex-col items-center text-xs text-yellow-500"
                    style={{ 
                      top: yPos + 5,
                      height: ((energyValues[i + 1] - energy) / maxEnergy) * (diagramHeight - 40) - 10
                    }}
                  >
                    <div className="flex-1 w-px bg-yellow-500/50" />
                    <span className="my-1 bg-gray-800 px-1">
                      ΔE = {deltaEValues[i].toFixed(3)}
                    </span>
                    <div className="flex-1 w-px bg-yellow-500/50" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/**
 * Transition rates display
 */
function TransitionRates({ demon }) {
  if (!demon) return null;
  
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <h3 className="text-lg font-medium text-white mb-4">Transition Rates</h3>
      
      <div className="space-y-4">
        {/* Hot reservoir transitions (intrinsic) */}
        <div>
          <h4 className="text-sm text-red-400 mb-2">🔥 Hot Reservoir (Intrinsic)</h4>
          <div className="space-y-1 text-sm">
            {Array.from({ length: demon.n_states - 1 }, (_, i) => (
              <div key={i} className="flex justify-between text-gray-300">
                <span>d{i} ↔ d{i + 1}</span>
                <span className="font-mono text-red-300">γ · exp(±ΔE/{`{2T_h}`})</span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Cold reservoir transitions (outgoing) */}
        <div>
          <h4 className="text-sm text-blue-400 mb-2">❄️ Cold Reservoir (Bit-coupled)</h4>
          <div className="space-y-1 text-sm">
            {Array.from({ length: demon.n_states - 1 }, (_, i) => (
              <div key={i} className="flex justify-between text-gray-300">
                <span>0_d{i} → 1_d{i + 1}</span>
                <span className="font-mono text-blue-300">1 - ω</span>
              </div>
            ))}
            {Array.from({ length: demon.n_states - 1 }, (_, i) => (
              <div key={`rev-${i}`} className="flex justify-between text-gray-300">
                <span>1_d{i + 1} → 0_d{i}</span>
                <span className="font-mono text-blue-300">1 + ω</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function DemonDesignerPage() {
  const [n, setN] = useState(4);
  const [energyDistribution, setEnergyDistribution] = useState('uniform');
  const [totalDeltaE, setTotalDeltaE] = useState(3.0);
  const [sigma, setSigma] = useState(0.3);
  const [omega, setOmega] = useState(0.8);
  const [gamma, setGamma] = useState(1.0);
  
  const [demonInfo, setDemonInfo] = useState(null);
  const [selectedState, setSelectedState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Validate demon whenever config changes
  useEffect(() => {
    const validate = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await validateDemon({
          n,
          K: 1,
          energy_distribution: energyDistribution,
          init_state: 'd0',
        });
        
        if (result.valid) {
          setDemonInfo(result);
        } else {
          setError(result.error);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    validate();
  }, [n, energyDistribution]);

  // Calculate derived parameters
  const derivedParams = useMemo(() => {
    if (!demonInfo) return null;
    
    const Th = totalDeltaE / (2 * (n - 1) * Math.atanh(sigma));
    const Tc = totalDeltaE / (2 * (n - 1) * Math.atanh(omega));
    
    // Boltzmann factors
    const deltaEPerState = totalDeltaE / (n - 1);
    const boltzmannHot = Math.exp(-deltaEPerState / Th);
    const boltzmannCold = Math.exp(-deltaEPerState / Tc);
    
    return {
      Th: isFinite(Th) ? Th : null,
      Tc: isFinite(Tc) ? Tc : null,
      deltaEPerState,
      boltzmannHot: isFinite(boltzmannHot) ? boltzmannHot : null,
      boltzmannCold: isFinite(boltzmannCold) ? boltzmannCold : null,
    };
  }, [demonInfo, totalDeltaE, n, sigma, omega]);

  // Scale energy values based on totalDeltaE
  const scaledEnergyValues = useMemo(() => {
    if (!demonInfo) return [];
    const scale = totalDeltaE / demonInfo.total_delta_e;
    return demonInfo.energy_values.map(e => e * scale);
  }, [demonInfo, totalDeltaE]);

  const scaledDeltaEValues = useMemo(() => {
    if (!demonInfo) return [];
    const scale = totalDeltaE / demonInfo.total_delta_e;
    return demonInfo.delta_e_values.map(e => e * scale);
  }, [demonInfo, totalDeltaE]);

  const exportConfig = () => {
    const config = {
      demon_config: {
        n,
        energy_distribution: energyDistribution,
      },
      phys_params: {
        sigma,
        omega,
        DeltaE: totalDeltaE,
        gamma,
        delta_e_mode: 'total',
        demon_n: n,
      },
      derived: derivedParams,
      energy_levels: scaledEnergyValues,
      delta_e_values: scaledDeltaEValues,
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `demon_config_n${n}_${energyDistribution}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">😈 Demon Designer</h1>
        <Button onClick={exportConfig} variant="secondary" disabled={!demonInfo}>
          📥 Export Config
        </Button>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Left Panel - Configuration */}
        <div className="md:col-span-1 space-y-4">
          <FormSection title="Demon Structure">
            <NumberInput
              label="n (Number of States)"
              value={n}
              onChange={(v) => setN(Math.max(2, Math.min(20, Math.round(v))))}
              min={2}
              max={20}
              step={1}
              tooltip="Number of energy levels (d0 to d{n-1})"
            />
            <SelectInput
              label="Energy Distribution"
              value={energyDistribution}
              onChange={setEnergyDistribution}
              options={ENERGY_DISTRIBUTIONS}
              tooltip="How energy gaps are distributed between states"
            />
            <NumberInput
              label="Total ΔE"
              value={totalDeltaE}
              onChange={(v) => setTotalDeltaE(Math.max(0.1, v))}
              min={0.1}
              max={10}
              step={0.1}
              tooltip="Total energy difference from d0 to d{n-1}"
            />
          </FormSection>

          <FormSection title="Transition Parameters">
            <SliderInput
              label="σ (Intrinsic Rate)"
              value={sigma}
              onChange={setSigma}
              min={0.01}
              max={0.99}
              step={0.01}
              tooltip="Controls hot reservoir coupling: σ = tanh(ΔE/2Th)"
            />
            <SliderInput
              label="ω (Outgoing Rate)"
              value={omega}
              onChange={setOmega}
              min={0.01}
              max={0.99}
              step={0.01}
              tooltip="Controls cold reservoir coupling: ω = tanh(ΔE/2Tc)"
            />
            <NumberInput
              label="γ (Base Rate)"
              value={gamma}
              onChange={(v) => setGamma(Math.max(0.01, v))}
              min={0.01}
              max={10}
              step={0.1}
              tooltip="Base transition rate with hot reservoir"
            />
          </FormSection>

          {/* Derived Parameters */}
          {derivedParams && (
            <FormSection title="Derived Parameters">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Hot Temperature (Th)</span>
                  <span className="text-red-400 font-mono">
                    {derivedParams.Th ? derivedParams.Th.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Cold Temperature (Tc)</span>
                  <span className="text-blue-400 font-mono">
                    {derivedParams.Tc ? derivedParams.Tc.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">ΔE per state</span>
                  <span className="text-yellow-400 font-mono">
                    {derivedParams.deltaEPerState.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Th / Tc ratio</span>
                  <span className="text-white font-mono">
                    {derivedParams.Th && derivedParams.Tc 
                      ? (derivedParams.Th / derivedParams.Tc).toFixed(4) 
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </FormSection>
          )}

          {error && (
            <div className="bg-red-900/50 border border-red-500 rounded-lg p-3 text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Middle Panel - Energy Diagram */}
        <div className="md:col-span-1">
          {loading && (
            <div className="flex items-center justify-center h-64 bg-gray-800 rounded-xl">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            </div>
          )}
          
          {!loading && demonInfo && (
            <EnergyLevelDiagram
              n={n}
              energyValues={scaledEnergyValues}
              deltaEValues={scaledDeltaEValues}
              selectedState={selectedState}
              onStateClick={setSelectedState}
            />
          )}

          {/* State Details */}
          {selectedState !== null && demonInfo && (
            <div className="mt-4 bg-gray-800 rounded-xl p-4">
              <h3 className="text-lg font-medium text-white mb-2">
                State d{selectedState}
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Energy</span>
                  <span className="text-white font-mono">{scaledEnergyValues[selectedState].toFixed(4)}</span>
                </div>
                {selectedState < n - 1 && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">ΔE to d{selectedState + 1}</span>
                    <span className="text-yellow-400 font-mono">{scaledDeltaEValues[selectedState].toFixed(4)}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-gray-400">Can transition to</span>
                  <span className="text-blue-400">
                    {[
                      selectedState > 0 ? `d${selectedState - 1}` : null,
                      selectedState < n - 1 ? `d${selectedState + 1}` : null,
                    ].filter(Boolean).join(', ') || 'None'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Transitions */}
        <div className="md:col-span-1">
          <TransitionRates demon={demonInfo} />
          
          {/* Configuration Summary */}
          <div className="mt-4 bg-gray-800 rounded-xl p-4">
            <h3 className="text-lg font-medium text-white mb-4">Configuration Summary</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Number of states</span>
                <span className="text-white">{n}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Distribution</span>
                <span className="text-white">{energyDistribution}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Total ΔE</span>
                <span className="text-white">{totalDeltaE}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">States</span>
                <span className="text-gray-300 text-xs">
                  {demonInfo?.states?.join(', ') || 'Loading...'}
                </span>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-4 space-y-2">
            <Button className="w-full" variant="secondary">
              🔄 Reset to Defaults
            </Button>
            <Button className="w-full" variant="primary" onClick={() => window.location.href = '/simulation'}>
              ⚡ Use in Simulation
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
