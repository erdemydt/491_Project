"""
Maxwell Demon Lab - Backend API

FastAPI server exposing simulation endpoints for the React frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import numpy as np
import sys
import os
import uvicorn
# Add parent directory to path to import simulation modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Demon import Demon, PhysParams
from Tape import Tape, SmartTape
from Simulation import StackedDemonSimulation

app = FastAPI(
    title="Maxwell Demon Lab API",
    description="Backend API for Maxwell's Demon simulation experiments",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============

class PhysParamsInput(BaseModel):
    """Physical parameters for the simulation."""
    sigma: Optional[float] = Field(None, ge=-1, le=1, description="Intrinsic transition parameter")
    omega: Optional[float] = Field(None, ge=-1, le=1, description="Outgoing transition parameter")
    Th: Optional[float] = Field(None, gt=0, description="Hot reservoir temperature")
    Tc: Optional[float] = Field(None, gt=0, description="Cold reservoir temperature")
    DeltaE: float = Field(1.0, gt=0, description="Energy difference")
    gamma: float = Field(1.0, gt=0, description="Transition rate")
    delta_e_mode: Literal['per_state', 'total'] = 'per_state'
    preserve_mode: Literal['sigma_omega', 'temperatures'] = 'sigma_omega'


class TapeConfigInput(BaseModel):
    """Tape configuration."""
    N: int = Field(1000, ge=10, le=100000, description="Number of bits")
    p0: float = Field(0.5, ge=0, le=1, description="Probability of 0")
    seed: Optional[int] = Field(None, description="Random seed")
    tape_type: Literal['standard', 'smart'] = 'standard'
    # SmartTape options
    correlation_type: Literal['none', 'markov', 'block', 'periodic'] = 'none'
    correlation_strength: float = Field(0.0, ge=0, le=1)
    block_size: Optional[int] = None
    period: Optional[int] = None


class DemonConfigInput(BaseModel):
    """Demon configuration."""
    n: int = Field(2, ge=2, le=100, description="Number of demon states")
    K: int = Field(1, ge=1, le=50, description="Number of stacked demons")
    energy_distribution: Literal['uniform', 'exponential', 'quadratic'] = 'uniform'
    init_state: str = 'd0'


class SimulationRequest(BaseModel):
    """Full simulation request."""
    phys_params: PhysParamsInput
    tape_config: TapeConfigInput
    demon_config: DemonConfigInput
    tau: float = Field(1.0, gt=0, description="Interaction time per demon")


class SweepRequest(BaseModel):
    """1D parameter sweep request."""
    base_config: SimulationRequest
    sweep_param: str = Field(..., description="Parameter to sweep (tau, K, n, sigma, omega, p0)")
    sweep_values: List[float] = Field(..., min_length=2, description="Values to sweep over")
    output_metric: Literal['phi', 'Q_c', 'delta_S_b', 'bias', 'entropy_out'] = 'phi'


class PhaseDiagramRequest(BaseModel):
    """2D phase diagram request."""
    base_config: SimulationRequest
    x_param: str = Field(..., description="X-axis parameter")
    y_param: str = Field(..., description="Y-axis parameter")
    x_values: List[float] = Field(..., min_length=2)
    y_values: List[float] = Field(..., min_length=2)
    output_metric: Literal['phi', 'Q_c', 'delta_S_b', 'bias', 'entropy_out'] = 'phi'


class SimulationResult(BaseModel):
    """Simulation result."""
    phi: float
    Q_c: float
    delta_S_b: float
    bias_in: float
    bias_out: float
    entropy_in: float
    entropy_out: float
    p0_in: float
    p1_in: float
    p0_out: float
    p1_out: float
    N: int
    K: int


class SweepResult(BaseModel):
    """1D sweep result."""
    param_name: str
    param_values: List[float]
    output_name: str
    output_values: List[float]


class PhaseDiagramResult(BaseModel):
    """2D phase diagram result."""
    x_param: str
    y_param: str
    x_values: List[float]
    y_values: List[float]
    output_name: str
    grid: List[List[float]]  # 2D array [y][x]


# ============== Helper Functions ==============

def create_phys_params(config: PhysParamsInput, demon_n: int = 2) -> PhysParams:
    """Create PhysParams from input config."""
    kwargs = {
        'DeltaE': config.DeltaE,
        'gamma': config.gamma,
        'delta_e_mode': config.delta_e_mode,
        'preserve_mode': config.preserve_mode,
    }
    
    if config.delta_e_mode == 'total':
        kwargs['demon_n'] = demon_n
    
    # Use either sigma/omega or Th/Tc
    if config.sigma is not None and config.omega is not None:
        kwargs['sigma'] = config.sigma
        kwargs['omega'] = config.omega
    elif config.Th is not None and config.Tc is not None:
        kwargs['Th'] = config.Th
        kwargs['Tc'] = config.Tc
    else:
        # Default to sigma/omega
        kwargs['sigma'] = 0.3
        kwargs['omega'] = 0.8
    
    return PhysParams(**kwargs)


def create_tape(config: TapeConfigInput) -> Tape:
    """Create Tape from input config."""
    seed = config.seed if config.seed is not None else np.random.randint(0, 10000)
    
    if config.tape_type == 'smart':
        return SmartTape(
            N=config.N,
            p0=config.p0,
            seed=seed,
            correlation_type=config.correlation_type,
            correlation_strength=config.correlation_strength,
            block_size=config.block_size,
            period=config.period
        )
    else:
        return Tape(N=config.N, p0=config.p0, seed=seed)


def run_single_simulation(request: SimulationRequest) -> dict:
    """Run a single simulation and return results."""
    # Create physics params
    phys_params = create_phys_params(
        request.phys_params, 
        demon_n=request.demon_config.n
    )
    
    # Create tape
    tape = create_tape(request.tape_config)
    
    # Create demons
    demons = [
        Demon(
            n=request.demon_config.n,
            phys_params=phys_params,
            init_state=request.demon_config.init_state,
            energy_distribution=request.demon_config.energy_distribution
        )
        for _ in range(request.demon_config.K)
    ]
    for _ in range(request.demon_config.K):
        print("Validating demon config:", request.demon_config)
    
    # Run simulation
    sim = StackedDemonSimulation(demons=demons, tape=tape, tau=request.tau)
    final_tape, initial_tape, _ = sim.run_full_simulation()
    
    # Compute statistics
    stats = sim.compute_statistics(final_tape)
    
    return {
        'phi': stats['phi'],
        'Q_c': stats['Q_c'],
        'delta_S_b': stats['outgoing']['DeltaS_B'],
        'bias_in': stats['incoming']['bias'],
        'bias_out': stats['outgoing']['bias'],
        'entropy_in': stats['incoming']['entropy'],
        'entropy_out': stats['outgoing']['entropy'],
        'p0_in': stats['incoming']['p0'],
        'p1_in': stats['incoming']['p1'],
        'p0_out': stats['outgoing']['p0'],
        'p1_out': stats['outgoing']['p1'],
        'N': stats['N'],
        'K': stats['K'],
    }


def get_metric_value(result: dict, metric: str) -> float:
    """Extract a specific metric from simulation result."""
    metric_map = {
        'phi': 'phi',
        'Q_c': 'Q_c',
        'delta_S_b': 'delta_S_b',
        'bias': 'bias_out',
        'entropy_out': 'entropy_out',
    }
    return result[metric_map.get(metric, metric)]


def apply_param_to_config(config: SimulationRequest, param: str, value: float) -> SimulationRequest:
    """Create a modified config with a specific parameter value."""
    # Deep copy by recreating the models
    import copy
    new_config = SimulationRequest(
        phys_params=PhysParamsInput(**config.phys_params.model_dump()),
        tape_config=TapeConfigInput(**config.tape_config.model_dump()),
        demon_config=DemonConfigInput(**config.demon_config.model_dump()),
        tau=config.tau
    )
    
    # Apply the parameter change
    if param == 'tau':
        new_config.tau = value
    elif param == 'K':
        new_config.demon_config.K = int(value)
    elif param == 'n':
        new_config.demon_config.n = int(value)
    elif param == 'sigma':
        new_config.phys_params.sigma = value
    elif param == 'omega':
        new_config.phys_params.omega = value
    elif param == 'p0':
        new_config.tape_config.p0 = value
    elif param == 'DeltaE':
        new_config.phys_params.DeltaE = value
    elif param == 'gamma':
        new_config.phys_params.gamma = value
    elif param == 'Th':
        new_config.phys_params.Th = value
        new_config.phys_params.sigma = None
        new_config.phys_params.omega = None
    elif param == 'Tc':
        new_config.phys_params.Tc = value
        new_config.phys_params.sigma = None
        new_config.phys_params.omega = None
    elif param == 'correlation_strength':
        new_config.tape_config.correlation_strength = value
    else:
        raise ValueError(f"Unknown parameter: {param}")
    
    return new_config


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """API health check."""
    return {"status": "ok", "message": "Maxwell Demon Lab API is running"}


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "modules_loaded": True
    }


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate(request: SimulationRequest):
    """Run a single simulation with given parameters."""
    try:
        result = run_single_simulation(request)
        return SimulationResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sweep", response_model=SweepResult)
async def parameter_sweep(request: SweepRequest):
    """Run a 1D parameter sweep."""
    try:
        output_values = []
        
        for value in request.sweep_values:
            config = apply_param_to_config(request.base_config, request.sweep_param, value)
            result = run_single_simulation(config)
            output_values.append(get_metric_value(result, request.output_metric))
        
        return SweepResult(
            param_name=request.sweep_param,
            param_values=request.sweep_values,
            output_name=request.output_metric,
            output_values=output_values
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/phase-diagram", response_model=PhaseDiagramResult)
async def phase_diagram(request: PhaseDiagramRequest):
    """Generate a 2D phase diagram."""
    try:
        grid = []
        total = len(request.y_values) * len(request.x_values)
        count = 0
        
        for y_val in request.y_values:
            row = []
            for x_val in request.x_values:
                count += 1
                # Apply both parameters
                config = apply_param_to_config(request.base_config, request.x_param, x_val)
                config = apply_param_to_config(config, request.y_param, y_val)
                
                result = run_single_simulation(config)
                row.append(get_metric_value(result, request.output_metric))
            grid.append(row)
        
        return PhaseDiagramResult(
            x_param=request.x_param,
            y_param=request.y_param,
            x_values=request.x_values,
            y_values=request.y_values,
            output_name=request.output_metric,
            grid=grid
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/demon/validate")
async def validate_demon_config(config: DemonConfigInput):
    """Validate a demon configuration and return computed properties."""
    try:
        phys_params = PhysParams(sigma=0.3, omega=0.8, DeltaE=1.0, gamma=1.0)

        demon = Demon(
            n=config.n,
            phys_params=phys_params,
            init_state=config.init_state,
            energy_distribution=config.energy_distribution
        )
        
        return {
            "valid": True,
            "n_states": demon.n,
            "states": demon.states,
            "energy_values": demon.energy_values.tolist(),
            "delta_e_values": demon.delta_e_values.tolist(),
            "total_delta_e": demon.get_total_delta_e(),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.get("/api/parameters/info")
async def get_parameter_info():
    """Get information about all available parameters."""
    return {
        "sweep_parameters": {
            "tau": {"label": "τ (Interaction Time)", "min": 0.01, "max": 100, "default": 1.0, "step": 0.1},
            "K": {"label": "K (Stacked Demons)", "min": 1, "max": 50, "default": 1, "step": 1, "integer": True},
            "n": {"label": "n (Demon States)", "min": 2, "max": 100, "default": 2, "step": 1, "integer": True},
            "sigma": {"label": "σ (Intrinsic Rate)", "min": 0, "max": 1, "default": 0.3, "step": 0.01},
            "omega": {"label": "ω (Outgoing Rate)", "min": 0, "max": 1, "default": 0.8, "step": 0.01},
            "p0": {"label": "p₀ (Initial Probability)", "min": 0, "max": 1, "default": 0.5, "step": 0.01},
            "DeltaE": {"label": "ΔE (Energy Difference)", "min": 0.01, "max": 10, "default": 1.0, "step": 0.1},
            "gamma": {"label": "γ (Transition Rate)", "min": 0.01, "max": 10, "default": 1.0, "step": 0.1},
        },
        "output_metrics": {
            "phi": {"label": "φ (Bit Flip Fraction)", "description": "Fraction of bits flipped from 0 to 1"},
            "Q_c": {"label": "Q_c (Energy to Cold)", "description": "Total energy transferred to cold reservoir"},
            "delta_S_b": {"label": "ΔS_B (Entropy Change)", "description": "Change in bit entropy"},
            "bias": {"label": "Bias", "description": "Output bias (p₀ - p₁)"},
            "entropy_out": {"label": "S_out (Output Entropy)", "description": "Final tape entropy"},
        }
    }


if __name__ == "__main__":
 
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
