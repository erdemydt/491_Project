# Maxwell Demon Lab - Development Roadmap

## 🎯 Project Overview

A React-based frontend for the Maxwell's Demon simulation, enabling interactive parameter exploration, visualization, and data export without touching code.

---

## 📋 Phase 1: Foundation (Current Sprint)

### 1.1 Backend API Setup
- [x] Create Flask/FastAPI backend to expose simulation endpoints
- [x] Define API schema for:
  - Single simulation run
  - Phase diagram grid computation
  - Demon configuration

### 1.2 Frontend Architecture
- [x] Set up routing structure
- [x] Create shared layout with navigation
- [x] Install additional dependencies (if needed)
- [x] Set up Tailwind CSS for styling

### 1.3 Core Components
- [x] Parameter input forms with validation
- [x] Loading states and error handling
- [x] Responsive layout

---

## 📋 Phase 2: Single Simulation Page

### 2.1 Simulation Runner
- [ ] Form for core parameters:
  - `N` (tape length)
  - `p0` (initial probability)
  - `n` (demon states)
  - `K` (stacked demons)
  - `τ` (interaction time)
  - `σ`, `ω` (transition parameters)
- [ ] Run button → calls backend → displays results
- [ ] Real-time progress indicator

### 2.2 Results Display
- [ ] Key metrics cards (φ, Q_c, ΔS_B, bias)
- [ ] Tape visualization (before/after)
- [ ] Demon state evolution chart

---

## 📋 Phase 3: Phase Diagram Page

### 3.1 Parameter Sweep Form
- [ ] Select X-axis parameter (from: σ, ω, τ, K, n, p0)
- [ ] Select Y-axis parameter
- [ ] Select output metric (φ, Q_c, ΔS_B, bias)
- [ ] Define ranges and resolution

### 3.2 Heatmap Visualization
- [ ] 2D heatmap using recharts or custom canvas
- [ ] Color scale legend
- [ ] Hover tooltips showing exact values
- [ ] Click to run single simulation at that point

### 3.3 1D Parameter Sweeps
- [ ] Line plots for single parameter variation
- [ ] Multiple output metrics overlay

---

## 📋 Phase 4: Demon Designer

### 4.1 Visual State Editor
- [ ] Set number of states (n)
- [ ] Energy level diagram (draggable levels)
- [ ] Choose energy distribution: uniform, exponential, quadratic
- [ ] Auto-generate transition rates

### 4.2 Tape Configuration
- [ ] Tape type: standard, SmartTape
- [ ] Correlation settings (for SmartTape):
  - Type: none, markov, block, periodic
  - Correlation strength
  - Block size / period

### 4.3 Stacking Configuration
- [ ] Number of stacked demons (K)
- [ ] Per-state vs total ΔE mode
- [ ] Preserve mode: σ/ω or temperatures

---

## 📋 Phase 5: Export & Analysis

### 5.1 Data Export
- [ ] Download CSV for all computed data
- [ ] Download JSON for full simulation config + results
- [ ] Batch export for phase diagrams

### 5.2 Plot Export
- [ ] Download PNG (raster)
- [ ] Download SVG (vector)
- [ ] Customizable figure size

### 5.3 Session Management
- [ ] Save/load simulation configurations
- [ ] History of recent runs
- [ ] Compare multiple runs

---

## 📋 Phase 6: Advanced Features (Future)

### 6.1 Real-time Visualization
- [ ] Animated Gillespie evolution
- [ ] Step-by-step demon interaction

### 6.2 Competing Demons
- [ ] Multi-demon tape interaction
- [ ] Competing objectives visualization

### 6.3 Theory Overlay
- [ ] Analytical predictions
- [ ] Compare simulation vs theory

---

## 🏗️ Architecture

```
maxwell-demon-lab/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── forms/           # Input forms
│   │   ├── charts/          # Visualization components
│   │   └── layout/          # Navigation, containers
│   ├── pages/               # Route pages
│   │   ├── HomePage.jsx
│   │   ├── SimulationPage.jsx
│   │   ├── PhaseDiagramPage.jsx
│   │   └── DemonDesignerPage.jsx
│   ├── hooks/               # Custom React hooks
│   ├── services/            # API calls
│   ├── utils/               # Helpers, constants
│   └── App.jsx
├── api/                     # Python backend
│   ├── app.py               # FastAPI/Flask server
│   ├── simulation.py        # Simulation wrapper
│   └── requirements.txt
```

---

## 🎨 UI/UX Principles

1. **Progressive disclosure**: Start simple, reveal advanced options
2. **Immediate feedback**: Show loading states, validation errors
3. **Contextual help**: Tooltips explaining physics parameters
4. **Consistent layout**: Same structure across pages

---

## 🚀 Getting Started

```bash
# Frontend
cd maxwell-demon-lab
npm install
npm run dev

# Backend (in separate terminal)
cd api
pip install -r requirements.txt
python app.py
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/simulate` | POST | Run single simulation |
| `/api/phase-diagram` | POST | Generate 2D parameter sweep |
| `/api/sweep` | POST | Generate 1D parameter sweep |
| `/api/demon/validate` | POST | Validate demon configuration |

---

*Last updated: November 2025*
