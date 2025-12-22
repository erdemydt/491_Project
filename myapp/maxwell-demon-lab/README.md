# 🔬 Maxwell Demon Lab

Interactive React frontend for Maxwell's Demon physics simulations.

## Features

- **Single Simulation**: Run individual simulations with custom parameters
- **Parameter Sweeps**: Explore 1D parameter variations
- **Phase Diagrams**: Generate 2D heatmaps of parameter spaces
- **Demon Designer**: Visual configuration of demon energy levels
- **Export**: Download CSV data and charts

## Quick Start

### Backend (Python API)

```bash
# Navigate to API directory
cd api

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# API runs at http://localhost:8000
```

### Frontend (React)

```bash
# In the maxwell-demon-lab directory
npm install
npm run dev
# App runs at http://localhost:5173
```

## Project Structure

```
maxwell-demon-lab/
├── api/                    # Python FastAPI backend
│   ├── app.py              # Main API server
│   └── requirements.txt    # Python dependencies
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── charts/         # Visualization components
│   │   ├── forms/          # Input components
│   │   └── layout/         # Navigation, layout
│   ├── pages/              # Route pages
│   ├── services/           # API client
│   └── utils/              # Constants, helpers
└── ROADMAP.md              # Development roadmap
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/simulate` | POST | Run single simulation |
| `/api/sweep` | POST | 1D parameter sweep |
| `/api/phase-diagram` | POST | 2D phase diagram |
| `/api/demon/validate` | POST | Validate demon config |

## Tech Stack

- **Frontend**: React 19, Vite, TailwindCSS, Recharts
- **Backend**: FastAPI, NumPy, Pydantic
- **Simulation**: Custom Gillespie algorithm implementation
