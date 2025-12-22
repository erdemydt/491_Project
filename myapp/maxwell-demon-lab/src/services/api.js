/**
 * API service for Maxwell Demon Lab
 * Handles all communication with the Python backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
api.interceptors.request.use((config) => {
  console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[API Error]', error.response?.data || error.message);
    throw error;
  }
);

/**
 * Check API health
 */
export const checkHealth = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

/**
 * Run a single simulation
 * @param {Object} config - Simulation configuration
 */
export const runSimulation = async (config) => {
  const response = await api.post('/api/simulate', config);
  return response.data;
};

/**
 * Run a 1D parameter sweep
 * @param {Object} sweepConfig - Sweep configuration
 */
export const runParameterSweep = async (sweepConfig) => {
  const response = await api.post('/api/sweep', sweepConfig);
  return response.data;
};

/**
 * Generate a 2D phase diagram
 * @param {Object} diagramConfig - Phase diagram configuration
 */
export const generatePhaseDiagram = async (diagramConfig) => {
  const response = await api.post('/api/phase-diagram', diagramConfig);
  return response.data;
};

/**
 * Validate demon configuration
 * @param {Object} demonConfig - Demon configuration
 */
export const validateDemon = async (demonConfig) => {
  const response = await api.post('/api/demon/validate', demonConfig);
  return response.data;
};

/**
 * Get parameter info
 */
export const getParameterInfo = async () => {
  const response = await api.get('/api/parameters/info');
  return response.data;
};

export default api;
