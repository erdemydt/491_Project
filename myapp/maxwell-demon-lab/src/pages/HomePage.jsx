import { Link } from 'react-router-dom';

const features = [
  {
    title: 'Single Simulation',
    description: 'Run individual simulations with custom parameters and visualize results instantly.',
    icon: '⚡',
    link: '/simulation',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    title: 'Phase Diagrams',
    description: 'Explore 2D parameter spaces with heatmaps. See how φ, Q_c, and entropy change.',
    icon: '📊',
    link: '/phase-diagram',
    color: 'from-purple-500 to-pink-500',
  },
  {
    title: 'Demon Designer',
    description: 'Design custom demons with specific energy levels and transition rates.',
    icon: '😈',
    link: '/demon-designer',
    color: 'from-orange-500 to-red-500',
  },
];

const quickStats = [
  { label: 'Demon States', value: '2-100', icon: '🔢' },
  { label: 'Stacked Demons', value: '1-50', icon: '📚' },
  { label: 'Energy Distributions', value: '3 types', icon: '📈' },
  { label: 'Tape Correlations', value: '4 types', icon: '🔗' },
];

export default function HomePage() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center py-12 bg-gradient-to-r from-gray-900 to-gray-800 rounded-2xl">
        <h1 className="text-4xl font-bold text-white mb-4">
          🔬 Maxwell Demon Lab
        </h1>
        <p className="text-xl text-gray-300 max-w-2xl mx-auto">
          Interactive simulation suite for exploring Maxwell's Demon physics.
          Run experiments, generate phase diagrams, and analyze thermodynamic phenomena.
        </p>
        
        <div className="mt-8 flex justify-center space-x-4">
          <Link
            to="/simulation"
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
          >
            Start Simulation →
          </Link>
          <Link
            to="/phase-diagram"
            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-medium rounded-lg transition-colors"
          >
            View Phase Diagrams
          </Link>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="grid md:grid-cols-3 gap-6">
        {features.map((feature) => (
          <Link
            key={feature.link}
            to={feature.link}
            className="group relative bg-gray-900 rounded-xl p-6 hover:bg-gray-800 transition-all duration-300 overflow-hidden"
          >
            <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
            <div className="relative">
              <span className="text-4xl mb-4 block">{feature.icon}</span>
              <h3 className="text-xl font-semibold text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-400">
                {feature.description}
              </p>
              <span className="mt-4 inline-block text-blue-400 group-hover:text-blue-300">
                Explore →
              </span>
            </div>
          </Link>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-xl font-semibold text-white mb-6">Simulation Capabilities</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {quickStats.map((stat) => (
            <div key={stat.label} className="bg-gray-800 rounded-lg p-4 text-center">
              <span className="text-2xl block mb-2">{stat.icon}</span>
              <div className="text-2xl font-bold text-blue-400">{stat.value}</div>
              <div className="text-sm text-gray-400">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Physics Overview */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Physics Background</h2>
        <div className="grid md:grid-cols-2 gap-6 text-gray-300">
          <div>
            <h3 className="font-medium text-white mb-2">The Maxwell's Demon Model</h3>
            <p className="text-sm">
              A demon operates on a tape of bits, interacting with each bit for time τ.
              The demon can be in n states with energy levels E₀ &lt; E₁ &lt; ... &lt; Eₙ₋₁.
              Transitions occur stochastically via the Gillespie algorithm.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-white mb-2">Key Parameters</h3>
            <ul className="text-sm space-y-1">
              <li><span className="text-blue-400">σ</span> - Intrinsic transition rate (hot reservoir)</li>
              <li><span className="text-blue-400">ω</span> - Outgoing transition rate (cold reservoir)</li>
              <li><span className="text-blue-400">τ</span> - Interaction time per bit</li>
              <li><span className="text-blue-400">K</span> - Number of stacked demons</li>
            </ul>
          </div>
        </div>
      </div>

      {/* API Status */}
      <div className="bg-gray-900 rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
          <span className="text-gray-300">Backend API</span>
        </div>
        <span className="text-gray-500 text-sm">http://localhost:8000</span>
      </div>
    </div>
  );
}
