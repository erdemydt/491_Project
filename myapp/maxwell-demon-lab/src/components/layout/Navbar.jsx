import { Link, useLocation } from 'react-router-dom';
import classNames from 'classnames';

const navItems = [
  { path: '/', label: 'Home', icon: '🏠' },
  { path: '/simulation', label: 'Simulation', icon: '⚡' },
  { path: '/phase-diagram', label: 'Phase Diagram', icon: '📊' },
  { path: '/demon-designer', label: 'Demon Designer', icon: '😈' },
];

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="bg-gray-900 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <span className="text-2xl">🔬</span>
            <span className="text-white font-bold text-lg">Maxwell Demon Lab</span>
          </Link>

          {/* Navigation Links */}
          <div className="flex space-x-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={classNames(
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  location.pathname === item.path
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                )}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>

          {/* API Status Indicator */}
          <div className="flex items-center space-x-2">
            <span className="inline-block w-2 h-2 rounded-full bg-green-500"></span>
            <span className="text-gray-400 text-xs">API Connected</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
