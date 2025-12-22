import classNames from 'classnames';

/**
 * Reusable number input with label and optional tooltip
 */
export function NumberInput({ 
  label, 
  value, 
  onChange, 
  min, 
  max, 
  step = 1,
  tooltip,
  disabled = false,
  className = '',
  error = null,
}) {
  return (
    <div className={classNames('mb-3', className)}>
      <label className="block text-sm font-medium text-gray-300 mb-1">
        {label}
        {tooltip && (
          <span className="ml-1 text-gray-500 cursor-help" title={tooltip}>
            ⓘ
          </span>
        )}
      </label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className={classNames(
          'w-full px-3 py-2 bg-gray-800 border rounded-lg text-white',
          'focus:outline-none focus:ring-2 focus:ring-blue-500',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          error ? 'border-red-500' : 'border-gray-700'
        )}
      />
      {error && <p className="mt-1 text-xs text-red-400">{error}</p>}
    </div>
  );
}

/**
 * Reusable slider input with value display
 */
export function SliderInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01,
  tooltip,
  disabled = false,
  showValue = true,
  className = '',
}) {
  return (
    <div className={classNames('mb-3', className)}>
      <div className="flex justify-between items-center mb-1">
        <label className="text-sm font-medium text-gray-300">
          {label}
          {tooltip && (
            <span className="ml-1 text-gray-500 cursor-help" title={tooltip}>
              ⓘ
            </span>
          )}
        </label>
        {showValue && (
          <span className="text-sm text-blue-400 font-mono">{value}</span>
        )}
      </div>
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
      />
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

/**
 * Reusable select dropdown
 */
export function SelectInput({
  label,
  value,
  onChange,
  options,
  tooltip,
  disabled = false,
  className = '',
}) {
  return (
    <div className={classNames('mb-3', className)}>
      <label className="block text-sm font-medium text-gray-300 mb-1">
        {label}
        {tooltip && (
          <span className="ml-1 text-gray-500 cursor-help" title={tooltip}>
            ⓘ
          </span>
        )}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={classNames(
          'w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white',
          'focus:outline-none focus:ring-2 focus:ring-blue-500',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

/**
 * Reusable checkbox
 */
export function CheckboxInput({
  label,
  checked,
  onChange,
  tooltip,
  disabled = false,
  className = '',
}) {
  return (
    <div className={classNames('mb-3 flex items-center', className)}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="w-4 h-4 text-blue-500 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
      />
      <label className="ml-2 text-sm font-medium text-gray-300">
        {label}
        {tooltip && (
          <span className="ml-1 text-gray-500 cursor-help" title={tooltip}>
            ⓘ
          </span>
        )}
      </label>
    </div>
  );
}

/**
 * Form section with title
 */
export function FormSection({ title, children, className = '' }) {
  return (
    <div className={classNames('bg-gray-900 rounded-xl p-4 mb-4', className)}>
      {title && (
        <h3 className="text-lg font-semibold text-white mb-4 border-b border-gray-700 pb-2">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

/**
 * Primary button
 */
export function Button({
  children,
  onClick,
  disabled = false,
  loading = false,
  variant = 'primary',
  className = '',
  type = 'button',
}) {
  const variants = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white',
    secondary: 'bg-gray-700 hover:bg-gray-600 text-white',
    danger: 'bg-red-600 hover:bg-red-700 text-white',
    success: 'bg-green-600 hover:bg-green-700 text-white',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={classNames(
        'px-4 py-2 rounded-lg font-medium transition-colors',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900',
        variants[variant],
        className
      )}
    >
      {loading ? (
        <span className="flex items-center">
          <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Running...
        </span>
      ) : children}
    </button>
  );
}
