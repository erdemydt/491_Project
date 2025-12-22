import { 
  LineChart as RechartsLineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

/**
 * Line chart for 1D parameter sweeps
 */
export function SweepChart({ 
  data, 
  xKey, 
  yKey, 
  xLabel, 
  yLabel,
  color = '#3b82f6',
  referenceLines = [],
  title = ''
}) {
  return (
    <div className="bg-gray-900 rounded-xl p-4">
      {title && (
        <h3 className="text-lg font-semibold text-white mb-4 text-center">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={400}>
        <RechartsLineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey={xKey} 
            stroke="#9ca3af"
            label={{ value: xLabel, position: 'bottom', fill: '#9ca3af' }}
          />
          <YAxis 
            stroke="#9ca3af"
            label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1f2937', 
              border: '1px solid #374151',
              borderRadius: '8px'
            }}
            labelStyle={{ color: '#fff' }}
          />
          <Legend />
          {referenceLines.map((ref, i) => (
            <ReferenceLine 
              key={i} 
              y={ref.value} 
              stroke={ref.color || '#6b7280'} 
              strokeDasharray="5 5"
              label={{ value: ref.label, fill: ref.color || '#6b7280' }}
            />
          ))}
          <Line 
            type="monotone" 
            dataKey={yKey} 
            stroke={color} 
            strokeWidth={2}
            dot={{ fill: color, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
          />
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Multi-line chart for comparing multiple metrics
 */
export function MultiLineChart({ 
  data, 
  xKey, 
  lines,  // Array of { key, label, color }
  xLabel,
  title = ''
}) {
  return (
    <div className="bg-gray-900 rounded-xl p-4">
      {title && (
        <h3 className="text-lg font-semibold text-white mb-4 text-center">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={400}>
        <RechartsLineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey={xKey} 
            stroke="#9ca3af"
            label={{ value: xLabel, position: 'bottom', fill: '#9ca3af' }}
          />
          <YAxis stroke="#9ca3af" />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1f2937', 
              border: '1px solid #374151',
              borderRadius: '8px'
            }}
            labelStyle={{ color: '#fff' }}
          />
          <Legend />
          {lines.map((line) => (
            <Line 
              key={line.key}
              type="monotone" 
              dataKey={line.key} 
              name={line.label}
              stroke={line.color} 
              strokeWidth={2}
              dot={{ fill: line.color, strokeWidth: 2, r: 3 }}
            />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}
