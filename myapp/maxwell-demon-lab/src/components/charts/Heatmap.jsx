import { useMemo, useRef } from 'react';

/**
 * Custom 2D Heatmap for phase diagrams
 * Uses canvas for performance with large grids
 */
export function Heatmap({
  grid,           // 2D array [y][x] of values
  xValues,        // X-axis values
  yValues,        // Y-axis values
  xLabel,
  yLabel,
  colorLabel,
  title = '',
  colorScale = 'viridis',
  onCellClick,    // Optional click handler (x, y, value)
}) {
  const canvasRef = useRef(null);
  
  // Calculate value range
  const { minVal, maxVal } = useMemo(() => {
    let min = Infinity, max = -Infinity;
    for (const row of grid) {
      for (const val of row) {
        if (val < min) min = val;
        if (val > max) max = val;
      }
    }
    return { minVal: min, maxVal: max };
  }, [grid]);

  // Color scale functions
  const getColor = (value) => {
    const t = maxVal > minVal ? (value - minVal) / (maxVal - minVal) : 0.5;
    
    if (colorScale === 'viridis') {
      // Approximate viridis colormap
      const r = Math.round(68 + t * (253 - 68));
      const g = Math.round(1 + t * (231 - 1) * Math.sin(t * Math.PI * 0.5 + 0.3));
      const b = Math.round(84 + (1 - t) * (150 - 84));
      return `rgb(${r}, ${g}, ${b})`;
    } else if (colorScale === 'coolwarm') {
      // Blue to red
      const r = Math.round(59 + t * (180 - 59));
      const g = Math.round(76 + Math.abs(t - 0.5) * -100);
      const b = Math.round(192 - t * (192 - 77));
      return `rgb(${r}, ${g}, ${b})`;
    } else if (colorScale === 'plasma') {
      const r = Math.round(13 + t * (240 - 13));
      const g = Math.round(8 + t * (249 - 8) * t);
      const b = Math.round(135 - t * (135 - 33));
      return `rgb(${r}, ${g}, ${b})`;
    }
    
    // Default grayscale
    const gray = Math.round(t * 255);
    return `rgb(${gray}, ${gray}, ${gray})`;
  };

  // Cell size based on grid dimensions
  const cellWidth = Math.max(10, Math.floor(500 / xValues.length));
  const cellHeight = Math.max(10, Math.floor(400 / yValues.length));
  
  const width = cellWidth * xValues.length;
  const height = cellHeight * yValues.length;

  // Handle cell click
  const handleCanvasClick = (e) => {
    if (!onCellClick) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / cellWidth);
    const y = Math.floor((e.clientY - rect.top) / cellHeight);
    
    if (x >= 0 && x < xValues.length && y >= 0 && y < yValues.length) {
      onCellClick(xValues[x], yValues[yValues.length - 1 - y], grid[yValues.length - 1 - y][x]);
    }
  };

  return (
    <div className="bg-gray-900 rounded-xl p-4">
      {title && (
        <h3 className="text-lg font-semibold text-white mb-4 text-center">{title}</h3>
      )}
      
      <div className="flex">
        {/* Y-axis label */}
        <div className="flex items-center justify-center w-8">
          <span 
            className="text-gray-400 text-sm transform -rotate-90 whitespace-nowrap"
            style={{ transformOrigin: 'center' }}
          >
            {yLabel}
          </span>
        </div>
        
        {/* Y-axis ticks */}
        <div className="flex flex-col justify-between pr-2" style={{ height }}>
          {yValues.slice().reverse().filter((_, i) => i % Math.ceil(yValues.length / 5) === 0).map((val) => (
            <span key={val} className="text-xs text-gray-400">
              {typeof val === 'number' ? val.toFixed(2) : val}
            </span>
          ))}
        </div>
        
        {/* Heatmap grid */}
        <div className="flex flex-col">
          <div 
            className="grid border border-gray-700 cursor-pointer"
            style={{ 
              gridTemplateColumns: `repeat(${xValues.length}, ${cellWidth}px)`,
              gridTemplateRows: `repeat(${yValues.length}, ${cellHeight}px)`,
            }}
            onClick={handleCanvasClick}
            ref={canvasRef}
          >
            {yValues.slice().reverse().map((yVal, yi) => (
              xValues.map((xVal, xi) => {
                const value = grid[yValues.length - 1 - yi][xi];
                return (
                  <div
                    key={`${xi}-${yi}`}
                    className="transition-opacity hover:opacity-80"
                    style={{ 
                      backgroundColor: getColor(value),
                      width: cellWidth,
                      height: cellHeight,
                    }}
                    title={`${xLabel}: ${xVal.toFixed(3)}\n${yLabel}: ${yVal.toFixed(3)}\n${colorLabel}: ${value.toFixed(4)}`}
                  />
                );
              })
            ))}
          </div>
          
          {/* X-axis ticks */}
          <div className="flex justify-between mt-2" style={{ width }}>
            {xValues.filter((_, i) => i % Math.ceil(xValues.length / 5) === 0).map((val) => (
              <span key={val} className="text-xs text-gray-400">
                {typeof val === 'number' ? val.toFixed(2) : val}
              </span>
            ))}
          </div>
          
          {/* X-axis label */}
          <div className="text-center mt-2">
            <span className="text-gray-400 text-sm">{xLabel}</span>
          </div>
        </div>
        
        {/* Color bar */}
        <div className="flex flex-col ml-4" style={{ height }}>
          <div 
            className="w-4 flex-1 rounded"
            style={{
              background: `linear-gradient(to bottom, ${getColor(maxVal)}, ${getColor((maxVal + minVal) / 2)}, ${getColor(minVal)})`
            }}
          />
          <div className="flex flex-col justify-between h-full absolute right-0 ml-6">
            <span className="text-xs text-gray-400">{maxVal.toFixed(3)}</span>
            <span className="text-xs text-gray-400">{((maxVal + minVal) / 2).toFixed(3)}</span>
            <span className="text-xs text-gray-400">{minVal.toFixed(3)}</span>
          </div>
        </div>
        
        {/* Color label */}
        <div className="flex items-center ml-8">
          <span className="text-gray-400 text-sm">{colorLabel}</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Simple color bar legend
 */
export function ColorBar({ min, max, label, colorScale = 'viridis' }) {
  const getColor = (t) => {
    if (colorScale === 'viridis') {
      const r = Math.round(68 + t * (253 - 68));
      const g = Math.round(1 + t * (231 - 1) * Math.sin(t * Math.PI * 0.5 + 0.3));
      const b = Math.round(84 + (1 - t) * (150 - 84));
      return `rgb(${r}, ${g}, ${b})`;
    }
    const gray = Math.round(t * 255);
    return `rgb(${gray}, ${gray}, ${gray})`;
  };

  return (
    <div className="flex items-center space-x-2">
      <span className="text-xs text-gray-400">{min.toFixed(2)}</span>
      <div 
        className="w-32 h-4 rounded"
        style={{
          background: `linear-gradient(to right, ${getColor(0)}, ${getColor(0.5)}, ${getColor(1)})`
        }}
      />
      <span className="text-xs text-gray-400">{max.toFixed(2)}</span>
      <span className="text-xs text-gray-500 ml-2">{label}</span>
    </div>
  );
}
