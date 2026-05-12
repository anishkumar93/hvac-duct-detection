import React, { useState, useRef, useEffect, useCallback } from 'react';

const API_BASE = process.env.REACT_APP_API_BASE || '';

const PRESSURE_COLORS = {
  'Low Pressure': 'rgba(0, 150, 255, 0.2)',
  'Medium Pressure': 'rgba(255, 165, 0, 0.2)',
  'High Pressure': 'rgba(255, 50, 50, 0.2)',
  'Unknown': 'rgba(0, 200, 0, 0.2)',
};

const PRESSURE_STROKES = {
  'Low Pressure': '#2b6cb0',
  'Medium Pressure': '#d69e2e',
  'High Pressure': '#e53e3e',
  'Unknown': '#38a169',
};

export default function DrawingCanvas({ result, onDuctClick, selectedDuct }) {
  const viewportRef = useRef(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [imgNatural, setImgNatural] = useState({ width: 0, height: 0 });
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [hoveredDuct, setHoveredDuct] = useState(null);

  // Reset when new image loads
  useEffect(() => {
    setTransform({ x: 0, y: 0, scale: 1 });
  }, [result?.annotated_image_path]);

  const handleImageLoad = (e) => {
    const img = e.target;
    setImgNatural({ width: img.naturalWidth, height: img.naturalHeight });
    // Fit to viewport
    const vp = viewportRef.current;
    if (vp) {
      const fitScale = Math.min(vp.clientWidth / img.naturalWidth, vp.clientHeight / img.naturalHeight, 1);
      setTransform({ x: 0, y: 0, scale: fitScale });
    }
  };

  // Zoom centered on cursor
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const vp = viewportRef.current;
    if (!vp) return;

    const rect = vp.getBoundingClientRect();
    // Mouse position relative to viewport
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;

    setTransform((prev) => {
      const newScale = Math.min(Math.max(prev.scale * zoomFactor, 0.1), 8);
      const ratio = newScale / prev.scale;
      // Adjust pan so the point under cursor stays fixed
      const newX = mx - ratio * (mx - prev.x);
      const newY = my - ratio * (my - prev.y);
      return { x: newX, y: newY, scale: newScale };
    });
  }, []);

  // Pan with mouse drag
  const handleMouseDown = (e) => {
    // Left click = pan (unless clicking on a duct overlay)
    if (e.button === 0) {
      setIsPanning(true);
      setLastMouse({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    }
  };

  const handleMouseMove = (e) => {
    if (!isPanning) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    setLastMouse({ x: e.clientX, y: e.clientY });
    setTransform((prev) => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
  };

  const handleMouseUp = () => setIsPanning(false);

  const resetView = () => {
    const vp = viewportRef.current;
    if (vp && imgNatural.width) {
      const fitScale = Math.min(vp.clientWidth / imgNatural.width, vp.clientHeight / imgNatural.height, 1);
      setTransform({ x: 0, y: 0, scale: fitScale });
    }
  };

  const zoomIn = () => {
    const vp = viewportRef.current;
    if (!vp) return;
    const cx = vp.clientWidth / 2;
    const cy = vp.clientHeight / 2;
    setTransform((prev) => {
      const newScale = Math.min(prev.scale * 1.3, 8);
      const ratio = newScale / prev.scale;
      return { x: cx - ratio * (cx - prev.x), y: cy - ratio * (cy - prev.y), scale: newScale };
    });
  };

  const zoomOut = () => {
    const vp = viewportRef.current;
    if (!vp) return;
    const cx = vp.clientWidth / 2;
    const cy = vp.clientHeight / 2;
    setTransform((prev) => {
      const newScale = Math.max(prev.scale / 1.3, 0.1);
      const ratio = newScale / prev.scale;
      return { x: cx - ratio * (cx - prev.x), y: cy - ratio * (cy - prev.y), scale: newScale };
    });
  };

  if (!result?.annotated_image_path) return null;

  const displayWidth = imgNatural.width * transform.scale;
  const displayHeight = imgNatural.height * transform.scale;

  return (
    <div className="canvas-wrapper">
      {/* Toolbar */}
      <div className="canvas-toolbar">
        <div className="toolbar-group">
          <button onClick={zoomOut} title="Zoom Out">−</button>
          <span className="zoom-level">{Math.round(transform.scale * 100)}%</span>
          <button onClick={zoomIn} title="Zoom In">+</button>
          <button onClick={resetView} title="Fit to View">⟲</button>
        </div>
        <div className="toolbar-group">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showAnnotations}
              onChange={(e) => setShowAnnotations(e.target.checked)}
            />
            <span>Annotations</span>
          </label>
        </div>
        <div className="toolbar-hint">
          Drag to pan • Scroll to zoom
        </div>
      </div>

      {/* Viewport */}
      <div
        ref={viewportRef}
        className="canvas-viewport"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
      >
        <div
          style={{
            position: 'absolute',
            left: transform.x,
            top: transform.y,
            width: displayWidth,
            height: displayHeight,
          }}
        >
          <img
            src={`${API_BASE}${result.annotated_image_path}`}
            alt="Annotated HVAC Drawing"
            onLoad={handleImageLoad}
            style={{ width: '100%', height: '100%', display: 'block', pointerEvents: 'none' }}
            crossOrigin="anonymous"
            draggable={false}
          />

          {/* SVG overlay */}
          {showAnnotations && imgNatural.width > 0 && (
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
              viewBox={`0 0 ${result.image_width} ${result.image_height}`}
              preserveAspectRatio="none"
            >
              {result.ducts.map((duct) => {
                const isSelected = selectedDuct?.id === duct.id;
                const isHovered = hoveredDuct === duct.id;
                const cx = duct.bbox.x;
                const cy = duct.bbox.y;
                const w = duct.bbox.width;
                const h = duct.bbox.height;

                return (
                  <g key={duct.id}>
                    <rect
                      x={cx - w / 2}
                      y={cy - h / 2}
                      width={w}
                      height={h}
                      transform={`rotate(${duct.bbox.angle}, ${cx}, ${cy})`}
                      fill={
                        isSelected
                          ? 'rgba(255,255,0,0.35)'
                          : isHovered
                          ? 'rgba(255,255,255,0.25)'
                          : PRESSURE_COLORS[duct.pressure_class]
                      }
                      stroke={isSelected ? '#ffd700' : isHovered ? '#fff' : PRESSURE_STROKES[duct.pressure_class]}
                      strokeWidth={isSelected ? 5 : isHovered ? 3 : 2}
                      style={{ pointerEvents: 'all', cursor: 'pointer' }}
                      onClick={(e) => { e.stopPropagation(); onDuctClick(duct); }}
                      onMouseEnter={() => setHoveredDuct(duct.id)}
                      onMouseLeave={() => setHoveredDuct(null)}
                    />
                    {(isSelected || isHovered) && (
                      <text
                        x={cx}
                        y={cy - h / 2 - 20}
                        textAnchor="middle"
                        fill={isSelected ? '#ffd700' : '#fff'}
                        fontSize="28"
                        fontWeight="bold"
                        stroke="#000"
                        strokeWidth="1"
                        style={{ pointerEvents: 'none' }}
                      >
                        #{duct.id} {duct.dimension || ''}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}
