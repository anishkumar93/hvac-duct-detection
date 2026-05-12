import React, { useState, useCallback } from 'react';
import UploadPanel from './components/UploadPanel';
import DrawingCanvas from './components/DrawingCanvas';
import DuctSchedule from './components/DuctSchedule';
import './App.css';

function App() {
  const [result, setResult] = useState(null);
  const [selectedDuct, setSelectedDuct] = useState(null);
  const [activeTab, setActiveTab] = useState('canvas');

  const handleDuctClick = useCallback((duct) => {
    setSelectedDuct((prev) => (prev?.id === duct.id ? null : duct));
  }, []);

  const handleResult = (r) => {
    setResult(r);
    setSelectedDuct(null);
    setActiveTab('canvas');
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <h1>🔧 HVAC Duct Detection</h1>
            <p className="subtitle">Detect & annotate ductwork from mechanical drawings</p>
          </div>
          {result && (
            <div className="header-actions">
              <a
                href={`${process.env.REACT_APP_API_BASE || ''}${result.annotated_image_path}`}
                target="_blank"
                rel="noopener noreferrer"
                className="export-btn"
              >
                ⬇ Export PNG
              </a>
            </div>
          )}
        </div>
      </header>

      <div className="app-content">
        <UploadPanel onResult={handleResult} />

        {result && (
          <>
            {/* Stats */}
            <div className="stats-bar">
              <div className="stat-item">
                <span className="stat-value">{result.ducts.length}</span>
                <span className="stat-label">Ducts</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{result.ducts.filter(d => d.dimension).length}</span>
                <span className="stat-label">With Dimensions</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{result.image_width}×{result.image_height}</span>
                <span className="stat-label">Resolution</span>
              </div>
              {result.scale && (
                <div className="stat-item">
                  <span className="stat-value">{result.scale}</span>
                  <span className="stat-label">Scale</span>
                </div>
              )}
            </div>

            {/* Tabs */}
            <div className="tab-bar">
              <button
                className={`tab ${activeTab === 'canvas' ? 'active' : ''}`}
                onClick={() => setActiveTab('canvas')}
              >
                🖼 Drawing
              </button>
              <button
                className={`tab ${activeTab === 'schedule' ? 'active' : ''}`}
                onClick={() => setActiveTab('schedule')}
              >
                📋 Schedule
              </button>
            </div>

            {/* Content */}
            {activeTab === 'canvas' && (
              <div className="main-layout">
                <div className="canvas-section">
                  <DrawingCanvas
                    result={result}
                    onDuctClick={handleDuctClick}
                    selectedDuct={selectedDuct}
                  />
                </div>

                {selectedDuct && (
                  <div className="detail-panel">
                    <div className="detail-header">
                      <h3>Duct #{selectedDuct.id}</h3>
                      <button className="close-btn" onClick={() => setSelectedDuct(null)}>✕</button>
                    </div>
                    <div className="detail-body">
                      <DetailRow label="Dimension" value={selectedDuct.dimension} />
                      <DetailRow label="Type" value={selectedDuct.duct_type} />
                      <DetailRow
                        label="Pressure"
                        value={
                          <span className={`pressure-badge ${selectedDuct.pressure_class.split(' ')[0].toLowerCase()}`}>
                            {selectedDuct.pressure_class}
                          </span>
                        }
                      />
                      <DetailRow label="Length" value={selectedDuct.length} />
                      <DetailRow label="Confidence" value={`${(selectedDuct.confidence * 100).toFixed(0)}%`} />
                      <DetailRow label="Position" value={`(${Math.round(selectedDuct.bbox.x)}, ${Math.round(selectedDuct.bbox.y)})`} />
                      <DetailRow label="Size" value={`${Math.round(selectedDuct.bbox.width)} × ${Math.round(selectedDuct.bbox.height)} px`} />
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'schedule' && (
              <DuctSchedule
                ducts={result.ducts}
                onRowClick={handleDuctClick}
                selectedDuct={selectedDuct}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="detail-row">
      <label>{label}</label>
      <span>{value || '—'}</span>
    </div>
  );
}

export default App;
