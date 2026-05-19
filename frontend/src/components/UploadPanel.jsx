import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_BASE
  ? `${process.env.REACT_APP_API_BASE}/api`
  : '/api';

export default function UploadPanel({ onResult }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filename, setFilename] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState(0);
  const [method, setMethod] = useState('default');
  const inputRef = useRef();

  const processFile = async (file) => {
    if (!file) return;

    const allowed = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      setError(`Unsupported file type: ${ext}`);
      return;
    }

    setFilename(file.name);
    setLoading(true);
    setError(null);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    const params = method === 'morph' ? '?method=morph' : '';

    try {
      const res = await axios.post(`${API_URL}/upload${params}`, formData, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded / e.total) * 50));
        },
      });
      setProgress(100);
      onResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Check backend is running.');
    } finally {
      setLoading(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  const handleUpload = (e) => processFile(e.target.files[0]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0]);
  }, []);

  return (
    <div
      className={`upload-panel ${dragActive ? 'drag-active' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp"
        onChange={handleUpload}
        disabled={loading}
        id="file-input"
      />

      {!loading ? (
        <>
          <div className="upload-icon">📐</div>
          <div className="method-toggle">
            <label>
              <input
                type="radio"
                name="method"
                value="default"
                checked={method === 'default'}
                onChange={() => setMethod('default')}
              />
              OCR + Lines (Full)
            </label>
            <label>
              <input
                type="radio"
                name="method"
                value="morph"
                checked={method === 'morph'}
                onChange={() => setMethod('morph')}
              />
              Morphological Only
            </label>
          </div>
          <label htmlFor="file-input" className="upload-label">
            Choose HVAC Drawing
          </label>
          <p className="upload-hint">or drag & drop a PDF / Image here</p>
          {filename && <div className="upload-filename">Last: {filename}</div>}
        </>
      ) : (
        <div className="upload-progress">
          <div className="spinner" />
          <p>Processing <strong>{filename}</strong>...</p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="progress-text">
            {progress < 50 ? 'Uploading...' : method === 'morph' ? 'Detecting ducts (morphological)...' : 'Detecting ducts & running OCR...'}
          </p>
        </div>
      )}

      {error && <div className="upload-error">❌ {error}</div>}
    </div>
  );
}
