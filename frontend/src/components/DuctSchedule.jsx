import React, { useState, useMemo } from 'react';

const PRESSURE_ORDER = { 'High Pressure': 0, 'Medium Pressure': 1, 'Low Pressure': 2, 'Unknown': 3 };

export default function DuctSchedule({ ducts, onRowClick, selectedDuct }) {
  const [sortKey, setSortKey] = useState('pressure_class');
  const [sortAsc, setSortAsc] = useState(true);
  const [filterPressure, setFilterPressure] = useState('All');

  const showTypeColumn = useMemo(() => ducts.some((d) => d.duct_type !== 'Unknown'), [ducts]);

  const pressureClasses = useMemo(() => {
    const classes = new Set(ducts.map((d) => d.pressure_class));
    return ['All', ...classes];
  }, [ducts]);

  const sorted = useMemo(() => {
    let filtered = ducts;
    if (filterPressure !== 'All') {
      filtered = ducts.filter((d) => d.pressure_class === filterPressure);
    }
    return [...filtered].sort((a, b) => {
      if (sortKey === 'pressure_class') {
        const diff = PRESSURE_ORDER[a.pressure_class] - PRESSURE_ORDER[b.pressure_class];
        return sortAsc ? diff : -diff;
      }
      let av = a[sortKey], bv = b[sortKey];
      if (av == null) av = '';
      if (bv == null) bv = '';
      if (typeof av === 'number') return sortAsc ? av - bv : bv - av;
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
  }, [ducts, sortKey, sortAsc, filterPressure]);

  const handleSort = (key) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  };

  const SortIcon = ({ col }) => {
    if (sortKey !== col) return <span className="sort-icon">⇅</span>;
    return <span className="sort-icon active">{sortAsc ? '↑' : '↓'}</span>;
  };

  if (!ducts?.length) return null;

  const withDimension = ducts.filter((d) => d.dimension).length;

  return (
    <div className="schedule-section">
      <div className="schedule-header">
        <h3>Duct Schedule</h3>
        <div className="schedule-meta">
          <span>{ducts.length} segments</span>
          <span className="meta-sep">•</span>
          <span>{withDimension} with dimensions</span>
        </div>
        <div className="schedule-filters">
          <select
            value={filterPressure}
            onChange={(e) => setFilterPressure(e.target.value)}
          >
            {pressureClasses.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="schedule-table-wrap">
        <table className="schedule-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('id')}># <SortIcon col="id" /></th>
              {showTypeColumn && <th onClick={() => handleSort('duct_type')}>Type <SortIcon col="duct_type" /></th>}
              <th onClick={() => handleSort('dimension')}>Dimension <SortIcon col="dimension" /></th>
              <th onClick={() => handleSort('pressure_class')}>Pressure <SortIcon col="pressure_class" /></th>
              <th onClick={() => handleSort('confidence')}>Conf. <SortIcon col="confidence" /></th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((d) => (
              <tr
                key={d.id}
                onClick={() => onRowClick(d)}
                className={selectedDuct?.id === d.id ? 'selected' : ''}
              >
                <td><strong>{d.id}</strong></td>
                {showTypeColumn && <td>{d.duct_type}</td>}
                <td className="dim-cell">{d.dimension || '—'}</td>
                <td>
                  <span className={`pressure-badge ${d.pressure_class.split(' ')[0].toLowerCase()}`}>
                    {d.pressure_class}
                  </span>
                </td>
                <td>
                  <div className="conf-bar">
                    <div
                      className="conf-fill"
                      style={{ width: `${d.confidence * 100}%` }}
                    />
                    <span>{(d.confidence * 100).toFixed(0)}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
