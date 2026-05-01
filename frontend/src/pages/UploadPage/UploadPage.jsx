import { useState, useCallback, useRef } from 'react';
import {
  Upload, FileUp, CheckCircle2, AlertTriangle,
  XCircle, Info, FileSpreadsheet, Loader2,
  ChevronDown, ChevronUp, Eye
} from 'lucide-react';
import { uploadFile, getDatasetPreview } from '../../services/api';
import './UploadPage.css';

export default function UploadPage() {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [showIssues, setShowIssues] = useState(true);
  const fileInputRef = useRef(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  }, []);

  const processFile = useCallback(async (file) => {
    setResult(null);
    setError(null);
    setPreview(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      const data = await uploadFile(file, null, (e) => {
        if (e.total) setUploadProgress(Math.round((e.loaded / e.total) * 100));
      });
      setResult(data);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Upload failed';
      setError(msg);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const loadPreview = async () => {
    if (!result?.dataset_id) return;
    try {
      const data = await getDatasetPreview(result.dataset_id, 20);
      setPreview(data);
    } catch {
      /* silently fail */
    }
  };

  const severityIcon = (sev) => {
    switch (sev) {
      case 'error':   return <XCircle size={14} className="issue-icon issue-icon--error" />;
      case 'warning': return <AlertTriangle size={14} className="issue-icon issue-icon--warning" />;
      default:        return <Info size={14} className="issue-icon issue-icon--info" />;
    }
  };

  const report = result?.report;
  const quickInsights = buildQuickInsights(report);

  return (
    <div className="upload-page animate-fadeIn">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="page-header">
        <h1 className="page-title">
          <Upload size={24} />
          Upload Data
        </h1>
        <p className="page-subtitle">
          Upload your CSV file and we'll automatically detect the text column, validate the data, and prepare it for analysis.
        </p>
      </div>

      {/* ── Drop Zone ───────────────────────────────────────────── */}
      <div
        className={`drop-zone ${dragActive ? 'drop-zone--active' : ''} ${uploading ? 'drop-zone--uploading' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="file-upload-input"
        />

        {uploading ? (
          <div className="drop-zone-uploading">
            <Loader2 size={40} className="animate-spin" style={{ color: 'var(--color-accent)' }} />
            <p className="drop-zone-text">Processing your file…</p>
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }} />
            </div>
            <span className="drop-zone-hint">{uploadProgress}% uploaded</span>
          </div>
        ) : (
          <>
            <div className="drop-zone-icon">
              <FileUp size={32} />
            </div>
            <p className="drop-zone-text">
              Drag & drop your CSV file here
            </p>
            <p className="drop-zone-hint">
              or click to browse · Supports CSV files up to 50MB
            </p>
            <div className="drop-zone-formats">
              <span className="format-badge">
                <FileSpreadsheet size={12} /> CSV
              </span>
            </div>
          </>
        )}
      </div>

      {/* ── Error ───────────────────────────────────────────────── */}
      {error && (
        <div className="upload-error card animate-fadeInUp">
          <XCircle size={18} />
          <div>
            <strong>Upload failed</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* ── Results ─────────────────────────────────────────────── */}
      {report && (
        <div className="results animate-fadeInUp">
          {/* Status banner */}
          <div className={`result-banner ${report.success ? 'result-banner--success' : 'result-banner--error'}`}>
            {report.success ? <CheckCircle2 size={20} /> : <XCircle size={20} />}
            <div>
              <strong>{report.success ? 'Ingestion Successful' : 'Ingestion Failed'}</strong>
              <span className="result-banner-file">{report.file_name}</span>
            </div>
          </div>

          {/* Column Detection */}
          {report.text_column && (
            <div className="card result-card animate-fadeInUp stagger-1">
              <h3 className="result-card-title">Text Column Detection</h3>
              <div className="result-grid">
                <div className="result-item">
                  <span className="result-label">Column</span>
                  <span className="result-value result-value--highlight">{report.text_column.column_name}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Method</span>
                  <span className="result-value">{report.text_column.method}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Confidence</span>
                  <span className={`badge badge-${report.text_column.confidence === 'high' ? 'success' : report.text_column.confidence === 'medium' ? 'warning' : 'error'}`}>
                    {report.text_column.confidence}
                  </span>
                </div>
              </div>
              {report.text_column.reasoning && (
                <p className="result-reasoning">{report.text_column.reasoning}</p>
              )}
            </div>
          )}

          {/* Stats */}
          {report.stats && (
            <div className="card result-card animate-fadeInUp stagger-2">
              <h3 className="result-card-title">Dataset Statistics</h3>
              <div className="stats-grid">
                <StatCard label="Total Rows" value={report.stats.total_rows.toLocaleString()} />
                <StatCard label="Clean Rows" value={report.stats.clean_count.toLocaleString()} accent />
                <StatCard label="Null Values" value={report.stats.null_count.toLocaleString()} warn={report.stats.null_count > 0} />
                <StatCard label="Null Ratio" value={`${(report.stats.null_ratio * 100).toFixed(1)}%`} warn={report.stats.null_ratio > 0.1} />
                <StatCard label="Duplicates" value={report.stats.duplicate_count.toLocaleString()} />
                <StatCard label="Avg Length" value={`${report.stats.avg_text_length.toFixed(0)} chars`} />
                <StatCard label="Empty Rows" value={report.stats.empty_count.toLocaleString()} warn={report.stats.empty_count > 0} />
                <StatCard label="Columns" value={report.stats.total_columns} />
              </div>
            </div>
          )}

          {quickInsights.length > 0 && (
            <div className="card result-card animate-fadeInUp stagger-3">
              <h3 className="result-card-title">Quick Insights</h3>
              <div className="insights-list">
                {quickInsights.map((insight) => (
                  <div key={insight.label} className={`insight-item insight-item--${insight.tone}`}>
                    <span className="insight-label">{insight.label}</span>
                    <span className="insight-value">{insight.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Issues */}
          {report.issues && report.issues.length > 0 && (
            <div className="card result-card animate-fadeInUp stagger-4">
              <button className="result-card-toggle" onClick={() => setShowIssues(!showIssues)}>
                <h3 className="result-card-title" style={{ margin: 0 }}>
                  Validation Issues
                  <span className="issue-count">{report.issues.length}</span>
                </h3>
                {showIssues ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>

              {showIssues && (
                <div className="issues-list">
                  {report.issues.map((issue, idx) => (
                    <div key={idx} className={`issue-item issue-item--${issue.severity}`}>
                      {severityIcon(issue.severity)}
                      <div className="issue-content">
                        <span className="issue-category">{issue.category.replace(/_/g, ' ')}</span>
                        <span className="issue-message">{issue.message}</span>
                      </div>
                      <span className={`badge badge-${issue.severity === 'error' ? 'error' : issue.severity === 'warning' ? 'warning' : 'info'}`}>
                        {issue.count}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Preview button */}
          {report.success && !preview && (
            <button className="btn btn-secondary preview-btn animate-fadeInUp stagger-4" onClick={loadPreview}>
              <Eye size={16} />
              Preview Clean Data
            </button>
          )}

          {/* Data Preview */}
          {preview && (
            <div className="card result-card animate-fadeInUp">
              <h3 className="result-card-title">
                Data Preview
                <span className="result-card-subtitle">
                  Showing {preview.showing} of {preview.total_rows.toLocaleString()} rows
                </span>
              </h3>
              <div className="preview-table-wrapper">
                <table className="preview-table">
                  <thead>
                    <tr>
                      {preview.columns.map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <tr key={i}>
                        {preview.columns.map((col) => (
                          <td key={col} title={row[col]}>
                            {row[col]?.length > 80 ? row[col].slice(0, 80) + '…' : row[col]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


/* ── StatCard sub-component ─────────────────────────────────────────────── */

function StatCard({ label, value, accent = false, warn = false }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${accent ? 'stat-value--accent' : ''} ${warn ? 'stat-value--warn' : ''}`}>
        {value}
      </span>
    </div>
  );
}

function buildQuickInsights(report) {
  if (!report?.success || !report.stats) return [];

  const stats = report.stats;
  const totalRows = stats.total_rows || 0;
  const cleanRows = stats.clean_count || 0;
  const usableRatio = totalRows ? cleanRows / totalRows : 0;
  const insights = [
    {
      label: 'Text column',
      value: `${report.text_column?.column_name || stats.text_column} (${report.text_column?.confidence || 'detected'} confidence)`,
      tone: 'accent',
    },
    {
      label: 'Usable rows',
      value: `${cleanRows.toLocaleString()} of ${totalRows.toLocaleString()} (${Math.round(usableRatio * 100)}%)`,
      tone: usableRatio >= 0.8 ? 'success' : usableRatio >= 0.5 ? 'warning' : 'error',
    },
    {
      label: 'Typical text length',
      value: `${Math.round(stats.avg_text_length)} avg chars, ${Math.round(stats.median_text_length)} median`,
      tone: 'neutral',
    },
  ];

  if (stats.null_count > 0) {
    insights.push({
      label: 'Missing text',
      value: `${stats.null_count.toLocaleString()} rows need attention`,
      tone: stats.null_ratio > 0.3 ? 'error' : 'warning',
    });
  }

  if (stats.duplicate_count > 0) {
    insights.push({
      label: 'Duplicates',
      value: `${stats.duplicate_count.toLocaleString()} repeated text rows found`,
      tone: 'warning',
    });
  }

  if (report.issues?.length) {
    insights.push({
      label: 'Validation checks',
      value: `${report.issues.length} finding${report.issues.length === 1 ? '' : 's'} surfaced`,
      tone: report.issues.some((issue) => issue.severity === 'error') ? 'error' : 'neutral',
    });
  }

  return insights;
}
