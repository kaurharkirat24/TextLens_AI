import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  ChevronDown,
  Database,
  Download,
  FileText,
  Info,
  LayoutDashboard,
  Loader2,
  MessageSquareText,
  PieChart,
  Play,
  RefreshCw,
  Sparkles,
  ThumbsUp,
  TrendingUp,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart as RechartsPieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { analyzeDataset, downloadCleanDataset, getAnalysis, getDatasets } from '../../services/api';
import './DashboardPage.css';

const CHART_COLORS = [
  '#4f6ef7',
  '#34d399',
  '#fbbf24',
  '#f87171',
  '#a855f7',
  '#38bdf8',
  '#fb923c',
  '#f472b6',
];

const SUPPORTED_CHARTS = new Set(['bar', 'horizontal_bar', 'pie', 'donut', 'line', 'area', 'histogram', 'scatter']);

const DASHBOARD_SECTIONS = [
  { id: 'key', title: 'Key Insights', icon: Sparkles },
  { id: 'content', title: 'Content Insights', icon: MessageSquareText },
  { id: 'engagement', title: 'Engagement Insights', icon: ThumbsUp },
  { id: 'time', title: 'Time Insights', icon: TrendingUp },
];

const SELECTED_DATASET_KEY = 'textlens:selectedDatasetId';

export default function DashboardPage() {
  const [datasets, setDatasets] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');
  const [toast, setToast] = useState('');

  useEffect(() => {
    let active = true;

    async function loadDatasets() {
      try {
        const payload = await getDatasets();
        if (!active) return;

        const available = (payload.datasets || []).filter((dataset) => dataset.status !== 'failed');
        const cachedId = localStorage.getItem(SELECTED_DATASET_KEY);
        const cachedAvailable = available.some((dataset) => dataset.id === cachedId);
        setDatasets(available);
        setSelectedId((current) => current || (cachedAvailable ? cachedId : available[0]?.id || ''));
      } catch (err) {
        if (active) setError(err.message || 'Unable to load datasets.');
      }
    }

    loadDatasets();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) {
      return;
    }

    let active = true;

    async function loadSavedAnalysis() {
      setLoading(true);
      setError('');
      try {
        const payload = await getAnalysis(selectedId);
        if (active) setAnalysis(payload);
      } catch {
        if (active) setAnalysis(null);
      } finally {
        if (active) setLoading(false);
      }
    }

    loadSavedAnalysis();
    return () => {
      active = false;
    };
  }, [selectedId]);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedId),
    [datasets, selectedId],
  );

  const charts = useMemo(() => (analysis?.charts || []).filter((chart) => SUPPORTED_CHARTS.has(chart.type)), [analysis]);

  const chartsBySection = useMemo(() => {
    const grouped = { key: [], content: [], engagement: [], time: [], other: [] };
    charts.forEach((chart) => {
      const section = grouped[chart.section] ? chart.section : 'other';
      grouped[section].push(chart);
    });
    return grouped;
  }, [charts]);

  const keyInsights = analysis?.insights?.dataset?.key_insights || [];

  const runAnalysis = useCallback(async () => {
    if (!selectedId) return;

    setAnalyzing(true);
    setError('');
    setToast('');
    setAnalysis(null);

    try {
      const payload = await analyzeDataset(selectedId);
      setAnalysis(payload);
      localStorage.setItem(SELECTED_DATASET_KEY, selectedId);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed.');
    } finally {
      setAnalyzing(false);
    }
  }, [selectedId]);

  const downloadCleanData = useCallback(async () => {
    if (!selectedId) return;
    if (!analysis) {
      setToast('Run analysis first to generate a clean dataset download.');
      return;
    }

    setDownloading(true);
    setError('');
    setToast('');
    try {
      const { blob, filename } = await downloadCleanDataset(selectedId);
      const href = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = href;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(href);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Unable to download clean dataset.');
    } finally {
      setDownloading(false);
    }
  }, [analysis, selectedId]);

  return (
    <main className="dashboard-page">
      <header className="dashboard-header">
        <div>
          <h1>
            <LayoutDashboard size={24} />
            Dashboard
          </h1>
          <p>{selectedDataset?.original_filename || 'Select a dataset to view analytics.'}</p>
        </div>
      </header>

      <section className="dashboard-controls card">
        <label className="dataset-picker">
          <span>
            <Database size={15} />
            Dataset
          </span>
          <div className="dataset-select-wrap">
            <select
              value={selectedId}
              onChange={(event) => {
                setAnalysis(null);
                setSelectedId(event.target.value);
                if (event.target.value) {
                  localStorage.setItem(SELECTED_DATASET_KEY, event.target.value);
                } else {
                  localStorage.removeItem(SELECTED_DATASET_KEY);
                }
              }}
            >
              <option value="">Select a dataset</option>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.original_filename}
                </option>
              ))}
            </select>
            <ChevronDown size={16} />
          </div>
        </label>

        <button className="btn btn-primary" type="button" disabled={!selectedId || analyzing} onClick={runAnalysis}>
          {analyzing ? <Loader2 size={16} className="animate-spin" /> : analysis ? <RefreshCw size={16} /> : <Play size={16} />}
          {analysis ? 'Re-analyze' : 'Analyze'}
        </button>

        <button
          className="btn btn-secondary"
          type="button"
          disabled={!selectedId || downloading}
          onClick={downloadCleanData}
        >
          {downloading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
          Download Clean Data
        </button>
      </section>

      {error && (
        <div className="dashboard-message dashboard-message--error card">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {toast && (
        <div className="dashboard-toast card" role="status">
          <Info size={18} />
          <span>{toast}</span>
          <button type="button" onClick={() => setToast('')}>Dismiss</button>
        </div>
      )}

      {(loading || analyzing) && (
        <div className="dashboard-loading">
          <Loader2 size={36} className="animate-spin" />
          <span>{analyzing ? 'Running analysis...' : 'Loading analysis...'}</span>
        </div>
      )}

      {!loading && !analyzing && !analysis && !error && (
        <div className="dashboard-empty card">
          <BarChart3 size={28} />
          <h2>No analysis loaded</h2>
          <p>Choose a dataset and run analysis to populate this dashboard.</p>
        </div>
      )}

      {analysis && !analyzing && (
        <>
          <DashboardSection title="Overview" icon={BarChart3}>
            <StatsStrip analysis={analysis} chartCount={charts.length} />
            <CleaningSummary report={analysis.cleaning_report} />
          </DashboardSection>

          {(keyInsights.length > 0 || chartsBySection.key.length > 0) && (
            <DashboardSection title="Key Insights" icon={Sparkles}>
              <KeyInsights insights={keyInsights} />
              {chartsBySection.key.length > 0 && (
                <div className="charts-grid charts-grid--featured">
                  {chartsBySection.key.map((chart, index) => (
                    <ChartCard chart={chart} key={chart.id || `${chart.type}-${index}`} />
                  ))}
                </div>
              )}
            </DashboardSection>
          )}

          {DASHBOARD_SECTIONS.filter((section) => section.id !== 'key').map(({ id, title, icon }) => (
            chartsBySection[id].length > 0 && (
              <DashboardSection title={title} icon={icon} key={id}>
                <div className="charts-grid">
                  {chartsBySection[id].map((chart, index) => (
                    <ChartCard chart={chart} key={chart.id || `${chart.type}-${index}`} />
                  ))}
                </div>
              </DashboardSection>
            )
          ))}

          {chartsBySection.other.length > 0 && (
            <DashboardSection title="Additional Signals" icon={FileText}>
              <div className="charts-grid">
                {chartsBySection.other.map((chart, index) => (
                  <ChartCard chart={chart} key={chart.id || `${chart.type}-${index}`} />
                ))}
              </div>
            </DashboardSection>
          )}

          {charts.length === 0 && (
            <div className="dashboard-empty card">
              <PieChart size={28} />
              <h2>No supported charts</h2>
              <p>The current analysis did not return enough dataset-level signals for charting.</p>
            </div>
          )}
        </>
      )}
    </main>
  );
}

function CleaningSummary({ report }) {
  if (!report) {
    return null;
  }

  const status = report.cleaning_status === 'already_clean' ? 'Already clean' : 'Cleaned';
  const rowsBefore = Number(report.rows_before ?? report.original_rows ?? 0);
  const rowsAfter = Number(report.rows_after ?? report.final_rows ?? 0);
  const duplicates = Number(report.duplicates_removed ?? report.duplicate_rows_removed ?? 0);
  const nulls = Number(report.nulls_removed ?? report.null_rows_removed ?? 0);
  const columns = report.columns_processed || [];

  return (
    <div className="cleaning-summary card">
      <div>
        <span>Cleaning status</span>
        <strong>{status}</strong>
      </div>
      <div>
        <span>Rows</span>
        <strong>{rowsBefore.toLocaleString()} to {rowsAfter.toLocaleString()}</strong>
      </div>
      <div>
        <span>Removed</span>
        <strong>{duplicates.toLocaleString()} duplicates, {nulls.toLocaleString()} null rows</strong>
      </div>
      <div>
        <span>Columns processed</span>
        <strong>{columns.length ? columns.map(formatColumn).join(', ') : 'N/A'}</strong>
      </div>
    </div>
  );
}

function DashboardSection({ title, icon: Icon, children }) {
  return (
    <section className="dashboard-section">
      <div className="section-heading">
        <span className="section-heading-icon">
          <Icon size={17} />
        </span>
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function StatsStrip({ analysis, chartCount }) {
  const stats = analysis.stats || {};
  const schemaSummary = analysis.schema?.summary || {};
  const roles = analysis.column_roles || analysis.insights?.summary?.column_roles || {};
  const primaryText = roles.primary_text || schemaSummary.text?.[0];

  return (
    <section className="dashboard-stats">
      <StatTile label="Rows" value={(stats.total_rows_original || 0).toLocaleString()} hint="Uploaded records" />
      <StatTile label="Analyzed" value={(stats.total_rows_analyzed || 0).toLocaleString()} hint={stats.sampled ? 'Sampled for speed' : 'Full dataset'} />
      <StatTile label="Insights" value={chartCount.toLocaleString()} hint="Curated visuals" />
      <StatTile label="Primary Text" value={formatColumn(primaryText) || 'N/A'} hint="Used for sentiment" />
    </section>
  );
}

function StatTile({ label, value, hint }) {
  return (
    <div className="stat-tile card">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <em>{hint}</em>}
    </div>
  );
}

function KeyInsights({ insights }) {
  if (insights.length === 0) {
    return null;
  }

  return (
    <div className="key-insights">
      {insights.map((insight) => (
        <article className="insight-card card" key={`${insight.label}-${insight.value}`}>
          <span>{insight.label}</span>
          <strong>{insight.value}</strong>
          {insight.detail && <p>{insight.detail}</p>}
        </article>
      ))}
    </div>
  );
}

function ChartCard({ chart }) {
  return (
    <article className="chart-card card">
      <div className="chart-card-header">
        <h2>{chart.title || 'Untitled chart'}</h2>
        {chart.subtitle && <p>{chart.subtitle}</p>}
      </div>
      <DynamicChart chart={chart} />
    </article>
  );
}

function DynamicChart({ chart }) {
  const data = normalizeChartData(chart);

  if (data.length === 0) {
    return <div className="chart-placeholder">No chart data available.</div>;
  }

  switch (chart.type) {
    case 'bar':
      return <BarChartView chart={chart} data={data} />;
    case 'horizontal_bar':
      return <HorizontalBarChartView chart={chart} data={data} />;
    case 'histogram':
      return <HistogramView chart={chart} data={data} />;
    case 'line':
      return <LineChartView chart={chart} data={data} />;
    case 'area':
      return <AreaChartView chart={chart} data={data} />;
    case 'scatter':
      return <ScatterChartView chart={chart} data={normalizeScatterData(chart)} />;
    case 'pie':
    case 'donut':
      return <PieChartView chart={chart} data={data} />;
    default:
      return <div className="chart-placeholder">Unsupported chart type: {chart.type}</div>;
  }
}

function BarChartView({ chart, data }) {
  return (
    <ChartFrame>
      <BarChart data={data} margin={{ top: 8, right: 18, bottom: 36, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
        <XAxis dataKey="label" angle={-30} textAnchor="end" height={58} tick={axisTick} />
        <YAxis tick={axisTick} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(79,110,247,0.08)' }} />
        <Bar dataKey="value" radius={[5, 5, 0, 0]}>
          {data.map((entry, index) => (
            <Cell key={entry.label} fill={entry.color || chart.colors?.[index] || CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ChartFrame>
  );
}

function HorizontalBarChartView({ chart, data }) {
  const visibleData = data.slice(0, 12);
  return (
    <ChartFrame>
      <BarChart data={visibleData} layout="vertical" margin={{ top: 8, right: 18, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" horizontal={false} />
        <XAxis type="number" tick={axisTick} />
        <YAxis type="category" dataKey="label" width={128} tick={compactAxisTick} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(79,110,247,0.08)' }} />
        <Bar dataKey="value" fill={chart.color || '#4f6ef7'} radius={[0, 5, 5, 0]} />
      </BarChart>
    </ChartFrame>
  );
}

function HistogramView({ chart, data }) {
  return (
    <ChartFrame>
      <BarChart data={data} margin={{ top: 8, right: 18, bottom: 42, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
        <XAxis dataKey="label" angle={-35} textAnchor="end" height={64} tick={smallAxisTick} />
        <YAxis tick={axisTick} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(168,85,247,0.08)' }} />
        <Bar dataKey="value" fill={chart.color || '#a855f7'} radius={[5, 5, 0, 0]} />
      </BarChart>
    </ChartFrame>
  );
}

function LineChartView({ chart, data }) {
  return (
    <ChartFrame>
      <LineChart data={data} margin={{ top: 8, right: 18, bottom: 36, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
        <XAxis dataKey="label" angle={-30} textAnchor="end" height={58} tick={axisTick} />
        <YAxis tick={axisTick} />
        <Tooltip contentStyle={tooltipStyle} />
        <Line
          type="monotone"
          dataKey="value"
          stroke={chart.color || '#4f6ef7'}
          strokeWidth={2.4}
          dot={{ r: 3 }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ChartFrame>
  );
}

function AreaChartView({ chart, data }) {
  return (
    <ChartFrame>
      <AreaChart data={data} margin={{ top: 8, right: 18, bottom: 36, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
        <XAxis dataKey="label" angle={-30} textAnchor="end" height={58} tick={axisTick} />
        <YAxis tick={axisTick} />
        <Tooltip contentStyle={tooltipStyle} />
        <Area
          type="monotone"
          dataKey="value"
          stroke={chart.color || '#38bdf8'}
          fill={chart.color || '#38bdf8'}
          fillOpacity={0.18}
          strokeWidth={2.4}
        />
      </AreaChart>
    </ChartFrame>
  );
}

function ScatterChartView({ chart, data }) {
  return (
    <ChartFrame>
      <ScatterChart margin={{ top: 8, right: 18, bottom: 32, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
        <XAxis type="number" dataKey="x" name={chart.x_label || 'Engagement'} tick={axisTick} />
        <YAxis type="number" dataKey="y" name={chart.y_label || 'Sentiment'} tick={axisTick} domain={[-1, 1]} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }} />
        <Scatter data={data} fill={chart.color || '#f472b6'} />
      </ScatterChart>
    </ChartFrame>
  );
}

function PieChartView({ chart, data }) {
  const innerRadius = chart.type === 'donut' ? 58 : 0;

  return (
    <ChartFrame>
      <RechartsPieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="label"
          cx="50%"
          cy="48%"
          innerRadius={innerRadius}
          outerRadius={96}
          paddingAngle={2}
          stroke="rgba(10,12,20,0.65)"
          strokeWidth={2}
        >
          {data.map((entry, index) => (
            <Cell key={entry.label} fill={entry.color || CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip contentStyle={tooltipStyle} />
        <Legend formatter={(value) => <span className="chart-legend-label">{value}</span>} />
      </RechartsPieChart>
    </ChartFrame>
  );
}

function ChartFrame({ children }) {
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}

function normalizeScatterData(chart) {
  return (chart.data || [])
    .map((item, index) => ({
      x: Number(item.x ?? chart.x?.[index]),
      y: Number(item.y ?? chart.y?.[index]),
    }))
    .filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y));
}

function normalizeChartData(chart) {
  if (Array.isArray(chart.data) && chart.data.length > 0) {
    return chart.data
      .map((item, index) => ({
        label: String(item.name ?? item.label ?? item.x ?? chart.x?.[index] ?? ''),
        value: Number(item.value ?? item.y ?? chart.y?.[index] ?? 0),
        color: item.color,
      }))
      .filter((item) => item.label && Number.isFinite(item.value));
  }

  const labels = Array.isArray(chart.x) ? chart.x : [];
  const values = Array.isArray(chart.y) ? chart.y : [];

  return labels
    .map((label, index) => ({
      label: String(label),
      value: Number(values[index] ?? 0),
      color: chart.colors?.[index],
    }))
    .filter((item) => item.label && Number.isFinite(item.value));
}

function formatColumn(value) {
  return value ? String(value).replace(/[_-]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase()) : '';
}

const axisTick = { fill: '#8b90a8', fontSize: 11 };
const smallAxisTick = { fill: '#8b90a8', fontSize: 10 };
const compactAxisTick = { fill: '#8b90a8', fontSize: 10, width: 120 };

const tooltipStyle = {
  background: '#1a1d2e',
  border: '1px solid #2a2d3e',
  borderRadius: 8,
  color: '#e4e6f0',
};
