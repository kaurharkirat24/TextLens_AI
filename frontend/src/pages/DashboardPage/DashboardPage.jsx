import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  ChevronDown,
  Database,
  LayoutDashboard,
  Loader2,
  PieChart,
  Play,
  RefreshCw,
} from 'lucide-react';
import {
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
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { analyzeDataset, getAnalysis, getDatasets } from '../../services/api';
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

const SUPPORTED_CHARTS = new Set(['bar', 'pie', 'donut', 'line', 'histogram']);

export default function DashboardPage() {
  const [datasets, setDatasets] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;

    async function loadDatasets() {
      try {
        const payload = await getDatasets();
        if (!active) return;

        const available = (payload.datasets || []).filter((dataset) => dataset.status !== 'failed');
        setDatasets(available);
        setSelectedId((current) => current || available[0]?.id || '');
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

  const charts = useMemo(
    () => (analysis?.charts || []).filter((chart) => SUPPORTED_CHARTS.has(chart.type)),
    [analysis],
  );

  const runAnalysis = useCallback(async () => {
    if (!selectedId) return;

    setAnalyzing(true);
    setError('');
    setAnalysis(null);

    try {
      const payload = await analyzeDataset(selectedId);
      setAnalysis(payload);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed.');
    } finally {
      setAnalyzing(false);
    }
  }, [selectedId]);

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
      </section>

      {error && (
        <div className="dashboard-message dashboard-message--error card">
          <AlertTriangle size={18} />
          <span>{error}</span>
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
          <StatsStrip analysis={analysis} chartCount={charts.length} />

          {charts.length > 0 ? (
            <section className="charts-grid" aria-label="Generated charts">
              {charts.map((chart, index) => (
                <ChartCard chart={chart} key={chart.id || `${chart.type}-${index}`} />
              ))}
            </section>
          ) : (
            <div className="dashboard-empty card">
              <PieChart size={28} />
              <h2>No supported charts</h2>
              <p>The current analysis did not return bar, pie, line, or histogram configs.</p>
            </div>
          )}
        </>
      )}
    </main>
  );
}

function StatsStrip({ analysis, chartCount }) {
  const stats = analysis.stats || {};
  const schemaSummary = analysis.schema?.summary || {};
  const schemaText = Object.entries(schemaSummary)
    .filter(([, columns]) => columns.length > 0)
    .map(([type, columns]) => `${columns.length} ${type}`)
    .join(', ');

  return (
    <section className="dashboard-stats">
      <StatTile label="Rows" value={(stats.total_rows_original || 0).toLocaleString()} />
      <StatTile label="Analyzed" value={(stats.total_rows_analyzed || 0).toLocaleString()} />
      <StatTile label="Charts" value={chartCount.toLocaleString()} />
      <StatTile label="Schema" value={schemaText || 'N/A'} />
    </section>
  );
}

function StatTile({ label, value }) {
  return (
    <div className="stat-tile card">
      <span>{label}</span>
      <strong>{value}</strong>
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
    case 'histogram':
      return <HistogramView chart={chart} data={data} />;
    case 'line':
      return <LineChartView chart={chart} data={data} />;
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

const axisTick = { fill: '#8b90a8', fontSize: 11 };
const smallAxisTick = { fill: '#8b90a8', fontSize: 10 };

const tooltipStyle = {
  background: '#1a1d2e',
  border: '1px solid #2a2d3e',
  borderRadius: 8,
  color: '#e4e6f0',
};
