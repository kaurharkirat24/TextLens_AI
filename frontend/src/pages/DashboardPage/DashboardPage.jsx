import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  Calendar,
  CheckCircle2,
  ChevronDown,
  Database,
  Download,
  FileText,
  Info,
  LayoutDashboard,
  Lightbulb,
  Loader2,
  MessageSquare,
  MessageSquareText,
  PieChart,
  Play,
  RefreshCw,
  Smile,
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
  LabelList,
} from 'recharts';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { analyzeDataset, downloadCleanDataset, getAnalysis, getDatasets } from '../../services/api';
import { alpha2ToNumeric } from '../../countryCodes';
import './DashboardPage.css';

const CHART_COLORS = [
  '#a4161a',
  '#660708',
  '#161a1d',
  '#b1a7a6',
  '#d3d3d3',
  '#0b090a',
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

    grouped.other.sort((a, b) => {
      const aIsMap = a.section === 'geo' || a.title?.toLowerCase().includes('location');
      const bIsMap = b.section === 'geo' || b.title?.toLowerCase().includes('location');
      if (aIsMap && !bIsMap) return 1;
      if (!aIsMap && bIsMap) return -1;
      return 0;
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
          <h1>Dashboard</h1>
          <p>{selectedDataset?.original_filename ? 'Overview of your dataset analysis' : 'Select a dataset to view analytics.'}</p>
        </div>
        <button
          className="btn btn-secondary"
          type="button"
          disabled={!selectedId || downloading}
          onClick={downloadCleanData}
        >
          {downloading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
          Download Clean Data
        </button>
      </header>

      <section className="dataset-controls-card card">
        <label className="dataset-picker">
          <span>Dataset</span>
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
          {analyzing ? <Loader2 size={16} className="animate-spin" /> : analysis ? <RefreshCw size={16} /> : null}
          {analysis ? 'Re-analyze' : 'Analyze'}
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
      <div className="cleaning-table-header">
        <span>Status</span>
        <span>Rows</span>
        <span>Removed</span>
        <span>Columns Processed</span>
      </div>
      <div className="cleaning-table-row">
        <div className="status-badge"><CheckCircle2 size={14}/> {status}</div>
        <strong>{rowsBefore.toLocaleString()} to {rowsAfter.toLocaleString()}</strong>
        <strong>{duplicates.toLocaleString()} duplicates, {nulls.toLocaleString()} null rows</strong>
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
      <StatTile icon={Database} colorType="accent" label="Rows" value={(stats.total_rows_original || 0).toLocaleString()} hint="Uploaded records" />
      <StatTile icon={CheckCircle2} colorType="success" label="Analyzed" value={(stats.total_rows_analyzed || 0).toLocaleString()} hint={stats.sampled ? 'Sampled for speed' : 'Full dataset'} />
      <StatTile icon={Lightbulb} colorType="info" label="Insights" value={chartCount.toLocaleString()} hint="Curated visuals" />
      <StatTile icon={FileText} colorType="primary" label="Primary Text" value={formatColumn(primaryText) || 'N/A'} hint="Used for sentiment" />
    </section>
  );
}

function StatTile({ icon: Icon, colorType, label, value, hint }) {
  return (
    <div className="stat-tile card">
      <div className={`stat-icon-wrapper stat-icon-wrapper--${colorType}`}>
        <Icon size={20} />
      </div>
      <div className="stat-content">
        <span>{label}</span>
        <strong>{value}</strong>
        {hint && <em>{hint}</em>}
      </div>
    </div>
  );
}

function KeyInsights({ insights }) {
  if (insights.length === 0) {
    return null;
  }

  const icons = [FileText, CheckCircle2, Smile, MessageSquare, Calendar];
  const colorTypes = ['primary', 'success', 'warning', 'info', 'accent'];

  return (
    <div className="key-insights">
      {insights.map((insight, index) => {
        const Icon = icons[index % icons.length];
        const colorType = colorTypes[index % colorTypes.length];
        return (
          <article className="insight-card card" key={`${insight.label}-${insight.value}`}>
            <span className="insight-label-top">{insight.label}</span>
            <div className={`insight-icon-center insight-icon-center--${colorType}`}>
              <Icon size={28} />
            </div>
            <strong>{insight.value}</strong>
            {insight.detail && <p>{insight.detail}</p>}
          </article>
        );
      })}
    </div>
  );
}

function ChartCard({ chart }) {
  // Strip out the mention of "grouped by countrycode" if it exists in the subtitle
  const subtitle = chart.subtitle?.replace(/grouped by countrycode/i, '')?.trim();
  const isFullWidth = chart.section === 'geo';
  return (
    <article className={`chart-card card ${isFullWidth ? 'chart-card--full-width' : ''}`}>
      <div className="chart-card-header">
        <h2>{chart.title || 'Untitled chart'}</h2>
        {subtitle && <p>{subtitle}</p>}
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
    case 'horizontal_bar':
      if (chart.title?.toLowerCase().includes('location') || chart.subtitle?.toLowerCase().includes('country')) {
        const COMMON_COUNTRIES = new Set(['united states', 'united states of america', 'uk', 'united kingdom', 'india', 'canada', 'australia', 'germany', 'france', 'china', 'japan', 'brazil', 'russia', 'mexico', 'spain', 'italy']);
        const hasKnownCountry = data.some((s) => {
          const label = String(s.label).toUpperCase();
          return alpha2ToNumeric[label] || COMMON_COUNTRIES.has(String(s.label).toLowerCase());
        });
        
        if (hasKnownCountry) {
          return <WorldMapView chart={chart} data={data} />;
        }
      }
      return chart.type === 'bar' ? (
        <BarChartView chart={chart} data={data} />
      ) : (
        <HorizontalBarChartView chart={chart} data={data} />
      );
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
            <Cell key={entry.label} fill={entry.color || CHART_COLORS[index % CHART_COLORS.length]} />
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
      <BarChart data={visibleData} layout="vertical" margin={{ top: 8, right: 48, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.03)" horizontal={false} />
        <XAxis type="number" tick={axisTick} hide />
        <YAxis type="category" dataKey="label" width={128} tick={compactAxisTick} tickFormatter={(val) => typeof val === 'string' && val.length > 20 ? `${val.substring(0, 20)}...` : val} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(164,22,26,0.08)' }} />
        <Bar dataKey="value" fill={CHART_COLORS[0]} radius={[0, 4, 4, 0]} barSize={16}>
          <LabelList dataKey="value" position="right" fill="var(--color-text-dim)" fontSize={11} formatter={(val) => val.toLocaleString()} />
          {visibleData.map((entry) => (
             <Cell key={entry.label} fill={entry.color} />
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
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(102,7,8,0.08)' }} />
        <Bar dataKey="value" fill={CHART_COLORS[1]} radius={[5, 5, 0, 0]} />
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
          stroke={CHART_COLORS[0]}
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
          stroke={CHART_COLORS[2]}
          fill={CHART_COLORS[2]}
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
        <Scatter data={data} fill={CHART_COLORS[3]} />
      </ScatterChart>
    </ChartFrame>
  );
}

function PieChartView({ chart, data }) {
  const innerRadius = chart.type === 'donut' || true ? 65 : 0;
  const total = data.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className="pie-chart-wrapper">
      <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <RechartsPieChart width={250} height={250}>
          <Pie
            data={data}
            dataKey="value"
            nameKey="label"
            cx="50%"
            cy="50%"
            innerRadius={innerRadius}
            outerRadius={96}
            paddingAngle={2}
            stroke="var(--color-bg-card)"
            strokeWidth={3}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip contentStyle={tooltipStyle} />
          {innerRadius > 0 && (
            <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="pie-center-text">
              <tspan x="50%" dy="-0.2em" className="pie-center-value">{total.toLocaleString()}</tspan>
              <tspan x="50%" dy="1.4em" className="pie-center-label">Total Analyzed</tspan>
            </text>
          )}
        </RechartsPieChart>
      </div>
      <div className="pie-custom-legend">
        {data.map(item => (
          <div key={item.label} className="pie-legend-item">
            <span className="pie-legend-dot" style={{ backgroundColor: item.color }} />
            <div className="pie-legend-text">
              <span className="pie-legend-label">{item.label}</span>
              <strong className="pie-legend-value">{item.value.toLocaleString()}</strong>
            </div>
            <span className="pie-legend-pct" style={{ color: item.color, backgroundColor: `${item.color}20` }}>
              {item.percentage}%
            </span>
          </div>
        ))}
      </div>
    </div>
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

const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

function WorldMapView({ chart, data }) {
  const [hoveredData, setHoveredData] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const colorScale = scaleLinear()
    .domain([0, maxValue])
    .range(["#ffccd5", "#660708"]);

  return (
    <div 
      className="world-map-wrapper"
      onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}
    >
      {hoveredData && (
        <div 
          className="world-map-cursor-tooltip" 
          style={{ left: mousePos.x + 15, top: mousePos.y + 15 }}
        >
          <strong>{hoveredData.name}</strong>
          <div>
            <span>{hoveredData.value.toLocaleString()}</span>
            <span>{hoveredData.percentage}%</span>
          </div>
        </div>
      )}

      <div className="world-map-table">
        <div className="world-map-rows">
          {data.slice(0, 5).map((item) => (
            <div className="world-map-row" key={item.label}
                 onMouseEnter={() => setHoveredData({ name: item.label, value: item.value, percentage: item.percentage })}
                 onMouseLeave={() => setHoveredData(null)}>
              <span className="world-map-color" style={{ backgroundColor: colorScale(item.value) }} />
              <span className="world-map-label">{item.label}</span>
              <strong className="world-map-value">{item.value.toLocaleString()}</strong>
              <span className="world-map-pct">{item.percentage}%</span>
            </div>
          ))}
        </div>
      </div>
      <div className="world-map-visual">
        <ComposableMap width={800} height={450} projectionConfig={{ scale: 145, center: [0, 5] }} style={{ width: "100%", height: "auto" }}>
          <Geographies geography={geoUrl}>
            {({ geographies }) =>
              geographies.map((geo) => {
                const d = data.find((s) => {
                  const sLabel = String(s.label).toUpperCase();
                  const numericCode = alpha2ToNumeric[sLabel];
                  if (numericCode && numericCode === geo.id) return true;
                  return String(s.label).toLowerCase() === String(geo.properties?.name).toLowerCase() || String(s.label) === String(geo.id);
                });
                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={d ? colorScale(d.value) : "#f5f3f4"}
                    stroke="#ffffff"
                    strokeWidth={0.5}
                    style={{
                      default: { outline: "none", transition: "all 0.2s ease" },
                      hover: { fill: d ? "#660708" : "#e0e0e0", outline: "none", cursor: d ? "pointer" : "default" },
                      pressed: { outline: "none" },
                    }}
                    onMouseEnter={() => {
                      if (d) setHoveredData({ name: geo.properties.name, value: d.value, percentage: d.percentage });
                    }}
                    onMouseLeave={() => setHoveredData(null)}
                  />
                );
              })
            }
          </Geographies>
        </ComposableMap>
      </div>
    </div>
  );
}

function normalizeChartData(chart) {
  let mappedData = [];
  
  const mapSentimentColor = (label) => {
    const l = String(label).toLowerCase();
    if (l === 'positive') return CHART_COLORS[3];
    if (l === 'neutral') return CHART_COLORS[4];
    if (l === 'negative') return CHART_COLORS[0];
    return null;
  };

  if (Array.isArray(chart.data) && chart.data.length > 0) {
    mappedData = chart.data
      .map((item, index) => {
        const label = String(item.name ?? item.label ?? item.x ?? chart.x?.[index] ?? '');
        return {
          label,
          value: Number(item.value ?? item.y ?? chart.y?.[index] ?? 0),
          color: mapSentimentColor(label) || CHART_COLORS[index % CHART_COLORS.length],
        };
      })
      .filter((item) => item.label && Number.isFinite(item.value));
  } else {
    const labels = Array.isArray(chart.x) ? chart.x : [];
    const values = Array.isArray(chart.y) ? chart.y : [];

    mappedData = labels
      .map((label, index) => ({
        label: String(label),
        value: Number(values[index] ?? 0),
        color: mapSentimentColor(label) || CHART_COLORS[index % CHART_COLORS.length],
      }))
      .filter((item) => item.label && Number.isFinite(item.value));
  }

  const total = mappedData.reduce((sum, item) => sum + item.value, 0);
  mappedData = mappedData.map(item => ({
    ...item,
    percentage: total > 0 ? ((item.value / total) * 100).toFixed(1) : 0
  }));

  return mappedData;
}

function formatColumn(value) {
  return value ? String(value).replace(/[_-]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase()) : '';
}

const axisTick = { fill: '#b1a7a6', fontSize: 11 };
const smallAxisTick = { fill: '#b1a7a6', fontSize: 10 };
const compactAxisTick = { fill: '#b1a7a6', fontSize: 10, width: 120 };

const tooltipStyle = {
  background: '#ffffff',
  border: '1px solid #d3d3d3',
  borderRadius: 8,
  color: '#161a1d',
};
