import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Bot,
  ChevronDown,
  Cpu,
  Database,
  FileSearch,
  Hash,
  Loader2,
  MessageSquareText,
  Play,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  TerminalSquare,
  Zap,
} from 'lucide-react';
import { askDatasetQuestion, embedDataset, getDatasets, searchDataset } from '../../services/api';
import './QAPage.css';

const SELECTED_DATASET_KEY = 'textlens:selectedDatasetId';
const DEFAULT_TOP_K = 5;

export default function QAPage() {
  const [datasets, setDatasets] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [embedding, setEmbedding] = useState(false);
  const [searching, setSearching] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [error, setError] = useState('');
  const [embedResult, setEmbedResult] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [searchResults, setSearchResults] = useState([]);
  const [qaResult, setQaResult] = useState(null);
  const [logs, setLogs] = useState([]);

  const addLog = useCallback((level, message, details = null) => {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      level,
      message,
      details,
    };
    setLogs((current) => [entry, ...current].slice(0, 18));

    const logger = level === 'error' ? console.error : level === 'warn' ? console.warn : console.info;
    logger(`[TextLens Phase 3] ${message}`, details || '');
  }, []);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedId),
    [datasets, selectedId],
  );

  const displayStatus = embedResult?.embedding_status || selectedDataset?.embedding_status || 'not_started';
  const displayDimension = embedResult?.dimension || selectedDataset?.embedding_dimension;
  const displayModel = embedResult?.model || selectedDataset?.embedding_model;
  const displayIndex = embedResult?.index_name || selectedDataset?.embedding_index_name || 'textlens-ai';
  const displayCount = embedResult
    ? (embedResult.embedded_count || 0) + (embedResult.skipped_existing || 0)
    : selectedDataset?.embedding_count || 0;
  const displayProgress = embedResult?.embedding_progress ?? selectedDataset?.embedding_progress ?? 0;
  const canUseSemantic = Boolean(selectedId && displayStatus === 'completed');

  const refreshDatasets = useCallback(async () => {
    setLoadingDatasets(true);
    try {
      const payload = await getDatasets();
      const available = (payload.datasets || []).filter((dataset) => dataset.status !== 'failed');
      setDatasets(available);
      return available.find((dataset) => dataset.id === selectedId);
    } finally {
      setLoadingDatasets(false);
    }
  }, [selectedId]);

  useEffect(() => {
    let active = true;

    async function loadDatasets() {
      setLoadingDatasets(true);
      setError('');
      addLog('info', 'Loading dataset registry');

      try {
        const payload = await getDatasets();
        if (!active) return;

        const available = (payload.datasets || []).filter((dataset) => dataset.status !== 'failed');
        const cachedId = localStorage.getItem(SELECTED_DATASET_KEY);
        const cachedAvailable = available.some((dataset) => dataset.id === cachedId);
        setDatasets(available);
        setSelectedId((current) => current || (cachedAvailable ? cachedId : available[0]?.id || ''));
        addLog('success', `Loaded ${available.length} datasets`);
      } catch (err) {
        if (!active) return;
        const message = readError(err, 'Unable to load datasets.');
        setError(message);
        addLog('error', message);
      } finally {
        if (active) setLoadingDatasets(false);
      }
    }

    loadDatasets();
    return () => {
      active = false;
    };
  }, [addLog]);

  // --- Background Polling for Embedding ---
  useEffect(() => {
    if (displayStatus !== 'processing' || !selectedId) return;

    let active = true;
    const interval = setInterval(async () => {
      try {
        const updated = await refreshDatasets();
        if (active && updated?.embedding_status === 'completed') {
          addLog('success', 'Background embedding completed successfully');
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Polling failed', err);
      }
    }, 3000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [displayStatus, selectedId, refreshDatasets, addLog]);

  const handleDatasetChange = (datasetId) => {
    setSelectedId(datasetId);
    setEmbedResult(null);
    setSearchResults([]);
    setQaResult(null);
    setError('');
    if (datasetId) {
      localStorage.setItem(SELECTED_DATASET_KEY, datasetId);
      addLog('info', `Selected dataset namespace ${datasetId}`);
    } else {
      localStorage.removeItem(SELECTED_DATASET_KEY);
    }
  };

  const handleRefresh = async () => {
    setError('');
    try {
      await refreshDatasets();
      addLog('success', 'Dataset registry refreshed');
    } catch (err) {
      const message = readError(err, 'Unable to refresh datasets.');
      setError(message);
      addLog('error', message, err.response?.data || null);
    }
  };

  const runEmbedding = async () => {
    if (!selectedId) return;

    setEmbedding(true);
    setError('');
    setEmbedResult(null);
    addLog('info', 'Embedding started', { dataset_id: selectedId, namespace: selectedId });

    try {
      const startedAt = performance.now();
      const result = await embedDataset(selectedId);
      const elapsedMs = Math.round(performance.now() - startedAt);
      setEmbedResult(result);
      setSearchResults([]);
      setQaResult(null);
      await refreshDatasets();

      addLog('success', result.message || 'Embedding completed with dimension safety checks', {
        elapsed_ms: elapsedMs,
        dimension: result.dimension,
        embedded_count: result.embedded_count,
        skipped_existing: result.skipped_existing,
        model: result.model,
        index_name: result.index_name,
        namespace: result.namespace,
      });
    } catch (err) {
      const message = readError(err, 'Embedding failed.');
      setError(message);
      addLog('error', message, err.response?.data || null);
    } finally {
      setEmbedding(false);
    }
  };

  const runSearch = async (event) => {
    event?.preventDefault();
    if (!canUseSemantic || !searchQuery.trim()) return;

    setSearching(true);
    setError('');
    addLog('info', 'Semantic search started', { dataset_id: selectedId, top_k: topK });

    try {
      const startedAt = performance.now();
      const result = await searchDataset(selectedId, searchQuery.trim(), topK);
      const elapsedMs = Math.round(performance.now() - startedAt);
      setSearchResults(result.results || []);
      addLog('success', 'Semantic search completed', {
        elapsed_ms: elapsedMs,
        matches: result.results?.length || 0,
        namespace: selectedId,
      });
    } catch (err) {
      const message = readError(err, 'Search failed.');
      setError(message);
      addLog('error', message, err.response?.data || null);
    } finally {
      setSearching(false);
    }
  };

  const runQA = async (event) => {
    event?.preventDefault();
    if (!canUseSemantic || !question.trim()) return;

    setAnswering(true);
    setError('');
    setQaResult(null);
    addLog('info', 'QA retrieval started', { dataset_id: selectedId, top_k: topK });

    try {
      const startedAt = performance.now();
      const result = await askDatasetQuestion(selectedId, question.trim(), topK);
      const elapsedMs = Math.round(performance.now() - startedAt);
      setQaResult(result);
      addLog(result.mode === 'llm' ? 'success' : 'warn', `QA completed in ${result.mode} mode`, {
        elapsed_ms: elapsedMs,
        supporting_rows: result.supporting_rows?.length || 0,
      });
    } catch (err) {
      const message = readError(err, 'QA failed.');
      setError(message);
      addLog('error', message, err.response?.data || null);
    } finally {
      setAnswering(false);
    }
  };

  return (
    <main className="qa-page">
      <header className="qa-header">
        <div>
          <h1>
            <MessageSquareText size={24} />
            Semantic Q&A
          </h1>
          <p>{selectedDataset?.original_filename || 'Select a cleaned dataset.'}</p>
        </div>
        <button className="btn btn-secondary" type="button" onClick={handleRefresh} disabled={loadingDatasets}>
          {loadingDatasets ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
          Refresh
        </button>
      </header>

      <section className="qa-controls card">
        <label className="qa-field qa-field--dataset">
          <span>
            <Database size={15} />
            Dataset
          </span>
          <div className="qa-select-wrap">
            <select
              value={selectedId}
              onChange={(event) => handleDatasetChange(event.target.value)}
              disabled={loadingDatasets || embedding || searching || answering}
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

        <label className="qa-field qa-field--topk">
          <span>
            <Hash size={15} />
            Top K
          </span>
          <div className="qa-select-wrap">
            <select
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              disabled={embedding || searching || answering}
            >
              {[3, 5, 8, 10].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <ChevronDown size={16} />
          </div>
        </label>

        <div className="qa-embed-action-group">
          <button
            className="btn btn-primary"
            type="button"
            onClick={runEmbedding}
            disabled={!selectedId || embedding || (selectedDataset?.status !== 'analyzed' && selectedDataset?.status !== 'embedded')}
          >
            {embedding ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            Embed
          </button>
          {selectedId && selectedDataset?.status !== 'analyzed' && selectedDataset?.status !== 'embedded' && (
            <span className="qa-hint">Analyze in Dashboard first</span>
          )}
        </div>
      </section>

      {error && (
        <div className="qa-message qa-message--error card">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      <section className="qa-status-grid">
        <StatusTile 
          icon={ShieldCheck} 
          label="Embedding" 
          value={displayStatus === 'processing' ? `${Math.round(displayProgress * 100)}%` : formatStatus(displayStatus)} 
          tone={statusTone(displayStatus)} 
        />
        <StatusTile icon={Cpu} label="Dimension" value={displayDimension || 'Pending'} tone={displayDimension ? 'info' : 'muted'} />
        <StatusTile icon={Zap} label="Model" value={displayModel || 'Configured in API'} tone="info" />
        <StatusTile icon={Database} label="Index" value={displayIndex} tone="info" />
        <StatusTile icon={Hash} label="Namespace" value={selectedId || 'None'} tone={selectedId ? 'success' : 'muted'} />
        <StatusTile icon={FileSearch} label="Vectors" value={displayCount.toLocaleString()} tone={displayCount ? 'success' : 'muted'} />
      </section>

      <section className="qa-workbench">
        <div className="qa-column">
          <form className="qa-panel card" onSubmit={runSearch}>
            <div className="qa-panel-header">
              <span className="qa-panel-icon qa-panel-icon--search">
                <Search size={17} />
              </span>
              <h2>Search</h2>
            </div>
            <div className="qa-input-row">
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search retrieved meaning"
                disabled={!canUseSemantic || searching}
              />
              <button className="btn btn-secondary" type="submit" disabled={!canUseSemantic || !searchQuery.trim() || searching}>
                {searching ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
                Search
              </button>
            </div>
          </form>

          <ResultsList title="Search results" rows={searchResults} emptyIcon={Search} emptyText="No matches yet." />
        </div>

        <div className="qa-column">
          <form className="qa-panel card" onSubmit={runQA}>
            <div className="qa-panel-header">
              <span className="qa-panel-icon qa-panel-icon--qa">
                <Bot size={17} />
              </span>
              <h2>Ask</h2>
              {qaResult?.mode && <span className={`qa-mode qa-mode--${qaResult.mode}`}>{qaResult.mode}</span>}
            </div>
            <div className="qa-input-row qa-input-row--textarea">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask a question about this dataset"
                disabled={!canUseSemantic || answering}
                rows={3}
              />
              <button className="btn btn-primary" type="submit" disabled={!canUseSemantic || !question.trim() || answering}>
                {answering ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                Ask
              </button>
            </div>
          </form>

          <AnswerPanel result={qaResult} answering={answering} />
        </div>
      </section>

      <section className="qa-log card">
        <div className="qa-log-header">
          <TerminalSquare size={17} />
          <h2>Operation Log</h2>
        </div>
        {logs.length === 0 ? (
          <p className="qa-log-empty">No events yet.</p>
        ) : (
          <div className="qa-log-list">
            {logs.map((entry) => (
              <article className={`qa-log-entry qa-log-entry--${entry.level}`} key={entry.id}>
                <span>{entry.time}</span>
                <strong>{entry.message}</strong>
                {entry.details && <code>{compactJson(entry.details)}</code>}
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function StatusTile({ icon: Icon, label, value, tone }) {
  return (
    <article className={`qa-status-tile card qa-status-tile--${tone}`}>
      <span>
        <Icon size={15} />
        {label}
      </span>
      <strong title={String(value)}>{value}</strong>
    </article>
  );
}

function ResultsList({ title, rows, emptyIcon: EmptyIcon, emptyText }) {
  return (
    <section className="qa-results">
      <div className="qa-section-heading">
        <h2>{title}</h2>
        <span>{rows.length}</span>
      </div>
      {rows.length === 0 ? (
        <div className="qa-empty card">
          <EmptyIcon size={24} />
          <span>{emptyText}</span>
        </div>
      ) : (
        <div className="qa-result-list">
          {rows.map((row, index) => (
            <ResultCard row={row} index={index} key={row.id || index} />
          ))}
        </div>
      )}
    </section>
  );
}

function AnswerPanel({ result, answering }) {
  if (answering) {
    return (
      <div className="qa-answer card qa-answer--loading">
        <Loader2 size={26} className="animate-spin" />
        <span>Generating answer...</span>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="qa-answer card qa-answer--empty">
        <Bot size={24} />
        <span>No answer yet.</span>
      </div>
    );
  }

  return (
    <section className="qa-answer card">
      <div className="qa-answer-header">
        <span className={`qa-mode qa-mode--${result.mode}`}>{result.mode}</span>
        <span>{result.supporting_rows?.length || 0} rows</span>
      </div>
      <p>{result.answer}</p>
      <ResultsList
        title="Supporting rows"
        rows={result.supporting_rows || []}
        emptyIcon={FileSearch}
        emptyText="No supporting rows returned."
      />
    </section>
  );
}

function ResultCard({ row, index }) {
  const metadata = row.metadata || {};
  return (
    <article className="qa-result-card card">
      <div className="qa-result-topline">
        <span>#{index + 1}</span>
        <strong>{formatScore(row.score)}</strong>
      </div>
      <p>{row.text || metadata.text || 'No text returned.'}</p>
      <div className="qa-result-meta">
        <MetaPill label="row" value={metadata.row_id ?? 'N/A'} />
        <MetaPill label="sentiment" value={metadata.sentiment || 'N/A'} />
        <MetaPill label="engagement" value={formatNumber(metadata.engagement)} />
        <MetaPill label="time" value={metadata.timestamp || 'N/A'} />
      </div>
    </article>
  );
}

function MetaPill({ label, value }) {
  return (
    <span className="qa-meta-pill">
      <em>{label}</em>
      {value}
    </span>
  );
}

function readError(err, fallback) {
  return err?.response?.data?.detail || err?.message || fallback;
}

function formatStatus(status) {
  return String(status || 'not_started').replace(/_/g, ' ');
}

function statusTone(status) {
  if (status === 'completed') return 'success';
  if (status === 'processing') return 'info';
  if (status === 'failed') return 'error';
  return 'muted';
}

function formatScore(score) {
  const value = Number(score);
  return Number.isFinite(value) ? value.toFixed(3) : '0.000';
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : 'N/A';
}

function compactJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
