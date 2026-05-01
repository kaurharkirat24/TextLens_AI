/**
 * TextLens AI — API Client
 *
 * Centralised HTTP client for all backend calls.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120_000,   // 2 min — uploads can be large
});

// ── Ingestion ────────────────────────────────────────────────────────────────

/**
 * Upload a file and run the ingestion pipeline.
 * @param {File} file
 * @param {string|null} textColumn - Optional text column override
 * @param {function} onProgress - Axios progress callback
 * @returns {Promise<{dataset_id: string, report: object}>}
 */
export async function uploadFile(file, textColumn = null, onProgress = null) {
  const formData = new FormData();
  formData.append('file', file);

  const params = {};
  if (textColumn) params.text_column = textColumn;

  const res = await api.post('/upload', formData, {
    params,
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
  });
  return res.data;
}

/**
 * List all uploaded datasets.
 * @returns {Promise<{datasets: Array, total: number}>}
 */
export async function getDatasets() {
  const res = await api.get('/datasets');
  return res.data;
}

/**
 * Get a preview of a dataset's rows.
 * @param {string} datasetId
 * @param {number} limit
 * @returns {Promise<{columns: string[], rows: object[], total_rows: number, showing: number}>}
 */
export async function getDatasetPreview(datasetId, limit = 50) {
  const res = await api.get(`/datasets/${datasetId}/preview`, { params: { limit } });
  return res.data;
}

/**
 * Health check.
 * @returns {Promise<{status: string, service: string}>}
 */
export async function checkHealth() {
  const res = await api.get('/health');
  return res.data;
}

export default api;
