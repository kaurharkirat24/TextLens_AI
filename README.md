# 🚀 TextLens AI

### Transform Unstructured Text into Actionable Intelligence

TextLens AI is an end-to-end AI-powered platform that converts raw, messy text data (reviews, comments, feedback) into structured insights, semantic search, and conversational analytics.

It combines NLP preprocessing, vector search, and LLM-based reasoning (Gemini) to help users explore, analyze, and extract value from large-scale unstructured datasets—without requiring data science expertise.

---

## ✨ Features

### 📥 Smart Data Ingestion

- Upload CSV datasets with no strict schema
- Automatic text column detection
- Dataset validation and quality reporting

### 🧹 Advanced Preprocessing Pipeline

- Noise removal (URLs, punctuation, whitespace)
- Emoji → semantic conversion
- Slang and contraction normalization
- Language detection (multilingual-ready)
- Duplicate-aware processing with frequency mapping

### 🧠 Semantic Search (Vector DB)

- Embedding-based similarity search
- Optimized for large datasets (50K–500K+ rows)
- Fast and relevant context retrieval

### 💬 Conversational AI (RAG with Gemini)

- Ask natural language questions about your dataset
- Context-aware answers grounded in your data
- Source-backed responses to reduce hallucinations

### 📊 Analytics Dashboard

- Sentiment distribution
- Top issues and complaint trends
- Frequency-based insights
- Dataset quality metrics

### 🤖 Automated Insights

- FAQ generation
- Trend detection
- Summarization of large datasets

---

## 🧠 System Architecture

React Frontend (Vite, Tailwind, Recharts)
↓
FastAPI Backend (Data Cleaning, Routing, Q&A)
↓
Remote Colab GPU Worker (SentenceTransformers Embeddings)
↓
Vector DB (Pinecone)
↓
RAG Layer (Gemini API with Retrieval Intelligence)
↓
Insights + Dashboard

---

## 🏗️ Tech Stack

### Frontend

- React (Vite)
- Tailwind CSS
- Chart.js / Recharts

### Backend

- FastAPI (Python)

### AI / NLP

- Gemini API (LLM)
- Sentence Transformers (Embeddings)
- spaCy, langdetect, emoji

### Vector Database

- Pinecone (cloud-native vector search)
- gRPC data-plane for high-throughput upserts

---

## ⚙️ Getting Started

### 🔧 Prerequisites

- Python 3.10+
- Node.js 18+
- Git

---

### 🚀 Backend Setup

1. Create and activate a virtual environment

```bash
cd backend
python -m venv venv
# Windows (PowerShell)
venv\\Scripts\\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

2. Install Python dependencies and run the app

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Gemini API key, Pinecone credentials, and backend worker token.

The backend provides CSV upload, preprocessing, semantic chunking, and the Retrieval Intelligence Q&A endpoints.

---

### 🖥️ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite to interact with the dashboard and upload flows.

---

## 🔄 Data Pipeline Overview

1. **CSV Upload** → Validation & Automatic Column Detection
2. **Analysis Pipeline** → Text Cleaning, Sentiment Enrichment, Frequency Mapping
3. **Semantic Chunking** → Documents are sliced into JSONL chunks
4. **Remote GPU Embedding** → A Google Colab worker claims the job, embeds the text using `SentenceTransformers`, and async-upserts to Pinecone.
5. **Insights** → Dashboard renders dynamic charts based on analysis facts.
6. **Retrieval Intelligence Q&A** → Gemini answers user queries, intelligently routed between factual RAG, semantic exploration, and dataset-wide aggregation.
---

## 📊 Example Use Cases

- Customer feedback analysis
- Product review mining
- Social media sentiment tracking
- Support ticket analysis
- Market research insights

### ⚡ Performance Considerations

- Optimized for datasets up to 500K+ rows
- Uses frequency-based deduplication to reduce embedding load
- Batch processing for efficient embedding generation
- Separate logic for semantic retrieval (unique texts) and analytics (frequency-aware)

---

## 🧪 Current Status

- ✅ CSV upload, validation, and schema detection
- ✅ Data cleaning & sentiment enrichment pipeline
- ✅ Analytics Dashboard with dynamic charts
- ✅ Semantic Chunking & Remote Colab GPU Embedding Worker
- ✅ Pinecone Vector DB integration (with gRPC and async batched upserts)
- ✅ RAG (Gemini) integration
- ✅ Retrieval Intelligence (Intelligently routing between factual RAG, exploratory semantic search, and dataset aggregation)
- 🔄 Advanced Insights (Topic clustering, FAQ extraction)


### 🎯 Future Enhancements

- Multilingual translation pipeline
- Real-time data ingestion (APIs, streams)
- Advanced topic modeling
- Exportable reports (PDF/CSV)
- Cloud deployment and scaling

---

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request describing your changes.

---

## 💡 Vision

Make unstructured text as easy to analyze and query as structured data—powered by AI.
