# 🚀 TextLens AI  
### Transform Unstructured Text into Actionable Intelligence

TextLens AI is an end-to-end AI-powered platform that converts raw, messy text data (reviews, comments, feedback) into structured insights, semantic search, and conversational analytics.

It combines **NLP preprocessing, vector search, and LLM-based reasoning (Gemini)** to help users explore, analyze, and extract value from large-scale unstructured datasets—without requiring data science expertise.

---

# ✨ Features

## 📥 Smart Data Ingestion
- Upload CSV datasets with no strict schema
- Automatic text column detection
- Dataset validation and quality reporting

## 🧹 Advanced Preprocessing Pipeline
- Noise removal (URLs, punctuation, whitespace)
- Emoji → semantic conversion 
- Slang and contraction normalization
- Language detection (multilingual-ready)
- Duplicate-aware processing with frequency mapping

## 🧠 Semantic Search (Vector DB)
- Embedding-based similarity search
- Optimized for large datasets (50K–500K+ rows)
- Fast and relevant context retrieval

## 💬 Conversational AI (RAG with Gemini)
- Ask natural language questions about your dataset
- Context-aware answers grounded in your data
- Source-backed responses to reduce hallucinations

## 📊 Analytics Dashboard
- Sentiment distribution
- Top issues and complaint trends
- Frequency-based insights
- Dataset quality metrics

## 🤖 Automated Insights
- FAQ generation
- Trend detection
- Summarization of large datasets

---

# 🧠 System Architecture
React Frontend
↓
FastAPI Backend
↓
Preprocessing Pipeline
↓
Embeddings + Vector DB (ChromaDB)
↓
RAG Layer (Gemini API)
↓
Insights + Dashboard

---

# 🏗️ Tech Stack

## Frontend
- React (Vite)
- Tailwind CSS
- Chart.js / Recharts

## Backend
- FastAPI (Python)

## AI / NLP
- Gemini API (LLM)
- Sentence Transformers (Embeddings)
- spaCy, langdetect, emoji

## Vector Database
- ChromaDB (local, scalable for MVP)

---

# ⚙️ Getting Started

## 🔧 Prerequisites
- Python 3.10+
- Node.js 18+
- Git

---

## 🚀 Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

🔄 Data Pipeline Overview
CSV Upload
   ↓
Validation + Column Detection
   ↓
Basic Cleaning (Phase 1)
   ↓
Text Normalization (Phase 2)
   ↓
Language Detection
   ↓
Frequency Mapping (Duplicate Handling)
   ↓
Embeddings + Vector Storage
   ↓
RAG + Insights

---

##📊 Example Use Cases
Customer feedback analysis
Product review mining
Social media sentiment tracking
Support ticket analysis
Market research insights
⚡ Performance Considerations
Optimized for datasets up to 500K+ rows
Uses frequency-based deduplication to reduce embedding load
Batch processing for efficient embedding generation
Separate logic for:
Semantic retrieval (unique texts)
Analytics (frequency-aware)

---

##🧪 Current Status
✅ CSV upload & validation
✅ Dataset quality analysis
🔄 Preprocessing pipeline (in progress)
🔄 Vector DB + embeddings integration
🔄 RAG (Gemini) integration
🔄 Dashboard development
🎯 Future Enhancements
Multilingual translation pipeline
Real-time data ingestion (APIs, streams)
Advanced topic modeling
Exportable reports (PDF/CSV)
Cloud deployment and scaling

---

🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements.

---

💡 Vision

Make unstructured text as easy to analyze and query as structured data—powered by AI.
