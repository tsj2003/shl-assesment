# SHL Assessment Recommendation System

## Overview
This project serves as a Generative AI-powered recommendation engine for SHL assessments. It accepts job descriptions or natural language queries and returns the most relevant assessments from the SHL catalog. The system covers over 380 individual test solutions, utilizing a Retrieval-Augmented Generation (RAG) approach with vector embeddings.

## Key Features
- **Data Collection**: Custom scraper for the SHL Product Catalog.
- **2-Stage RAG Pipeline**: 
  1. **Retrieval**: FAISS vector search (`all-MiniLM-L6-v2`) for high recall (Top 50).
  2. **Reranking**: Cross-Encoder (`ms-marco-MiniLM`) for high precision (Top 10).
- **Hybrid Logic**: Balances technical and behavioral assessments based on query intent.
- **Evaluation**: Validated against a training dataset using Mean Recall@10 metric.
- **Interface**: REST API (FastAPI) and a lightweight web UI.

## Project Structure
```text
.
├── data/
│   ├── raw/             # Scraped JSON data
│   └── processed/       # FAISS index and mappings
├── src/
│   ├── scraper.py       # Data collection script
│   ├── embeddings.py    # Vector encoding and indexing
│   ├── recommender.py   # Core recommendation logic
│   ├── main.py          # FastAPI application
│   └── evaluation.py    # Recall metrics and prediction generation
├── frontend/
│   └── index.html       # Web Interface
└── requirements.txt     # Dependencies
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd shl_assignment
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Collection
First, scrape the latest catalog data:
```bash
python src/scraper.py
```
*Note: This process may take a few minutes.*

### 2. Generate Embeddings
Build the vector index from scraped data:
```bash
python src/embeddings.py
```

### 3. Run the Server
Start the API backend:
```bash
python src/main.py
```
The server will run at `http://0.0.0.0:8000`.

### 4. Access Frontend
Open `frontend/index.html` in any web browser to use the interface.

## Evaluation
To run the evaluation script and generate the submission CSV:
```bash
python src/evaluation.py
```
This will output the `Mean Recall@10` score and create `submission.csv`.

## API Endpoints
- `GET /health`: Health check.
- `POST /recommend`: 
  - Body: `{"query": "..."}`
  - Returns: List of recommended assessments.

## Technologies
- **Language**: Python 3.10+
- **Backend**: FastAPI
- **ML/AI**: Sentence-Transformers (HuggingFace), FAISS
- **Web**: Vanilla HTML/JS/CSS

