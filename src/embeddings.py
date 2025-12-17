import json
import os
import logging
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Robust Pathing
# src/embeddings.py is in src/, so project root is one level up
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_FILE = os.path.join(DATA_DIR, "assessments.json")
INDEX_FILE = os.path.join(PROCESSED_DIR, "assessments.index")
MAPPING_FILE = os.path.join(PROCESSED_DIR, "mapping.pkl")

# Use a lightweight model for speed and assignments
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

class EmbeddingEngine:
    def __init__(self):
        logging.info(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.assessments = []

    def load_data(self):
        # Try full JSON first
        if os.path.exists(INPUT_FILE):
             try:
                 with open(INPUT_FILE, 'r') as f:
                    self.assessments = json.load(f)
                 if len(self.assessments) > 0:
                     logging.info(f"Loaded {len(self.assessments)} assessments from JSON")
                     return True
                 else:
                     logging.warning("JSON file empty, trying JSONL...")
             except: pass
        
        # Try partial JSONL
        partial_file = INPUT_FILE.replace("assessments.json", "assessments_partial.jsonl")
        if os.path.exists(partial_file):
            self.assessments = []
            with open(partial_file, 'r') as f:
                for line in f:
                    try:
                        self.assessments.append(json.loads(line))
                    except: pass
            logging.info(f"Loaded {len(self.assessments)} assessments from JSONL")
            return len(self.assessments) > 0

        logging.error(f"Input file not found: {INPUT_FILE} or {partial_file}")
        return False

    def create_index(self):
        if not self.assessments:
            logging.warning("No assessments to index.")
            return

        # Prepare text for embedding
        # Combine Name + Description + Type for rich context
        corpus = [
            f"{item.get('name', '')} {item.get('description', '')} {item.get('type', '')}"
            for item in self.assessments
        ]

        logging.info("Generating embeddings...")
        embeddings = self.model.encode(corpus, convert_to_numpy=True)
        
        # Normalize for cosine similarity (InnerProduct with normalized vectors = Cosine)
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        logging.info(f"Index created with {self.index.ntotal} vectors.")
        
        self.save_index()

    def save_index(self):
        if not os.path.exists(PROCESSED_DIR):
            os.makedirs(PROCESSED_DIR)
            
        faiss.write_index(self.index, INDEX_FILE)
        with open(MAPPING_FILE, 'wb') as f:
            pickle.dump(self.assessments, f)
        logging.info(f"Index and mapping saved to {PROCESSED_DIR}")

    def load_index(self):
        if not os.path.exists(INDEX_FILE) or not os.path.exists(MAPPING_FILE):
            logging.error("Index or mapping file missing.")
            return False
            
        self.index = faiss.read_index(INDEX_FILE)
        with open(MAPPING_FILE, 'rb') as f:
            self.assessments = pickle.load(f)
        logging.info("Index and mapping loaded.")
        return True

    def search(self, query, k=10):
        if not self.index:
            loaded = self.load_index()
            if not loaded:
                logging.info("Index load failed. Attempting to create new index...")
                if self.load_data():
                    self.create_index()
                else:
                    logging.error("Failed to load data for indexing.")
                    return []
        
        if not self.index:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.assessments):
                item = self.assessments[idx].copy()
                item['score'] = float(distances[0][i])
                results.append(item)
                
        return results

if __name__ == "__main__":
    engine = EmbeddingEngine()
    if engine.load_data():
        engine.create_index()
