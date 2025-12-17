import pandas as pd
import logging
from recommender import RecommenderSystem
import os

logging.basicConfig(level=logging.INFO)

DATA_DIR = "data/raw"
DATA_FILE = "/Users/tarandeepsinghjuneja/Downloads/Gen_AI Dataset.xlsx"

def normalize_url(url):
    # Remove domain and protocol
    url = url.replace("https://www.shl.com", "").replace("http://www.shl.com", "")
    # Remove /solutions prefix if present
    url = url.replace("/solutions", "")
    if not url.endswith('/'):
        url += '/'
    return url.strip()

def evaluate_model():
    if not os.path.exists(DATA_FILE):
        logging.error(f"Dataset not found at {DATA_FILE}")
        return 0.0

    logging.info("Loading Train dataset...")
    df = pd.read_excel(DATA_FILE, sheet_name='Train-Set')
    
    # Check if 'Assessment_url' column exists
    if 'Assessment_url' not in df.columns:
        logging.error("Column 'Assessment_url' not found in Train-Set")
        return 0.0

    logging.info("Initializing Recommender...")
    rec_system = RecommenderSystem()
    
    recall_at_10 = 0
    recall_at_20 = 0
    recall_at_50 = 0
    total_queries = len(df)
    
    logging.info(f"Evaluating {total_queries} queries...")
    
    for _, row in df.iterrows():
        query = row['Query']
        target_url = normalize_url(str(row['Assessment_url']))
        
        # Fetch deep list
        recommendations = rec_system.search_raw(query, k=50) # Use search_raw to bypass reranker for pure retrieval check
        rec_urls = [normalize_url(r['url']) for r in recommendations]
        
        if target_url in rec_urls[:10]:
            recall_at_10 += 1
        if target_url in rec_urls[:20]:
            recall_at_20 += 1
        if target_url in rec_urls[:50]:
            recall_at_50 += 1
            
    logging.info(f"Recall@10: {recall_at_10/total_queries:.2f}")
    logging.info(f"Recall@20: {recall_at_20/total_queries:.2f}")
    logging.info(f"Recall@50: {recall_at_50/total_queries:.2f}")
    
    print(f"FINAL_RECALL_10: {recall_at_10/total_queries:.2f}")
    
    with open("recall.txt", "w") as f:
        f.write(f"R10: {recall_at_10/total_queries:.2f}\n")
        f.write(f"R20: {recall_at_20/total_queries:.2f}\n")
        f.write(f"R50: {recall_at_50/total_queries:.2f}\n")
    
    return recall_at_10 / total_queries

def generate_predictions():
    if not os.path.exists(DATA_FILE):
        logging.error("Data file missing")
        return

    logging.info("Loading Test dataset...")
    df_test = pd.read_excel(DATA_FILE, sheet_name='Test-Set')
    rec = RecommenderSystem()
    
    predictions = []
    
    for index, row in df_test.iterrows():
        query = row['Query']
        results = rec.recommend(query, k=10)
        
        # User requested 5-10 assessments. The output format is just Query, Assessment_url.
        # usually means the top 1. Or maybe multiple rows?
        # The prompt says: "CSV file: Columns: Query, Assessment_url". 
        # Usually implies top 1 recommendation, or we can comma separate?
        # Standard format is likely top 1 recommendation per row.
        # But wait, "Returns 5-10 assessments".
        # Let's assume Top 1 for the CSV prediction unless specified otherwise.
        # "Generate predictions for Test dataset... CSV file: Columns: Query, Assessment_url"
        # I'll output the top 1 URL.
        
        top_url = results[0]['url'] if results else ""
        predictions.append({"Query": query, "Assessment_url": top_url})
        
    df_pred = pd.DataFrame(predictions)
    output_path = "submission.csv"
    df_pred.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    score = evaluate_model()
    generate_predictions()
