from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from fastapi.middleware.cors import CORSMiddleware
from recommender import RecommenderSystem

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SHL Recommender API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Recommender
# Global variable to hold the system
rec_system = None

@app.on_event("startup")
def startup_event():
    global rec_system
    try:
        logging.info("Initializing Recommender System...")
        rec_system = RecommenderSystem()
        logging.info("Recommender System initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize recommender: {e}")

class QueryRequest(BaseModel):
    query: str

class AssessmentResponse(BaseModel):
    name: str
    url: str
    description: str = ""
    type: str = "Unknown"
    score: Optional[float] = 0.0

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": rec_system is not None}

@app.post("/recommend", response_model=List[AssessmentResponse])
def get_recommendations(request: QueryRequest):
    if not rec_system:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    results = rec_system.recommend(request.query)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
