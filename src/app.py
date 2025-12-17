import streamlit as st
from recommender import RecommenderSystem
import time

# Page Config
st.set_page_config(page_title="SHL Recommender", page_icon="🧩", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .card {
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 5px;
        font-weight: bold;
    }
    .tag-tech { background-color: #e3f2fd; color: #1565c0; }
    .tag-behav { background-color: #f3e5f5; color: #7b1fa2; }
    .score {
        float: right;
        color: #666;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize System (Cached)
@st.cache_resource
def load_recommender():
    return RecommenderSystem()

st.title("🧩 SHL Assessment Recommender")
st.markdown("Enter a role, job description, or skill to find the best SHL assessments.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("This GenAI tool uses Semantic Search + Cross-Encoder Reranking to find the most relevant tests from the SHL Catalog.")
    st.markdown("---")
    st.markdown("**Core Specs:**")
    st.markdown("- **Model**: `all-MiniLM-L6-v2`")
    st.markdown("- **Reranker**: `ms-marco`")
    st.markdown("- **Items**: 500+ Scraped")

# Main Interface
query = st.text_input("Search for an assessment...", placeholder="e.g. 'Senior Java Developer with leadership skills'")

if st.button("Get Recommendations", type="primary"):
    if query:
        rec = load_recommender()
        with st.spinner("Analyzing semantic intent and reranking candidates..."):
            start_time = time.time()
            results = rec.recommend(query, k=10)
            duration = time.time() - start_time
        
        st.success(f"Found {len(results)} relevant assessments in {duration:.2f}s")
        
        for r in results:
            # Determine tag style
            tag_class = "tag-tech" if "Technical" in r.get('type', '') else "tag-behav"
            score = r.get('rerank_score', 0)
            
            st.markdown(f"""
            <div class="card">
                <h3><a href="{r['url']}" target="_blank" style="text-decoration:none; color:#333;">{r['name']}</a></h3>
                <div>
                    <span class="tag {tag_class}">{r.get('type', 'General')}</span>
                    <span class="score">Relevance: {score:.2f}</span>
                </div>
                <p style="margin-top:10px; color:#555;">{r['description'][:200]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning("Please enter a query first.")
