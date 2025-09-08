import os
import sys
import subprocess
from dotenv import load_dotenv


# Check if required libraries are installed and install if not
required_libraries = ['pandas', 'python-docx', 'bert-score', 'torch']
for lib in required_libraries:
    try:
        __import__(lib.replace('-', '_'))
    except ImportError:
        print(f"Installing {lib}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])


# --- Paths and Directories ---
# Using environment variables is a best practice for production
DATA_DIR = os.getenv("DATA_DIR", "./data/")
os.makedirs(DATA_DIR, exist_ok=True)

# Data File Paths
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "Teacher one.json")
RAG_DATA_PATH = os.path.join(DATA_DIR, "rag_corpus.json")
EVAL_DATA_PATH = os.path.join(DATA_DIR, "evaluation_data.json")
PROCESSED_EVAL_DATA_PATH = os.path.join(DATA_DIR, "processed_evaluation_data.json")
GENERATED_RESPONSES_PATH = os.path.join(DATA_DIR, "generated_responses.json")
EXTRA_DOCS_PATH = os.path.join(DATA_DIR, "Rag Docs.docx")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
CSV_REPORT_PATH = os.path.join(DATA_DIR, "scoring_report.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.json")
TEST_RESPONSES_PATH = os.path.join(DATA_DIR, "test_generated_responses.json")
TEST_CSV_REPORT_PATH = os.path.join(DATA_DIR, "test_scoring_report.csv")


# --- Model Configurations ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GROQ_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- API Keys ---
# Use environment variables for sensitive info in a real production environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "GROQ_API_KEY")

# --- RAG Parameters ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_TEACHER_EXAMPLES = 3
TOP_K_FACTUAL_CHUNKS = 2

# --- LLM Call Parameters ---
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 16  # seconds
CALL_DELAY_BETWEEN_RECORDS = 0.5  # seconds