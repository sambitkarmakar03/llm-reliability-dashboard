import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the trained probability model
MODEL_PATH = os.path.join(BASE_DIR, "scoring_model.pkl")

# Exact feature order required by the .pkl model for inference
ML_FEATURES = [
    "evidence_max", 
    "evidence_mean", 
    "consistency",
    "p_contradiction_max", 
    "p_contradiction_mean",
    "p_entailment_max", 
    "p_entailment_mean",
    "refusal_fraction", 
    "verification_failed",
    "retrieval_relevance", 
    "answer_length_words"
]

class Settings:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    DEVICE = os.getenv("DEVICE", "cpu")

    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

    # Bounds on the retrieval pipeline so a slow/hanging search backend
    # can't make total request latency unpredictable.
    DDGS_TIMEOUT = int(os.getenv("DDGS_TIMEOUT", "6"))       # per search attempt
    DDGS_RETRIES = int(os.getenv("DDGS_RETRIES", "2"))       # attempts before Wikipedia fallback
    RETRIEVAL_TIMEOUT = float(os.getenv("RETRIEVAL_TIMEOUT", "20"))  # hard ceiling, from when we start waiting

settings = Settings()