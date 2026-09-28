import logging
import joblib
import numpy as np
from functools import lru_cache
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from core.config import MODEL_PATH, ML_FEATURES

logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
nli_analyzer = pipeline("text-classification", model="roberta-large-mnli", framework="pt")

# Load the ML scoring model once during startup
try:
    scoring_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    logger.warning(f"Model not found at {MODEL_PATH}. ML scoring will fallback to 0.0.")
    scoring_model = None

def calculate_ml_score(record_dict: dict) -> float:
    """Extracts features in the strict order defined in config and returns the ML probability."""
    if scoring_model is None:
        return 0.0
    
    # Extract features, defaulting to 0.0 if missing
    X_input = np.array([[float(record_dict.get(f, 0.0)) for f in ML_FEATURES]])
    try:
        return float(scoring_model.predict_proba(X_input)[0, 1])
    except Exception as e:
        logger.error(f"ML scoring inference error: {e}")
        return 0.0

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product < 1e-8:
        return 0.0
    return float(np.dot(a, b) / norm_product)

@lru_cache(maxsize=256)
def get_embedding(text: str) -> np.ndarray:
    return embedding_model.encode(text, convert_to_numpy=True)

def semantic_similarity(model_output: str, reference: Optional[str]) -> float:
    if not reference:
        return 0.0
    try:
        emb1, emb2 = get_embedding(model_output), get_embedding(reference)
        return cosine_similarity(emb1, emb2)
    except Exception:
        return 0.5

def evidence_score(evidence_passages: List[Dict[str, Any]]) -> float:
    if not evidence_passages:
        return 0.0
    return max([p.get("score", 0.0) for p in evidence_passages])

def self_consistency(outputs: List[str]) -> float:
    if len(outputs) < 2:
        return 1.0
    embeddings = [get_embedding(o) for o in outputs]
    sims = [
        cosine_similarity(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    return float(np.mean(sims)) if sims else 1.0

def get_nli_metrics(response: str, evidence_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates text against passages for ML probabilities and the legacy hallucination penalty."""
    metrics = {
        "p_contradiction_max": 0.0, "p_contradiction_mean": 0.0,
        "p_entailment_max": 0.0, "p_entailment_mean": 0.0,
        "h_penalty": 1.0 # Default to 1.0 if no evidence
    }
    if not evidence_passages:
        return metrics

    c_probs, e_probs, penalties = [], [], []
    hypothesis = response[:200]

    for p in evidence_passages:
        try:
            premise = p["text"][:400]
            
            # Use tuple format which is universally supported by the text-classification pipeline
            result = nli_analyzer({"text": premise, "text_pair": hypothesis}, truncation=True)
            
            # Handle both list and dict return types safely
            if isinstance(result, list):
                result = result[0]
                
            label, score = result["label"].upper(), result["score"]

            # Map the labels properly
            if label in ["LABEL_0", "CONTRADICTION"]:
                c_probs.append(score)
                e_probs.append(0.0)
                penalties.append(score * 0.9)
            elif label in ["LABEL_1", "NEUTRAL"]:
                c_probs.append(0.0)
                e_probs.append(0.0)
                penalties.append(score * 0.2)
            elif label in ["LABEL_2", "ENTAILMENT"]:
                e_probs.append(score)
                c_probs.append(0.0)
                penalties.append(0.0)
            else:
                c_probs.append(0.0)
                e_probs.append(0.0)
                penalties.append(0.0)

        except Exception as e:
            logger.error(f"NLI check failed for passage {p.get('url')}: {e}")
            c_probs.append(0.0)
            e_probs.append(0.0)
            penalties.append(0.1)

    if c_probs:
        metrics["p_contradiction_max"] = max(c_probs)
        metrics["p_contradiction_mean"] = float(np.mean(c_probs))
    if e_probs:
        metrics["p_entailment_max"] = max(e_probs)
        metrics["p_entailment_mean"] = float(np.mean(e_probs))
    if penalties:
        metrics["h_penalty"] = float(np.mean(penalties))

    return metrics