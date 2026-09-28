import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np

from core.llm_client import generate_response, is_generation_failure
from core.refusal import is_refusal
from core.retriever import gather_candidate_passages, rank_passages
from core.scorer import (
    semantic_similarity,
    evidence_score,
    self_consistency,
    get_nli_metrics,
    calculate_ml_score,
)
from core.config import settings

app = FastAPI(title="LLM Reliability Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=False,   
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    prompt: str
    reference: Optional[str] = None

class EvidenceSource(BaseModel):
    url: str
    text: str
    score: float

class EvaluationResponse(BaseModel):
    response: str              
    model_response: str        
    answer_declined: bool      
    fallback_answer: Optional[str] = None
    similarity: Optional[float] = None
    evidence: Optional[float] = None
    consistency: Optional[float] = None
    hallucination_penalty: Optional[float] = None 
    final_score: Optional[float] = None
    evidence_sources: List[EvidenceSource] = []
    samples_used: int
    generation_failed: bool
    verification_failed: bool

def build_fallback_answer(evidence_passages: List[Dict[str, Any]], max_len: int = 500) -> Optional[str]:
    if not evidence_passages:
        return None
    return evidence_passages[0]["text"][:max_len]

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint(req: EvaluationRequest):

    passage_gather_task = asyncio.create_task(gather_candidate_passages(req.prompt))
    gen_tasks = [generate_response(req.prompt) for _ in range(3)]
    gen_results = await asyncio.gather(*gen_tasks)
    outputs = [r[0] for r in gen_results]

    successful_outputs = [o for o in outputs if not is_generation_failure(o)]
    samples_used = len(successful_outputs)

    if samples_used == 0:
        passage_gather_task.cancel()
        return EvaluationResponse(
            response="The model failed to respond, and no answer could be generated.",
            model_response=outputs[0],
            answer_declined=False,
            samples_used=0,
            generation_failed=True,
            verification_failed=False,
            evidence_sources=[],
        )

    try:
        candidate_passages = await asyncio.wait_for(
            passage_gather_task, timeout=settings.RETRIEVAL_TIMEOUT
        )
    except asyncio.TimeoutError:
        passage_gather_task.cancel()
        candidate_passages = []

    non_refusing = [o for o in successful_outputs if not is_refusal(o)]

    if not non_refusing:
        evidence_passages = rank_passages(candidate_passages, req.prompt, top_k=3)
        verification_failed = len(evidence_passages) == 0
        fallback = build_fallback_answer(evidence_passages)
        sources = [EvidenceSource(url=p["url"], text=p["text"], score=p["score"]) for p in evidence_passages]
        
        return EvaluationResponse(
            response=fallback or "The model couldn't answer this, and no supporting web evidence was found either.",
            model_response=successful_outputs[0],
            answer_declined=True,
            fallback_answer=fallback,
            samples_used=samples_used,
            generation_failed=False,
            verification_failed=verification_failed,
            evidence_sources=sources,
        )

    main_output = non_refusing[0]
    evidence_passages = rank_passages(candidate_passages, main_output, top_k=3)
    verification_failed = len(evidence_passages) == 0

    sim_score = semantic_similarity(main_output, req.reference) if req.reference else None
    cons_score = self_consistency(successful_outputs) if samples_used >= 2 else None

    # Calculate metrics for the ML model
    evid_max = evidence_score(evidence_passages)
    evid_mean = float(np.mean([p["score"] for p in evidence_passages])) if evidence_passages else 0.0
    nli_metrics = get_nli_metrics(main_output, evidence_passages)
    h_penalty = nli_metrics.get("h_penalty", 1.0) if not verification_failed else 1.0
    refusal_frac = (samples_used - len(non_refusing)) / samples_used if samples_used > 0 else 1.0

    # Build the feature dictionary for inference
    ml_features = {
        "evidence_max": evid_max,
        "evidence_mean": evid_mean,
        "consistency": cons_score if cons_score is not None else 1.0,
        "p_contradiction_max": nli_metrics["p_contradiction_max"],
        "p_contradiction_mean": nli_metrics["p_contradiction_mean"],
        "p_entailment_max": nli_metrics["p_entailment_max"],
        "p_entailment_mean": nli_metrics["p_entailment_mean"],
        "refusal_fraction": refusal_frac,
        "verification_failed": float(verification_failed),
        "retrieval_relevance": evid_max,  
        "answer_length_words": len(main_output.split())
    }

    final_score = calculate_ml_score(ml_features)

    sources = [EvidenceSource(url=p["url"], text=p["text"], score=p["score"]) for p in evidence_passages]

    return EvaluationResponse(
        response=main_output,
        model_response=main_output,
        answer_declined=False,
        similarity=sim_score,
        evidence=evid_max,
        consistency=cons_score,
        hallucination_penalty= h_penalty,
        final_score=final_score,
        evidence_sources=sources,
        samples_used=samples_used,
        generation_failed=False,
        verification_failed=verification_failed,
    )