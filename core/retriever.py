import asyncio
import aiohttp
import trafilatura
import numpy as np
from typing import List, Dict, Any, Optional
from ddgs import DDGS
from core.scorer import get_embedding
from core.config import settings


async def fetch_and_extract(session: aiohttp.ClientSession, url: str) -> str:
    """Fetches web page HTML and extracts clean main text using trafilatura."""
    try:
        async with session.get(url, timeout=8) as response:
            if response.status == 200:
                html = await response.text()
                text = trafilatura.extract(html)
                return text if text else ""
    except Exception:
        pass
    return ""


def _sync_ddgs_search(query: str, max_results: int) -> List[Dict[str, str]]:
    """Synchronous DDGS search wrapper, run in a background thread.
    Explicit timeout so a single attempt can't hang past a known bound."""
    with DDGS(timeout=settings.DDGS_TIMEOUT) as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return [{"url": r["href"], "text": r.get("body", "")} for r in results]


async def get_search_results(
    query: str, max_results: int = 5, retries: Optional[int] = None
) -> List[Dict[str, str]]:
    """Executes web search via DDGS in a background thread with capped
    retries + backoff, then falls back to the Wikipedia search API."""
    retries = retries if retries is not None else settings.DDGS_RETRIES

    for attempt in range(retries):
        try:
            results = await asyncio.to_thread(_sync_ddgs_search, query, max_results)
            if results:
                return results
        except Exception as e:
            print(f"DDGS attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(min(1.5 ** attempt, 4))

    print("DDGS exhausted. Falling back to Wikipedia API.")
    results = []
    try:
        async with aiohttp.ClientSession() as session:
            wiki_url = (
                "https://en.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={query}&utf8=&format=json"
            )
            async with session.get(wiki_url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    search_results = data.get("query", {}).get("search", [])
                    for r in search_results[:max_results]:
                        page_id = r["pageid"]
                        results.append(
                            {
                                "url": f"https://en.wikipedia.org/?curid={page_id}",
                                "text": r.get("snippet", ""),
                            }
                        )
    except Exception as e:
        print(f"Wikipedia fallback failed: {e}")

    return results


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Splits text into sliding window chunks."""
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


async def gather_candidate_passages(query: str) -> List[Dict[str, str]]:
    """
    Search + fetch + chunk. Deliberately independent of the model's answer
    so this can be kicked off concurrently with LLM generation — only the
    ranking step below needs the answer.
    """
    search_results = await get_search_results(query)
    if not search_results:
        return []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_extract(session, res["url"]) for res in search_results]
        extracted_texts = await asyncio.gather(*tasks, return_exceptions=True)

    passages = []
    for i, res in enumerate(search_results):
        full_text = (
            extracted_texts[i]
            if isinstance(extracted_texts[i], str) and extracted_texts[i]
            else res["text"]
        )
        for chunk in chunk_text(full_text):
            passages.append({"url": res["url"], "text": chunk})

    return passages


def rank_passages(
    passages: List[Dict[str, str]], answer: str, top_k: int = 3
) -> List[Dict[str, Any]]:
    """Ranks already-gathered passages by similarity to the model's answer.
    Just embeddings + cosine similarity — fast, so it's fine for this part
    alone to wait until generation has actually finished."""
    if not passages:
        return []

    answer_emb = get_embedding(answer)
    scored = []
    for p in passages:
        p_emb = get_embedding(p["text"])
        norm_product = np.linalg.norm(answer_emb) * np.linalg.norm(p_emb)
        score = 0.0 if norm_product < 1e-8 else float(np.dot(answer_emb, p_emb) / norm_product)
        scored.append({"url": p["url"], "text": p["text"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


async def retrieve_evidence(query: str, answer: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Convenience wrapper for callers that don't need the
    concurrent-with-generation optimization (e.g. a standalone script)."""
    passages = await gather_candidate_passages(query)
    return rank_passages(passages, answer, top_k)