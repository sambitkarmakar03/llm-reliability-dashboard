import time
import aiohttp
from typing import Tuple
from core.config import settings

# Shared sentinel so callers can reliably detect a failed generation
# instead of treating the error text as a real model answer.
LLM_ERROR_PREFIX = "[LLM Error:"


async def generate_response(prompt: str) -> Tuple[str, float]:
    """Asynchronously queries the local LLM endpoint to prevent event loop blocking."""
    start_time = time.time()
    payload = {
        "model": settings.MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 60,
            "temperature": settings.TEMPERATURE,
        },
    }

    try:
        timeout = aiohttp.ClientTimeout(total=settings.LLM_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(settings.OLLAMA_URL, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                output_text = data.get("response", "")
    except Exception as e:
        # type(e).__name__ matters here: several common exceptions
        # (asyncio.TimeoutError in particular) have an empty str(e),
        # which previously produced an unhelpful "[LLM Error: ]".
        output_text = f"{LLM_ERROR_PREFIX} {type(e).__name__}: {e}]"

    return output_text, (time.time() - start_time)


def is_generation_failure(output_text: str) -> bool:
    """True if this output is an error sentinel rather than a real model answer."""
    return output_text.startswith(LLM_ERROR_PREFIX)