LLM Reliability & Grounding Engine
An asynchronous API for evaluating the reliability and evidence-grounding of Large Language Model (LLM) responses. The engine combines repeated local LLM generations, web retrieval, semantic similarity, self-consistency, and Natural Language Inference (NLI)-based hallucination checks to produce a reliability score.
The API is built with FastAPI, uses Ollama for local text generation, and retrieves web evidence through DuckDuckGo and page extraction.
Implementation note: The current api/main.py computes final_score using a weighted reliability function and a hallucination penalty. The supplied scoring-model notebook trains and evaluates an ML-based scoring model, but the current API endpoint does not load that model. The README describes the implemented API behavior rather than implying that the trained estimator is already used in production.

Table of Contents
- Features
- Architecture
- Evaluation Pipeline
- Reliability Score
- Project Structure
- Requirements
- Installation and Setup
- Configuration
- Running the API
- API Reference
- Example Request
- Example Response
- Testing
- Scoring Model Training
- Limitations and Notes
Features
- Multiple generations: Generates up to three responses to the same prompt to measure consistency.
- Local LLM inference: Uses Ollama, configured for llama3.2:3b in the supplied training workflow.
- Web evidence retrieval: Searches for candidate passages, extracts page text, and ranks passages against the answer.
- Semantic comparison: Optionally compares the generated response against a user-provided reference.
- Self-consistency: Estimates agreement between successful generations using sentence embeddings and cosine similarity.
- NLI-based verification: Evaluates relationships between retrieved evidence and the generated answer.
- Refusal fallback: If all successful generations are refusals, returns the highest-ranked retrieved passage when available.
- Asynchronous execution: Runs retrieval concurrently with the three generation calls.
- Structured API output: Returns the answer, evidence sources, component scores, and failure flags as JSON.
Architecture
                     POST /evaluate
                            |
                            v
                  FastAPI Evaluation API
                            |
               +------------+------------+
               |                         |
               v                         v
       Ollama generation          Web retrieval
       (3 parallel samples)       Search and fetch
               |                         |
               v                         v
       Generation checks          Passage extraction
       and refusal detection      and ranking
               |                         |
               +------------+------------+
                            |
                            v
                 Evaluation and scoring
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
    Reference-based    Self-consistency   Evidence and NLI
     similarity           score           hallucination check
          |                 |                  |
          +-----------------+------------------+
                            |
                            v
                  Weighted reliability
                   minus penalty term
                            |
                            v
                    JSON API response
Technology Stack
Component	Technology
API framework	FastAPI
ASGI server	Uvicorn
Local LLM runtime	Ollama
Example LLM	llama3.2:3b
Embeddings	Sentence Transformers
Embedding model	all-MiniLM-L6-v2
NLI	Hugging Face Transformers NLI pipeline
NLI model	roberta-large-mnli
Web search	ddgs / DuckDuckGo Search
Web extraction	Trafilatura
Async HTTP	aiohttp
Numerical operations	NumPy


Evaluation Pipeline
1. Parallel generation
The API launches three calls to generate_response() for the incoming prompt using asyncio.gather().
Generation failures are filtered using is_generation_failure(). The samples_used field records the number of successful generation outputs.
If all three generations fail, the API returns a failure response and sets generation_failed to true.
2. Evidence retrieval
Web retrieval is started concurrently with generation using gather_candidate_passages().
The endpoint waits for retrieval up to the configured RETRIEVAL_TIMEOUT. If retrieval times out, the retrieval task is cancelled and the endpoint continues without evidence.
Candidate passages are ranked using rank_passages(). In the normal answer path, the API ranks passages against the selected model answer and retains up to three passages.
The evidence_sources field contains the URL, passage text, and relevance score for each retained passage.
3. Refusal handling
The engine checks successful outputs with is_refusal().
If every successful output is a refusal, the API ranks retrieved passages against the original prompt. It returns the highest-ranked passage as a fallback answer, if one is available.
In this branch:
- answer_declined is true.
- fallback_answer contains the retrieved passage, or null if no passage is available.
- verification_failed indicates whether no ranked evidence was found.
- The endpoint returns without calculating the normal reliability score.
4. Semantic similarity
If the request includes a non-null reference, the API calculates semantic similarity between the selected model response and the reference using semantic_similarity().
If no reference is provided, similarity is null. The reference is optional and is not required for web-based evidence evaluation.
5. Self-consistency
When at least two successful generations are available, the engine calculates self-consistency using self_consistency().
The consistency calculation includes successful outputs that were refusals. This prevents refusal samples from being silently excluded from the consistency signal.
If fewer than two successful generations are available, consistency is null.
6. Evidence and hallucination checks
If no ranked evidence is available:
- evidence is set to 0.0.
- hallucination_penalty is set to 1.0.
- verification_failed is set to true.
Otherwise, the API calculates the evidence score using evidence_score() and the hallucination penalty using check_hallucination_penalty().
The supplied scoring-model notebook also extracts NLI entailment and contradiction probabilities from evidence-answer pairs. Those features are used in the notebook's model-training and evaluation workflow; the current API endpoint calls the scorer's hallucination-penalty function rather than directly passing an 11-feature vector to a loaded estimator.
Reliability Score
The current API calculates a weighted base reliability score from the available components.
Let:
- \(S\) = reference similarity, when a reference is provided.
- \(E\) = evidence score.
- \(C\) = self-consistency score, when enough generations are available.
- \(H\) = hallucination penalty.
Base reliability
When reference similarity is available:
\[
R_{\text{base}} =
\frac{0.35S + 0.35E + w_C C}
{0.35 + 0.35 + w_C}
\]
where \(w_C=0.30\) when consistency is available.
When no reference is provided, evidence receives a weight of 0.60. If consistency is available, it receives a weight of 0.40. If a component is unavailable, the remaining weights are normalized.
The implementation performs this normalization in weighted_reliability().
Final score
The final score is calculated as:
\[
R_{\text{final}} =
\max\left(0,\min\left(1,R_{\text{base}}-0.30H\right)\right)
\]
The score is bounded between 0 and 1.
Interpretation: A higher score indicates that the response has stronger signals under the implemented scoring procedure. It should not be interpreted as a statistically calibrated probability of factual correctness unless calibration has been established on an appropriate held-out evaluation set.
Project Structure
The following is the structure represented by the supplied code bundle:
LLM_Reliability/
├── api/
│   ├── __init__.py
│   └── main.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_client.py
│   ├── refusal.py
│   ├── retriever.py
│   ├── scorer.py
│   └── scoring_model.json
├── Scoring_Model.ipynb
├── requirements.txt
└── test.py
Core modules
File	Responsibility
api/main.py	Defines the FastAPI app, request and response schemas, evaluation endpoint, fallback behavior, and weighted reliability calculation.
core/config.py	Loads the engine's configuration and settings.
core/llm_client.py	Handles LLM generation and generation-failure detection.
core/refusal.py	Detects refusal-style model responses.
core/retriever.py	Retrieves web results, extracts page content, gathers candidate passages, and ranks passages.
core/scorer.py	Provides embedding, similarity, consistency, evidence, and hallucination-scoring utilities.
core/scoring_model.json	Contains the scoring-model export included in the supplied bundle.
Scoring_Model.ipynb	Notebook for collecting evaluation data, extracting features, training, and evaluating the scoring model.
test.py	Sends a sample request to the local API and prints the JSON response.


Requirements
- Python 3.11 is recommended for the supplied setup.
- Ollama installed and available on the machine running the API.
- The Ollama model configured in core/config.py must be downloaded.
- Internet access is needed for web retrieval and initial downloads of Hugging Face models.
Python dependencies are listed in requirements.txt:
fastapi
uvicorn
python-dotenv
requests
aiohttp
trafilatura
numpy
duckduckgo-search
transformers
sentence-transformers
The retriever code imports the ddgs package. The package name in the supplied requirements file is duckduckgo-search; depending on the installed package version, you may need to install or pin a version that provides the ddgs import.
For reproducible deployment, pin dependency versions and use compatible versions of PyTorch, Transformers, Sentence Transformers, and the model artifacts.
Installation and Setup
1. Clone the repository
Replace the example URL with the actual repository URL.
git clone https://github.com/yourusername/llm-reliability-engine.git
cd llm-reliability-engine
2. Create and activate a virtual environment
python3.11 -m venv venv
macOS/Linux:
source venv/bin/activate
Windows:
venv\Scripts\activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
Install PyTorch using the appropriate instructions for your operating system and hardware if it is not installed automatically as a dependency.
4. Install and start Ollama
Install Ollama from the official website:
https://ollama.com/
Pull the model used by your configuration. For example:
ollama pull llama3.2:3b
Start the Ollama service:
ollama serve
Keep Ollama running while using the API. If Ollama is already running as a background service, do not start a second instance.
5. Configure environment variables
The configuration is defined in core/config.py. Review that file and set the Ollama URL, model name, and timeout values as needed.
The supplied notebook uses these environment variable names:
OLLAMA_URL=http://127.0.0.1:11434/api/generate
MODEL_NAME=llama3.2:3b
LLM_TIMEOUT=120
RETRIEVAL_TIMEOUT=45
These are example values from the supplied workflow. Confirm that they match the settings and defaults in your local core/config.py.
If the project uses a .env file, keep it local and do not commit credentials or private configuration.
Running the API
From the project root, start the FastAPI application:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
The API will be available at:
- API endpoint: http://127.0.0.1:8000/evaluate
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
The import path api.main:app assumes the project is launched from the repository root and that api is a Python package.
API Reference
POST /evaluate
Evaluates an LLM response against retrieved web evidence and optional reference text.
Request body
Field	Type	Required	Description
prompt	string	Yes	User question or prompt sent to the LLM and used for evidence retrieval in the refusal path.
reference	string or null	No	Optional reference answer used to calculate semantic similarity. Defaults to null.


Example schema:
{
  "prompt": "What is the difference between nuclear fission and nuclear fusion?",
  "reference": null
}
Response fields
Field	Type	Description
response	string	Answer served to the caller. May be a retrieved fallback passage if the model refused.
model_response	string	Selected model output, before fallback substitution.
answer_declined	boolean	Indicates that all successful model samples were detected as refusals and the fallback branch was used.
fallback_answer	string or null	Retrieved fallback passage when available.
similarity	number or null	Semantic similarity to the optional reference answer.
evidence	number or null	Evidence score computed from ranked passages in the normal evaluation path.
consistency	number or null	Self-consistency across successful generations when at least two are available.
hallucination_penalty	number or null	Penalty calculated by the hallucination-checking function.
final_score	number or null	Weighted reliability score after the hallucination penalty.
evidence_sources	array	Ranked evidence passages with source URL, text, and relevance score.
samples_used	integer	Number of successful generation samples.
generation_failed	boolean	Indicates that no successful generation was produced.
verification_failed	boolean	Indicates that no ranked evidence passages were available.


Some score fields are null in the all-refusal or total-generation-failure branches because the normal scoring procedure is not executed.
Example Request
Using cURL:
curl -X POST "http://127.0.0.1:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the difference between nuclear fission and nuclear fusion?",
    "reference": null
  }'
Using Python:
import requests

url = "http://127.0.0.1:8000/evaluate"

payload = {
    "prompt": "What is the difference between nuclear fission and nuclear fusion?",
    "reference": None
}

response = requests.post(url, json=payload, timeout=180)
response.raise_for_status()

result = response.json()
print(result)
Example Response
The following is an illustrative response showing the response structure. Scores and evidence will vary with model output, retrieved pages, and runtime configuration.
{
  "response": "Nuclear fission is the splitting of a heavy atomic nucleus into lighter nuclei, releasing energy. Nuclear fusion combines light nuclei into a heavier nucleus, also releasing energy.",
  "model_response": "Nuclear fission is the splitting of a heavy atomic nucleus into lighter nuclei, releasing energy. Nuclear fusion combines light nuclei into a heavier nucleus, also releasing energy.",
  "answer_declined": false,
  "fallback_answer": null,
  "similarity": null,
  "evidence": 0.85,
  "consistency": 0.96,
  "hallucination_penalty": 0.0,
  "final_score": 0.91,
  "evidence_sources": [
    {
      "url": "https://example.com/fission-and-fusion",
      "text": "Fission involves splitting a heavy nucleus, while fusion involves combining lighter nuclei.",
      "score": 0.85
    }
  ],
  "samples_used": 3,
  "generation_failed": false,
  "verification_failed": false
}
The example URL and numerical values are placeholders, not results from a live API call.
Testing
With Ollama and the FastAPI server running, execute:
python test.py
The supplied test script sends a request about findings from the James Webb Space Telescope regarding exoplanet WASP-39b to:
http://127.0.0.1:8000/evaluate
It prints the returned JSON response. You can also test interactively through the Swagger UI at /docs.
Scoring Model Training
The supplied Scoring_Model.ipynb notebook implements a separate model-training and evaluation workflow.
The notebook:
1. Loads a sample of TriviaQA validation questions and their answer aliases.
2. Generates model responses and gathers web evidence.
3. Extracts the reliability features defined in the notebook.
4. Labels answers using normalized whole-word matching against TriviaQA aliases.
5. Trains and evaluates scoring approaches, including calibration and an optional contradiction gate.
6. Exports scoring-model information to JSON.
The notebook defines the following 11 features:
1. evidence_max
2. evidence_mean
3. consistency
4. p_contradiction_max
5. p_contradiction_mean
6. p_entailment_max
7. p_entailment_mean
8. refusal_fraction
9. verification_failed
10. retrieval_relevance
11. answer_length_words
The notebook's exported JSON and the included core/scoring_model.json are not, by themselves, evidence that the API endpoint uses a trained estimator. The current endpoint imports scoring functions from core.scorer and computes the weighted formula described above.
To use an ML estimator for live scoring, the API would need to load a compatible serialized estimator and construct the exact feature vector in the same order and preprocessing format used during training. The training and inference pipelines should also share the same feature definitions, model version, and preprocessing steps.
Limitations and Notes
- Evidence is not proof: Retrieved passages can be incomplete, outdated, irrelevant, or incorrect. A high semantic similarity score does not establish factual correctness.
- Consistency is not truth: Multiple generations can repeat the same incorrect claim.
- Reference similarity is optional: Without a reference answer, the API does not calculate the similarity field.
- Retrieval can fail: Network errors, rate limits, timeouts, and inaccessible pages can result in no evidence being available.
- Refusal fallback is extractive: The fallback returns a truncated retrieved passage rather than generating a verified synthesis.
- Scoring is heuristic in the current API: The weighted score and penalty are not inherently a calibrated probability of correctness.
- CORS is permissive: The supplied API allows all origins for local development. Restrict allowed origins before exposing the service publicly.
- Model and dependency compatibility matters: Pin and test package versions, especially for Transformers, PyTorch, Sentence Transformers, and any serialized model artifacts.
- Do not expose an unauthenticated deployment publicly: Add appropriate authentication, request limits, logging controls, and operational safeguards before production use.
License
Add the license applicable to this repository before distributing or publishing it.
