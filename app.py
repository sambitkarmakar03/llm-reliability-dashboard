import streamlit as st
import time
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Reliability Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 LLM Reliability Analyzer")
st.markdown(
    "Evaluates an Ollama-hosted LLM across **Similarity · Evidence · Consistency · "
    "Hallucination** metrics — no TensorFlow required."
)

# ── Lazy model loading (cached across reruns) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading NLI model…")
def load_nli_pipeline():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model="roberta-large-mnli",
        framework="pt",          # PyTorch — no TF dependency
        device=-1,               # CPU
    )


# ── Cosine similarity (pure numpy) ───────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / norm)


# ── Reference retrieval ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_automated_reference(prompt: str):
    """Returns (reference_text, is_verified)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(prompt, max_results=1, timeout=5))
            if results:
                snippet = f"{results[0]['title']}: {results[0]['body']}"
                return snippet, True
    except Exception:
        pass

    try:
        import socket, wikipedia
        socket.setdefaulttimeout(5)
        text = wikipedia.summary(prompt, sentences=2)
        return text, True
    except Exception:
        pass

    return "No verifiable reference found.", False


# ── Hallucination penalty ─────────────────────────────────────────────────────
def check_hallucination_penalty(response: str, reference: str, nli) -> float:
    if not reference or "No verifiable reference" in reference:
        return 0.0
    try:
        premise    = reference[:400]
        hypothesis = response[:200]
        result = nli(f"{premise} [SEP] {hypothesis}", truncation=True)[0]
        label  = result["label"].upper()
        score  = result["score"]
        if label == "CONTRADICTION":
            return score * 0.9
        elif label == "NEUTRAL":
            return score * 0.2
        else:
            return 0.0
    except Exception:
        return 0.1


# ── Ollama query ──────────────────────────────────────────────────────────────
def run_model(prompt: str, ollama_url: str, model_name: str, temperature: float = 0.8):
    start = time.time()
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 60, "temperature": temperature},
    }
    try:
        r = requests.post(ollama_url, json=payload, timeout=15)
        text = r.json().get("response", "") if r.status_code == 200 \
               else f"[Ollama error: status {r.status_code}]"
    except Exception as e:
        text = f"[Ollama error: {e}]"
    return text, time.time() - start


# ── Per-prompt evaluation ─────────────────────────────────────────────────────
def evaluate_prompt(prompt, reference, ollama_url, model_name, emb_model, nli_model, n_samples=3):
    # 1. Reference
    if reference:
        ref, ref_verified = reference, True
    else:
        ref, ref_verified = get_automated_reference(prompt)

    # 2. Collect responses
    outputs, latencies = [], []
    for _ in range(n_samples):
        resp, lat = run_model(prompt, ollama_url, model_name)
        outputs.append(resp)
        latencies.append(lat)

    main_output = outputs[0]

    # 3. Embeddings (numpy)
    emb_main = emb_model.encode(main_output, convert_to_numpy=True)
    emb_ref  = emb_model.encode(ref,         convert_to_numpy=True)

    sim_score = cosine_similarity(emb_main, emb_ref)

    # Evidence: compare against auto-reference of the first sentence
    try:
        query     = main_output.split(".")[0][:100]
        wiki_ref, wiki_ok = get_automated_reference(query)
        if wiki_ok and "No verifiable reference" not in wiki_ref:
            emb_wiki = emb_model.encode(wiki_ref, convert_to_numpy=True)
            evidence = cosine_similarity(emb_main, emb_wiki)
        else:
            evidence = 0.5
    except Exception:
        evidence = 0.5

    # Self-consistency
    if len(outputs) >= 2:
        embs = [emb_model.encode(o, convert_to_numpy=True) for o in outputs]
        sims = [cosine_similarity(embs[i], embs[j])
                for i in range(len(embs)) for j in range(i + 1, len(embs))]
        consistency = float(np.mean(sims))
    else:
        consistency = 1.0

    # Hallucination penalty
    h_penalty = check_hallucination_penalty(main_output, ref, nli_model) \
                if ref_verified else 0.0

    base  = 0.35 * sim_score + 0.35 * evidence + 0.30 * consistency
    final = max(0.0, min(1.0, base - 0.30 * h_penalty))

    return {
        "response":    main_output,
        "reference":   ref,
        "similarity":  sim_score,
        "evidence":    evidence,
        "consistency": consistency,
        "h_penalty":   h_penalty,
        "latency":     float(np.mean(latencies)),
        "final":       final,
    }


# ── Default dataset ───────────────────────────────────────────────────────────
DEFAULT_DATASET = [
    {"prompt": "What is the capital of Japan?",              "reference": "Tokyo"},
    {"prompt": "What is the capital of France?",             "reference": "Paris"},
    {"prompt": "What is the capital of Canada?",             "reference": "Ottawa"},
    {"prompt": "What is the capital of Australia?",          "reference": "Canberra"},
    {"prompt": "What is the capital of Germany?",            "reference": "Berlin"},
    {"prompt": "Which is the largest country by area?",      "reference": "Russia"},
    {"prompt": "Which ocean is the largest?",                "reference": "Pacific Ocean"},
    {"prompt": "Which desert is the largest hot desert?",    "reference": "Sahara Desert"},
    {"prompt": "Which river is the longest in the world?",   "reference": "Nile River"},
    {"prompt": "What is 12 multiplied by 8?",                "reference": "96"},
    {"prompt": "What is the square root of 144?",            "reference": "12"},
    {"prompt": "What is the chemical symbol for water?",     "reference": "H2O"},
    {"prompt": "What planet is known as the Red Planet?",    "reference": "Mars"},
    {"prompt": "Who discovered penicillin?",                 "reference": "Alexander Fleming"},
    {"prompt": "Who painted the Mona Lisa?",                 "reference": "Leonardo da Vinci"},
    {"prompt": "Who was the first man to walk on the Moon?", "reference": "Neil Armstrong"},
    {"prompt": "What does CPU stand for?",                   "reference": "Central Processing Unit"},
    {"prompt": "What does RAM stand for?",                   "reference": "Random Access Memory"},
    {"prompt": "What is photosynthesis?",                    "reference": "Process by which plants convert sunlight into chemical energy"},
    {"prompt": "What is gravity?",                           "reference": "Force that attracts objects with mass toward each other"},
    {"prompt": "Who was the president of the United States in 1785?", "reference": "There was no US president in 1785"},
    {"prompt": "What is the capital of the fictional country Wakanda?", "reference": "Wakanda is fictional"},
    {"prompt": "Which company invented teleportation technology?",      "reference": "Teleportation technology does not exist"},
    {"prompt": "What is the exact number of stars in the universe?",    "reference": "Unknown"},
    {"prompt": "Which company will be the richest in 2100?",            "reference": "Unknown"},
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    ollama_url  = st.text_input("Ollama URL",   value="http://localhost:11434/api/generate")
    model_name  = st.text_input("Model name",   value="llama3.2:3b")
    n_samples   = st.slider("Responses per prompt (consistency)", 1, 5, 3)
    max_prompts = st.slider("Max prompts to evaluate", 1, len(DEFAULT_DATASET), 10)

    st.markdown("---")
    st.subheader("➕ Add custom prompt")
    custom_prompt    = st.text_area("Prompt")
    custom_reference = st.text_input("Reference answer (optional)")

    st.markdown("---")
    st.info("Models load on first run and are cached. Ollama must be running locally or be accessible from this machine.")


# ── Main ──────────────────────────────────────────────────────────────────────
dataset = DEFAULT_DATASET[:max_prompts]
if custom_prompt.strip():
    dataset.append({"prompt": custom_prompt.strip(),
                    "reference": custom_reference.strip() or None})

if st.button("▶ Run Evaluation", type="primary"):

    emb_model = load_embedding_model()
    nli_model = load_nli_pipeline()

    records   = []
    progress  = st.progress(0, text="Starting…")
    log_area  = st.empty()

    for idx, item in enumerate(dataset):
        p   = item["prompt"]
        ref = item.get("reference")

        progress.progress((idx + 1) / len(dataset),
                          text=f"[{idx+1}/{len(dataset)}] {p[:60]}…")
        log_area.info(f"Evaluating: **{p}**")

        result = evaluate_prompt(p, ref, ollama_url, model_name,
                                 emb_model, nli_model, n_samples)

        records.append({
            "Prompt":       p,
            "Reference":    str(result["reference"])[:120],
            "Response":     result["response"][:200],
            "Similarity":   round(result["similarity"],  3),
            "Evidence":     round(result["evidence"],    3),
            "Consistency":  round(result["consistency"], 3),
            "H_Penalty":    round(result["h_penalty"],   3),
            "Latency (s)":  round(result["latency"],     2),
            "Final Score":  round(result["final"],       3),
        })

    progress.empty()
    log_area.empty()

    df = pd.DataFrame(records).sort_values("Final Score", ascending=False)
    st.session_state["df"] = df
    st.success(f"✅ Evaluated {len(df)} prompts.")

# ── Results ───────────────────────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Final Score",   f"{df['Final Score'].mean():.3f}")
    c2.metric("Avg Similarity",    f"{df['Similarity'].mean():.3f}")
    c3.metric("Avg Consistency",   f"{df['Consistency'].mean():.3f}")
    c4.metric("Avg H-Penalty",     f"{df['H_Penalty'].mean():.3f}")

    st.markdown("### 📊 Results Table")
    st.dataframe(
        df.style.background_gradient(subset=["Final Score"], cmap="RdYlGn"),
        use_container_width=True,
        height=500,
    )

    # Score distribution
    st.markdown("### 📈 Score Distribution")
    chart_df = df[["Prompt", "Similarity", "Evidence", "Consistency", "Final Score"]] \
                 .set_index("Prompt")
    st.bar_chart(chart_df[["Final Score"]])

    st.markdown("### 🏆 Top 5 Most Reliable")
    st.table(df.head(5)[["Prompt", "Reference", "Final Score"]])

    st.markdown("### ⚠️ Bottom 5 (Least Reliable)")
    st.table(df.tail(5)[["Prompt", "Response", "H_Penalty", "Final Score"]])

    # CSV download
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "llm_reliability_results.csv", "text/csv")
