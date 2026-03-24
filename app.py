import streamlit as st
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Top-level imports — avoids ModuleNotFoundError inside cached functions ────
try:
    from groq import Groq
except ImportError:
    st.error("❌ 'groq' package not found. Check requirements.txt and reboot the app.")
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("❌ 'sentence-transformers' package not found.")
    st.stop()

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    st.error("❌ 'transformers' package not found.")
    st.stop()

# Check for matplotlib and provide fallback
try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("⚠️ matplotlib not installed. Gradient styling will be disabled. Run: pip install matplotlib")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Reliability Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 LLM Reliability Analyzer")
st.markdown(
    "Evaluates a Groq-hosted LLM across **Similarity · Evidence · Consistency · "
    "Hallucination** metrics — fully serverless, no Ollama needed."
)

# ── Groq client ───────────────────────────────────────────────────────────────
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error(
            "❌ GROQ_API_KEY not found in Streamlit secrets.\n\n"
            "Go to **Manage app → Settings → Secrets** and add:\n"
            "```toml\nGROQ_API_KEY = \"gsk_xxxx\"\n```"
        )
        st.stop()
    return Groq(api_key=api_key)


# ── Cached model loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading NLI model…")
def load_nli_pipeline():
    return hf_pipeline(
        "text-classification",
        model="roberta-large-mnli",
        framework="pt",
        device=-1,
    )


# ── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / norm)


# ── Reference retrieval ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_automated_reference(prompt: str):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(prompt, max_results=1, timeout=5))
            if results:
                return f"{results[0]['title']}: {results[0]['body']}", True
    except Exception:
        pass
    try:
        import socket, wikipedia
        socket.setdefaulttimeout(5)
        return wikipedia.summary(prompt, sentences=2), True
    except Exception:
        pass
    return "No verifiable reference found.", False


# ── Hallucination penalty ─────────────────────────────────────────────────────
def check_hallucination_penalty(response: str, reference: str, nli) -> float:
    if not reference or "No verifiable reference" in reference:
        return 0.0
    try:
        result = nli(f"{reference[:400]} [SEP] {response[:200]}", truncation=True)[0]
        label  = result["label"].upper()
        score  = result["score"]
        if label == "CONTRADICTION":
            return score * 0.9
        elif label == "NEUTRAL":
            return score * 0.2
        return 0.0
    except Exception:
        return 0.1


# ── Groq query ────────────────────────────────────────────────────────────────
def run_model(prompt: str, model_name: str, temperature: float = 0.8):
    client = get_groq_client()          # plain function call, no caching
    start  = time.time()
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=temperature,
            max_tokens=120,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        text = f"[Groq error: {e}]"
    return text, time.time() - start


# ── Per-prompt evaluation ─────────────────────────────────────────────────────
def evaluate_prompt(prompt, reference, model_name, emb_model, nli_model, n_samples=3):
    if reference:
        ref, ref_verified = str(reference), True
    else:
        ref, ref_verified = get_automated_reference(prompt)

    outputs, latencies = [], []
    for _ in range(n_samples):
        resp, lat = run_model(prompt, model_name)
        outputs.append(resp)
        latencies.append(lat)

    main_output = outputs[0]

    emb_main  = emb_model.encode(main_output, convert_to_numpy=True)
    emb_ref   = emb_model.encode(ref,         convert_to_numpy=True)
    sim_score = cosine_similarity(emb_main, emb_ref)

    try:
        query = main_output.split(".")[0][:100]
        wiki_ref, wiki_ok = get_automated_reference(query)
        if wiki_ok and "No verifiable reference" not in wiki_ref:
            emb_wiki = emb_model.encode(wiki_ref, convert_to_numpy=True)
            evidence = cosine_similarity(emb_main, emb_wiki)
        else:
            evidence = 0.5
    except Exception:
        evidence = 0.5

    if len(outputs) >= 2:
        embs = [emb_model.encode(o, convert_to_numpy=True) for o in outputs]
        sims = [cosine_similarity(embs[i], embs[j])
                for i in range(len(embs)) for j in range(i + 1, len(embs))]
        consistency = float(np.mean(sims))
    else:
        consistency = 1.0

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
    {"prompt": "What is the capital of Japan?",               "reference": "Tokyo"},
    {"prompt": "What is the capital of France?",              "reference": "Paris"},
    {"prompt": "What is the capital of Canada?",              "reference": "Ottawa"},
    {"prompt": "What is the capital of Australia?",           "reference": "Canberra"},
    {"prompt": "What is the capital of Germany?",             "reference": "Berlin"},
    {"prompt": "Which is the largest country by area?",       "reference": "Russia"},
    {"prompt": "Which ocean is the largest?",                 "reference": "Pacific Ocean"},
    {"prompt": "Which desert is the largest hot desert?",     "reference": "Sahara Desert"},
    {"prompt": "Which river is the longest in the world?",    "reference": "Nile River"},
    {"prompt": "What is 12 multiplied by 8?",                 "reference": "96"},
    {"prompt": "What is the square root of 144?",             "reference": "12"},
    {"prompt": "What is the chemical symbol for water?",      "reference": "H2O"},
    {"prompt": "What planet is known as the Red Planet?",     "reference": "Mars"},
    {"prompt": "Who discovered penicillin?",                  "reference": "Alexander Fleming"},
    {"prompt": "Who painted the Mona Lisa?",                  "reference": "Leonardo da Vinci"},
    {"prompt": "Who was the first man to walk on the Moon?",  "reference": "Neil Armstrong"},
    {"prompt": "What does CPU stand for?",                    "reference": "Central Processing Unit"},
    {"prompt": "What does RAM stand for?",                    "reference": "Random Access Memory"},
    {"prompt": "What is photosynthesis?",                     "reference": "Process by which plants convert sunlight into chemical energy"},
    {"prompt": "What is gravity?",                            "reference": "Force that attracts objects with mass toward each other"},
    {"prompt": "Who was the president of the United States in 1785?", "reference": "There was no US president in 1785"},
    {"prompt": "What is the capital of the fictional country Wakanda?", "reference": "Wakanda is fictional"},
    {"prompt": "Which company invented teleportation technology?",      "reference": "Teleportation technology does not exist"},
    {"prompt": "What is the exact number of stars in the universe?",    "reference": "Unknown"},
    {"prompt": "Which company will be the richest in 2100?",            "reference": "Unknown"},
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    model_name  = st.selectbox("Groq model", GROQ_MODELS, index=0)
    n_samples   = st.slider("Responses per prompt (consistency)", 1, 5, 3)
    max_prompts = st.slider("Max prompts to evaluate", 1, len(DEFAULT_DATASET),
                            min(10, len(DEFAULT_DATASET)))

    st.markdown("---")
    st.subheader("➕ Add a custom prompt")
    custom_prompt    = st.text_area("Prompt")
    custom_reference = st.text_input("Reference answer (optional)")

    st.markdown("---")
    st.subheader("🔑 Secrets setup")
    st.markdown(
        "In **Streamlit Cloud → Manage app → Settings → Secrets**, add:\n"
        "```toml\n"
        'GROQ_API_KEY = "gsk_xxxxxxxxxxxx"\n'
        "```\n"
        "Get a free key at [console.groq.com](https://console.groq.com)"
    )

# ── Build dataset ─────────────────────────────────────────────────────────────
dataset = DEFAULT_DATASET[:max_prompts]
if custom_prompt.strip():
    dataset.append({
        "prompt":    custom_prompt.strip(),
        "reference": custom_reference.strip() or None,
    })

# ── Run button ────────────────────────────────────────────────────────────────
if st.button("▶ Run Evaluation", type="primary"):

    emb_model = load_embedding_model()
    nli_model = load_nli_pipeline()

    records  = []
    progress = st.progress(0, text="Starting…")
    log_area = st.empty()

    for idx, item in enumerate(dataset):
        p   = item["prompt"]
        ref = item.get("reference")

        progress.progress(
            (idx + 1) / len(dataset),
            text=f"[{idx+1}/{len(dataset)}] {p[:70]}…"
        )
        log_area.info(f"⏳ Evaluating: **{p}**")

        result = evaluate_prompt(p, ref, model_name, emb_model, nli_model, n_samples)

        records.append({
            "Prompt":      p,
            "Reference":   str(result["reference"])[:120],
            "Response":    result["response"][:200],
            "Similarity":  round(result["similarity"],  3),
            "Evidence":    round(result["evidence"],    3),
            "Consistency": round(result["consistency"], 3),
            "H_Penalty":   round(result["h_penalty"],   3),
            "Latency (s)": round(result["latency"],     2),
            "Final Score": round(result["final"],       3),
        })

    progress.empty()
    log_area.empty()

    df = pd.DataFrame(records).sort_values("Final Score", ascending=False)
    st.session_state["df"]         = df
    st.session_state["model_used"] = model_name
    st.success(f"✅ Evaluated {len(df)} prompts using **{model_name}**.")

# ── Results display ───────────────────────────────────────────────────────────
if "df" in st.session_state:
    df         = st.session_state["df"]
    model_used = st.session_state.get("model_used", "")

    st.markdown(f"### 📊 Results — `{model_used}`")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Final Score",  f"{df['Final Score'].mean():.3f}")
    k2.metric("Avg Similarity",   f"{df['Similarity'].mean():.3f}")
    k3.metric("Avg Evidence",     f"{df['Evidence'].mean():.3f}")
    k4.metric("Avg Consistency",  f"{df['Consistency'].mean():.3f}")
    k5.metric("Avg H-Penalty",    f"{df['H_Penalty'].mean():.3f}")

    st.markdown("#### All Results")
    
    # Apply styling only if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        styled_df = df.style.background_gradient(subset=["Final Score"], cmap="RdYlGn")
        styled_df = styled_df.background_gradient(subset=["H_Penalty"], cmap="Reds")
        st.dataframe(styled_df, use_container_width=True, height=480)
    else:
        # Display without gradient styling
        st.dataframe(df, use_container_width=True, height=480)
        st.info("💡 Install matplotlib to enable gradient coloring: `pip install matplotlib`")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Final Score by Prompt")
        st.bar_chart(df.set_index("Prompt")[["Final Score"]].sort_values("Final Score"))
    with col_b:
        st.markdown("#### Avg Score Components")
        st.bar_chart(pd.DataFrame({
            "Value": [df["Similarity"].mean(), df["Evidence"].mean(),
                      df["Consistency"].mean(), df["H_Penalty"].mean()]
        }, index=["Similarity", "Evidence", "Consistency", "H_Penalty"]))

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 🏆 Top 5 Most Reliable")
        st.table(df.head(5)[["Prompt", "Final Score", "H_Penalty"]])
    with col_d:
        st.markdown("#### ⚠️ Bottom 5 — Needs Review")
        st.table(df.tail(5)[["Prompt", "Final Score", "H_Penalty"]])

    csv = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download full results as CSV",
        csv,
        f"llm_reliability_{model_used}.csv",
        "text/csv",
    )
