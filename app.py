import streamlit as st
import time
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Tuple
warnings.filterwarnings("ignore")

try:
    from groq import Groq
except ImportError:
    st.error("❌ 'groq' package not found. Install with: pip install groq")
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("❌ 'sentence-transformers' package not found. Install with: pip install sentence-transformers")
    st.stop()

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    st.error("❌ 'transformers' package not found. Install with: pip install transformers")
    st.stop()

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

st.set_page_config(
    page_title="LLM Reliability Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.title("LLM Reliability Analyzer")

@st.cache_resource
def init_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found in Streamlit secrets.")
        st.stop()
    return Groq(api_key=api_key)

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

@st.cache_resource(show_spinner="Loading NLI model...")
def load_nli_model():
    try:
        return hf_pipeline(
            "text-classification",
            model="roberta-large-mnli",
            framework="pt",
            device=-1,
        )
    except Exception as e:
        st.warning(f"Failed to load NLI model: {e}")
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / norm)

@st.cache_data(show_spinner=False, ttl=3600)
def get_web_reference(prompt: str) -> Tuple[str, bool]:
    if DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(prompt, max_results=1, timeout=5))
                if results:
                    return f"{results[0]['title']}: {results[0]['body']}", True
        except Exception:
            pass
    
    if WIKIPEDIA_AVAILABLE:
        try:
            import socket
            socket.setdefaulttimeout(5)
            summary = wikipedia.summary(prompt, sentences=2)
            return summary, True
        except Exception:
            pass
    
    return "No verifiable reference found.", False

def check_hallucination(response: str, reference: str, nli_model) -> float:
    if not reference or "No verifiable reference" in reference:
        return 0.0
    
    if nli_model is None:
        return 0.1
    
    try:
        premise = reference[:400]
        hypothesis = response[:200]
        
        result = nli_model(f"{premise} [SEP] {hypothesis}", truncation=True)[0]
        label = result["label"].upper()
        score = result["score"]
        
        if label == "CONTRADICTION":
            return score * 0.9
        elif label == "NEUTRAL":
            return score * 0.2
        else:
            return 0.0
    except Exception:
        return 0.1

def query_llama(prompt: str, model_name: str, temperature: float = 0.7) -> Tuple[str, float]:
    client = init_groq_client()
    start_time = time.time()
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=temperature,
            max_tokens=200,
        )
        response = completion.choices[0].message.content.strip()
        latency = time.time() - start_time
        return response, latency
    except Exception as e:
        return f"[Error: {str(e)}]", time.time() - start_time

def evaluate_prompt(
    prompt: str,
    reference: Optional[str],
    model_name: str,
    emb_model,
    nli_model,
    n_samples: int = 2
) -> Dict:
    if reference and reference.strip():
        ref = reference.strip()
        ref_verified = True
        auto_ref = False
    else:
        ref, ref_verified = get_web_reference(prompt)
        auto_ref = True
    
    responses = []
    latencies = []
    for _ in range(n_samples):
        resp, lat = query_llama(prompt, model_name)
        responses.append(resp)
        latencies.append(lat)
    
    main_response = responses[0]
    
    emb_response = emb_model.encode(main_response, convert_to_numpy=True)
    emb_reference = emb_model.encode(ref, convert_to_numpy=True)
    similarity = cosine_similarity(emb_response, emb_reference)
    
    try:
        search_query = main_response.split(".")[0][:100]
        web_ref, web_verified = get_web_reference(search_query)
        if web_verified and "No verifiable reference" not in web_ref:
            emb_web = emb_model.encode(web_ref, convert_to_numpy=True)
            evidence = cosine_similarity(emb_response, emb_web)
        else:
            evidence = 0.5
    except Exception:
        evidence = 0.5
    
    if len(responses) >= 2:
        embeddings = [emb_model.encode(r, convert_to_numpy=True) for r in responses]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        consistency = float(np.mean(similarities))
    else:
        consistency = 1.0
    
    hallucination = check_hallucination(main_response, ref, nli_model)
    
    base_score = 0.35 * similarity + 0.35 * evidence + 0.30 * consistency
    final_score = max(0.0, min(1.0, base_score - 0.30 * hallucination))
    
    return {
        "response": main_response,
        "reference": ref,
        "similarity": round(similarity, 3),
        "evidence": round(evidence, 3),
        "consistency": round(consistency, 3),
        "hallucination_penalty": round(hallucination, 3),
        "latency": round(np.mean(latencies), 2),
        "final_score": round(final_score, 3),
        "auto_reference": auto_ref,
        "reference_verified": ref_verified
    }

with st.sidebar:
    st.header("Configuration")
    
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    selected_model = st.selectbox("Model", model_options, index=0)
    
    n_samples = st.slider("Response Samples", 1, 3, 2)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.markdown("---")
    st.header("Custom Prompts")
    
    if "custom_prompts" not in st.session_state:
        st.session_state.custom_prompts = []
    
    with st.form("add_prompt_form", clear_on_submit=True):
        new_prompt = st.text_area("Prompt", height=100)
        new_reference = st.text_input("Reference (optional)")
        submitted = st.form_submit_button("Add Prompt")
        
        if submitted and new_prompt.strip():
            st.session_state.custom_prompts.append({
                "prompt": new_prompt.strip(),
                "reference": new_reference.strip() if new_reference.strip() else None
            })
            st.rerun()
    
    if st.session_state.custom_prompts:
        for idx, prompt_data in enumerate(st.session_state.custom_prompts):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{idx+1}. {prompt_data['prompt'][:80]}...")
                if prompt_data['reference']:
                    st.caption(f"Reference: {prompt_data['reference'][:60]}")
                else:
                    st.caption("Auto-search enabled")
            with col2:
                if st.button("X", key=f"del_{idx}"):
                    st.session_state.custom_prompts.pop(idx)
                    st.rerun()
    
    if st.session_state.custom_prompts and st.button("Clear All"):
        st.session_state.custom_prompts = []
        st.rerun()

if st.button("Run Evaluation", type="primary", use_container_width=True):
    
    if not st.session_state.custom_prompts:
        st.error("Please add at least one custom prompt before running evaluation!")
        st.stop()
    
    with st.spinner("Loading models..."):
        emb_model = load_embedding_model()
        nli_model = load_nli_model()
    
    total_prompts = len(st.session_state.custom_prompts)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, prompt_data in enumerate(st.session_state.custom_prompts):
        prompt = prompt_data["prompt"]
        reference = prompt_data["reference"]
        
        progress = (idx + 1) / total_prompts
        progress_bar.progress(progress)
        status_text.info(f"Evaluating prompt {idx + 1} of {total_prompts}")
        
        result = evaluate_prompt(
            prompt=prompt,
            reference=reference,
            model_name=selected_model,
            emb_model=emb_model,
            nli_model=nli_model,
            n_samples=n_samples
        )
        
        results.append({
            "Prompt": prompt,
            "Response": result["response"],
            "Reference": result["reference"],
            "Similarity": result["similarity"],
            "Evidence": result["evidence"],
            "Consistency": result["consistency"],
            "Hallucination Penalty": result["hallucination_penalty"],
            "Latency (s)": result["latency"],
            "Final Score": result["final_score"],
            "Auto-Reference": "Yes" if result["auto_reference"] else "No"
        })
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.evaluation_results = pd.DataFrame(results)
    st.session_state.model_used = selected_model
    
    st.success(f"Evaluation complete! {total_prompts} prompts analyzed.")

if "evaluation_results" in st.session_state:
    df = st.session_state.evaluation_results
    model_used = st.session_state.model_used
    
    st.header(f"Results - {model_used}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Avg Final Score", f"{df['Final Score'].mean():.3f}")
    with col2:
        st.metric("Avg Similarity", f"{df['Similarity'].mean():.3f}")
    with col3:
        st.metric("Avg Evidence", f"{df['Evidence'].mean():.3f}")
    with col4:
        st.metric("Avg Consistency", f"{df['Consistency'].mean():.3f}")
    with col5:
        st.metric("Avg Hallucination", f"{df['Hallucination Penalty'].mean():.3f}")
    
    sort_by = st.selectbox("Sort by", ["Final Score", "Similarity", "Evidence", "Consistency", "Hallucination Penalty"], index=0)
    
    st.markdown("### Detailed Results")
    display_df = df.sort_values(sort_by, ascending=False)
    
    if MATPLOTLIB_AVAILABLE:
        styled_df = display_df.style.background_gradient(
            subset=["Final Score"], cmap="RdYlGn", vmin=0, vmax=1
        ).background_gradient(
            subset=["Hallucination Penalty"], cmap="Reds_r", vmin=0, vmax=1
        )
        st.dataframe(styled_df, use_container_width=True, height=500)
    else:
        st.dataframe(display_df, use_container_width=True, height=500)
    
    st.markdown("### Performance Visualization")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("#### Final Score Distribution")
        st.bar_chart(display_df.set_index("Prompt")[["Final Score"]])
    
    with col_v2:
        st.markdown("#### Metrics Comparison")
        metrics_df = pd.DataFrame({
            "Similarity": [df["Similarity"].mean()],
            "Evidence": [df["Evidence"].mean()],
            "Consistency": [df["Consistency"].mean()],
            "Hallucination": [df["Hallucination Penalty"].mean()]
        })
        st.bar_chart(metrics_df.T)
    
    st.markdown("### Insights")
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown("#### Top 3 Most Reliable")
        top3 = display_df.head(3)[["Prompt", "Final Score", "Similarity", "Evidence"]]
        st.dataframe(top3, use_container_width=True)
    
    with col_ins2:
        st.markdown("#### Bottom 3")
        bottom3 = display_df.tail(3)[["Prompt", "Final Score", "Similarity", "Evidence"]]
        st.dataframe(bottom3, use_container_width=True)
    
    high_hallucination = df[df["Hallucination Penalty"] > 0.5]
    if len(high_hallucination) > 0:
        st.warning(f"{len(high_hallucination)} prompt(s) show high hallucination (>0.5)")
        with st.expander("View high-hallucination prompts"):
            for _, row in high_hallucination.iterrows():
                st.write(f"**{row['Prompt']}**")
                st.write(f"Hallucination Penalty: {row['Hallucination Penalty']:.3f}")
                st.write(f"Response: {row['Response'][:200]}...")
                st.markdown("---")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"llama_reliability_{model_used.split('-')[0]}_{len(df)}prompts.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    if st.button("Clear Results", use_container_width=True):
        del st.session_state.evaluation_results
        st.rerun()

else:
    st.info("Add custom prompts in the sidebar and click 'Run Evaluation' to start.")
