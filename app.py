import streamlit as st
import time
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Tuple
warnings.filterwarnings("ignore")

# ── Imports with error handling ───────────────────────────────────────────────
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

# Optional imports for web search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    st.warning("⚠️ duckduckgo-search not available. Web search will be limited.")

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    st.warning("⚠️ wikipedia not available. Web search will be limited.")

try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Reliability Analyzer - Llama 70B",
    page_icon="🦙",
    layout="wide",
)

st.title("🦙 LLM Reliability Analyzer - Llama 70B")
st.markdown(
    """
    Evaluate your Llama 70B model across **5 key metrics**:
    - 🎯 **Similarity**: How well responses match the reference
    - 🔍 **Evidence**: How well responses align with factual sources
    - 🔄 **Consistency**: How stable responses are across multiple queries
    - ⚠️ **Hallucination Penalty**: Penalty for generating false information
    - 📊 **Final Score**: Weighted combination of all metrics
    """
)

# ── Groq client initialization ────────────────────────────────────────────────
@st.cache_resource
def init_groq_client():
    """Initialize Groq client with API key from secrets."""
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error(
            "❌ GROQ_API_KEY not found!\n\n"
            "Please add your API key to Streamlit secrets:\n"
            "1. Go to Settings → Secrets\n"
            "2. Add: `GROQ_API_KEY = \"gsk_xxxx\"`\n"
            "3. Get your key from [console.groq.com](https://console.groq.com)"
        )
        st.stop()
    return Groq(api_key=api_key)


# ── Model loading with caching ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def load_embedding_model():
    """Load sentence transformer for embeddings."""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

@st.cache_resource(show_spinner="Loading NLI model for hallucination detection...")
def load_nli_model():
    """Load NLI model for hallucination detection."""
    try:
        return hf_pipeline(
            "text-classification",
            model="roberta-large-mnli",
            framework="pt",
            device=-1,  # CPU
        )
    except Exception as e:
        st.warning(f"Failed to load NLI model: {e}. Hallucination detection disabled.")
        return None


# ── Helper functions ──────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / norm)

@st.cache_data(show_spinner=False, ttl=3600)
def get_web_reference(prompt: str) -> Tuple[str, bool]:
    """
    Retrieve reference from web (DuckDuckGo then Wikipedia).
    Returns (reference_text, success_flag).
    """
    # Try DuckDuckGo first
    if DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(prompt, max_results=1, timeout=5))
                if results:
                    return f"{results[0]['title']}: {results[0]['body']}", True
        except Exception:
            pass
    
    # Try Wikipedia as fallback
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
    """
    Calculate hallucination penalty using NLI.
    Returns penalty between 0 and 1.
    """
    if not reference or "No verifiable reference" in reference:
        return 0.0
    
    if nli_model is None:
        return 0.1
    
    try:
        # Truncate to avoid token limits
        premise = reference[:400]
        hypothesis = response[:200]
        
        result = nli_model(f"{premise} [SEP] {hypothesis}", truncation=True)[0]
        label = result["label"].upper()
        score = result["score"]
        
        if label == "CONTRADICTION":
            return score * 0.9  # Heavy penalty
        elif label == "NEUTRAL":
            return score * 0.2  # Light penalty
        else:  # ENTAILMENT
            return 0.0
    except Exception:
        return 0.1

def query_llama(prompt: str, model_name: str, temperature: float = 0.7) -> Tuple[str, float]:
    """
    Query Llama model via Groq API.
    Returns (response, latency).
    """
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
    """
    Evaluate a single prompt with all metrics.
    """
    # Step 1: Get reference (provided or from web)
    if reference and reference.strip():
        ref = reference.strip()
        ref_verified = True
        auto_ref = False
    else:
        ref, ref_verified = get_web_reference(prompt)
        auto_ref = True
    
    # Step 2: Get multiple responses for consistency
    responses = []
    latencies = []
    for _ in range(n_samples):
        resp, lat = query_llama(prompt, model_name)
        responses.append(resp)
        latencies.append(lat)
    
    main_response = responses[0]
    
    # Step 3: Calculate similarity (response vs reference)
    emb_response = emb_model.encode(main_response, convert_to_numpy=True)
    emb_reference = emb_model.encode(ref, convert_to_numpy=True)
    similarity = cosine_similarity(emb_response, emb_reference)
    
    # Step 4: Calculate evidence (response vs web search)
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
    
    # Step 5: Calculate consistency (between multiple responses)
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
    
    # Step 6: Calculate hallucination penalty
    hallucination = check_hallucination(main_response, ref, nli_model)
    
    # Step 7: Calculate final score
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


# ── UI Components ─────────────────────────────────────────────────────────────
# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Model selection
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    selected_model = st.selectbox(
        "Llama Model",
        model_options,
        index=0,
        help="Llama 3.3 70B is the most capable model"
    )
    
    # Evaluation parameters
    st.markdown("---")
    st.header("📊 Evaluation Settings")
    
    n_samples = st.slider(
        "Response Samples",
        min_value=1,
        max_value=3,
        value=2,
        help="More samples = better consistency score, but more API calls"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, lower = more deterministic"
    )
    
    st.markdown("---")
    st.header("✏️ Custom Prompts")
    st.markdown("Add your own prompts for evaluation")
    
    # Initialize session state for custom prompts
    if "custom_prompts" not in st.session_state:
        st.session_state.custom_prompts = []
    
    # Add new custom prompt
    with st.form("add_prompt_form", clear_on_submit=True):
        new_prompt = st.text_area("Prompt", height=100, placeholder="Enter your question here...")
        new_reference = st.text_input("Reference (optional)", placeholder="Leave blank for automatic web search")
        submitted = st.form_submit_button("➕ Add Prompt")
        
        if submitted and new_prompt.strip():
            st.session_state.custom_prompts.append({
                "prompt": new_prompt.strip(),
                "reference": new_reference.strip() if new_reference.strip() else None
            })
            st.success(f"Added prompt: {new_prompt[:50]}...")
            st.rerun()
    
    # Display existing custom prompts
    if st.session_state.custom_prompts:
        st.markdown("### Current Custom Prompts")
        for idx, prompt_data in enumerate(st.session_state.custom_prompts):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{idx+1}.** {prompt_data['prompt'][:80]}...")
                if prompt_data['reference']:
                    st.caption(f"📝 Reference: {prompt_data['reference'][:60]}")
                else:
                    st.caption("🔍 Auto-search enabled")
            with col2:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.custom_prompts.pop(idx)
                    st.rerun()
    
    # Clear all custom prompts
    if st.session_state.custom_prompts and st.button("🗑️ Clear All Custom Prompts"):
        st.session_state.custom_prompts = []
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 **Tip**: Leave reference blank for automatic web search")
    st.caption("🔑 API key must be set in Streamlit secrets")

# Main area
st.markdown("---")

# Preset prompts
with st.expander("📋 Load Example Prompts", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🇯🇵 Geography Examples"):
            examples = [
                "What is the capital of Japan?",
                "Which ocean is the largest?",
                "What is the highest mountain peak?",
            ]
            for ex in examples:
                st.session_state.custom_prompts.append({"prompt": ex, "reference": None})
            st.success(f"Added {len(examples)} geography prompts")
            st.rerun()
    
    with col2:
        if st.button("🔬 Science Examples"):
            examples = [
                "What is photosynthesis?",
                "What is the chemical symbol for water?",
                "What planet is known as the Red Planet?",
            ]
            for ex in examples:
                st.session_state.custom_prompts.append({"prompt": ex, "reference": None})
            st.success(f"Added {len(examples)} science prompts")
            st.rerun()
    
    with col3:
        if st.button("🤔 Trick Questions"):
            examples = [
                "Who was the president of the US in 1785?",
                "What is the capital of Wakanda?",
                "Which company invented teleportation?",
            ]
            for ex in examples:
                st.session_state.custom_prompts.append({"prompt": ex, "reference": None})
            st.success(f"Added {len(examples)} trick questions")
            st.rerun()

# Run evaluation button
if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
    
    if not st.session_state.custom_prompts:
        st.error("❌ Please add at least one custom prompt before running evaluation!")
        st.stop()
    
    # Load models
    with st.spinner("Loading models (this may take a minute)..."):
        emb_model = load_embedding_model()
        nli_model = load_nli_model()
    
    # Prepare progress tracking
    total_prompts = len(st.session_state.custom_prompts)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_prompt_text = st.empty()
    
    # Evaluation loop
    for idx, prompt_data in enumerate(st.session_state.custom_prompts):
        prompt = prompt_data["prompt"]
        reference = prompt_data["reference"]
        
        # Update progress
        progress = (idx + 1) / total_prompts
        progress_bar.progress(progress)
        status_text.info(f"Evaluating prompt {idx + 1} of {total_prompts}")
        current_prompt_text.markdown(f"**Current:** {prompt[:100]}...")
        
        # Evaluate
        with st.spinner(f"Analyzing: {prompt[:80]}..."):
            result = evaluate_prompt(
                prompt=prompt,
                reference=reference,
                model_name=selected_model,
                emb_model=emb_model,
                nli_model=nli_model,
                n_samples=n_samples
            )
        
        # Store result
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
        
        # Show intermediate result
        with st.expander(f"✅ Result for: {prompt[:60]}...", expanded=False):
            st.markdown(f"**Final Score:** {result['final_score']:.3f}")
            st.markdown(f"**Response:** {result['response']}")
            if result['auto_reference']:
                st.caption(f"🔍 Reference auto-retrieved from web")
            st.caption(f"📊 Sim: {result['similarity']:.3f} | Evid: {result['evidence']:.3f} | "
                      f"Cons: {result['consistency']:.3f} | Hall: {result['hallucination_penalty']:.3f}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    current_prompt_text.empty()
    
    # Store results in session state
    st.session_state.evaluation_results = pd.DataFrame(results)
    st.session_state.model_used = selected_model
    st.session_state.total_prompts = total_prompts
    
    st.success(f"✅ Evaluation complete! {total_prompts} prompts analyzed.")
    st.balloons()

# Display results
if "evaluation_results" in st.session_state:
    df = st.session_state.evaluation_results
    model_used = st.session_state.model_used
    
    st.markdown("---")
    st.header(f"📊 Evaluation Results - {model_used}")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 Avg Final Score", f"{df['Final Score'].mean():.3f}")
    with col2:
        st.metric("🎯 Avg Similarity", f"{df['Similarity'].mean():.3f}")
    with col3:
        st.metric("🔍 Avg Evidence", f"{df['Evidence'].mean():.3f}")
    with col4:
        st.metric("🔄 Avg Consistency", f"{df['Consistency'].mean():.3f}")
    with col5:
        st.metric("⚠️ Avg Hallucination", f"{df['Hallucination Penalty'].mean():.3f}")
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["Final Score", "Similarity", "Evidence", "Consistency", "Hallucination Penalty"],
        index=0
    )
    
    # Display results table
    st.markdown("### Detailed Results")
    display_df = df.sort_values(sort_by, ascending=False)
    
    # Apply styling if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        styled_df = display_df.style.background_gradient(
            subset=["Final Score"], 
            cmap="RdYlGn",
            vmin=0, vmax=1
        ).background_gradient(
            subset=["Hallucination Penalty"],
            cmap="Reds_r",
            vmin=0, vmax=1
        )
        st.dataframe(styled_df, use_container_width=True, height=500)
    else:
        st.dataframe(display_df, use_container_width=True, height=500)
    
    # Visualization
    st.markdown("### 📊 Performance Visualization")
    
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
    
    # Top and bottom performers
    st.markdown("### 🏆 Insights")
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown("#### Top 3 Most Reliable")
        top3 = display_df.head(3)[["Prompt", "Final Score", "Similarity", "Evidence"]]
        st.dataframe(top3, use_container_width=True)
    
    with col_ins2:
        st.markdown("#### Bottom 3 (Need Improvement)")
        bottom3 = display_df.tail(3)[["Prompt", "Final Score", "Similarity", "Evidence"]]
        st.dataframe(bottom3, use_container_width=True)
    
    # Hallucination warnings
    high_hallucination = df[df["Hallucination Penalty"] > 0.5]
    if len(high_hallucination) > 0:
        st.warning(f"⚠️ {len(high_hallucination)} prompt(s) show high hallucination (>0.5)")
        with st.expander("View high-hallucination prompts"):
            for _, row in high_hallucination.iterrows():
                st.write(f"**{row['Prompt']}**")
                st.write(f"Hallucination Penalty: {row['Hallucination Penalty']:.3f}")
                st.write(f"Response: {row['Response'][:200]}...")
                st.markdown("---")
    
    # Download results
    st.markdown("### 📥 Export Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Download Results as CSV",
        data=csv,
        file_name=f"llama_reliability_{model_used.split('-')[0]}_{len(df)}prompts.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Option to clear results
    if st.button("🗑️ Clear Results", use_container_width=True):
        del st.session_state.evaluation_results
        st.rerun()

else:
    # Welcome message when no results yet
    st.info("👋 **Welcome!** Add custom prompts in the sidebar and click 'Run Evaluation' to start.")
    
    st.markdown("""
    ### 🎯 How it works:
    
    1. **Add custom prompts** in the sidebar (with optional reference answers)
    2. **Click Run Evaluation** to analyze all prompts
    3. **Get detailed metrics** for each prompt:
       - **Similarity**: How close is the response to the reference?
       - **Evidence**: Does the response align with web search results?
       - **Consistency**: Are responses stable across multiple queries?
       - **Hallucination Penalty**: Penalty for generating false information
       - **Final Score**: Weighted combination (higher is better)
    
    ### 💡 Tips:
    - Leave reference blank for automatic web search
     - Use 2-3 response samples for better consistency scores
    - Lower temperature (0.3-0.5) for more factual responses
    - Higher temperature (0.8-1.0) for creative responses
    
    ### 🔑 API Key Setup:
    Add your Groq API key to Streamlit secrets:
    ```toml
    GROQ_API_KEY = "gsk_xxxxxxxxxxxx"
