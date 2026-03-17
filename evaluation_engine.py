import numpy as np
import time
import wikipedia
import streamlit as st
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# LOAD API KEY (with fallback)
# ---------------------------
# Try Streamlit secrets first, then environment variable
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        st.error("❌ GROQ_API_KEY not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

client = Groq(api_key=api_key)

# Load embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return embed_model.encode(text)

# ---------------------------
# MODEL CALL (WITH ERROR HANDLING)
# ---------------------------
def run_model(prompt):
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model="llama3-70b-8192",          # You can also try "mixtral-8x7b-32768"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256
        )
        response = completion.choices[0].message.content
        latency = time.time() - start_time
        return response, latency
    except Exception as e:
        st.error(f"❌ Groq API error: {e}")
        # Return a fallback response so the app doesn't crash completely
        return f"Error: {e}", 0.0

# ---------------------------
# SEMANTIC SIMILARITY
# ---------------------------
def semantic_similarity(model_output, reference):
    emb1 = get_embedding(model_output)
    emb2 = get_embedding(reference)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return float(similarity)

# ---------------------------
# WIKIPEDIA EVIDENCE CHECK
# ---------------------------
def evidence_score(response):
    try:
        query = response.split(".")[0]
        wiki_text = wikipedia.summary(query, sentences=2)
        emb1 = get_embedding(response)
        emb2 = get_embedding(wiki_text)
        score = cosine_similarity([emb1], [emb2])[0][0]
        return float(score)
    except:
        return 0.5

# ---------------------------
# SELF CONSISTENCY
# ---------------------------
def self_consistency(outputs):
    embeddings = [get_embedding(o) for o in outputs]
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            sims.append(sim)
    return float(np.mean(sims))

# ---------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------
def evaluate_prompt(prompt, reference):
    outputs = []
    latencies = []
    for _ in range(3):
        response, latency = run_model(prompt)
        outputs.append(response)
        latencies.append(latency)

    main_output = outputs[0]
    sim_score = semantic_similarity(main_output, reference)
    evidence = evidence_score(main_output)
    consistency = self_consistency(outputs)
    avg_latency = np.mean(latencies)

    final_score = (
        0.4 * sim_score +
        0.4 * evidence +
        0.2 * consistency
    )

    return {
        "response": main_output,
        "similarity": sim_score,
        "evidence": evidence,
        "consistency": consistency,
        "latency": avg_latency,
        "final": final_score
    }
