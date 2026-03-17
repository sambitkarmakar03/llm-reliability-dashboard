from evaluation_engine import evaluate_prompt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="LLM Reliability Dashboard", layout="wide")

st.title("LLM Reliability Evaluation Dashboard")

df = pd.read_csv("llama3_results.csv")

st.header("Overall Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Similarity", round(df["Similarity"].mean(),3))
col2.metric("Evidence", round(df["Evidence"].mean(),3))
col3.metric("Consistency", round(df["Consistency"].mean(),3))
col4.metric("Reliability Score", round(df["Final_Score"].mean(),3))

st.write("Average Latency:", round(df["Latency"].mean(),2), "seconds")

st.header("Reliability Score Distribution")

fig1, ax1 = plt.subplots()
sns.histplot(df["Final_Score"], bins=20, ax=ax1)
st.pyplot(fig1)

st.header("Metric Correlation Matrix")

corr = df[["Similarity","Evidence","Consistency","Final_Score"]].corr()

fig2, ax2 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.header("Reliability vs Latency")

fig3, ax3 = plt.subplots()
sns.scatterplot(x=df["Latency"], y=df["Final_Score"], ax=ax3)
st.pyplot(fig3)

st.header("Prompt Diagnostics")

selected_prompt = st.selectbox("Select Prompt", df["Prompt"])

prompt_data = df[df["Prompt"] == selected_prompt]

st.write("Reference Answer:")
st.write(prompt_data["Reference"].values[0])

st.write("Model Response:")
st.write(prompt_data["Model_Response"].values[0])

st.write(prompt_data[["Similarity","Evidence","Consistency","Final_Score"]])

st.header("Potential Hallucinations")

hallucinations = df[df["Evidence"] < 0.3]

st.dataframe(hallucinations[["Prompt","Model_Response","Evidence"]])

st.header("Full Evaluation Data")

st.dataframe(df)

st.divider()
st.header("Real-Time Reliability Evaluation")

prompt_input = st.text_area(
    "Enter Prompt / Question",
    placeholder="Example: What is the capital of Japan?"
)

reference_input = st.text_area(
    "Enter Reference Answer",
    placeholder="Example: Tokyo"
)

if st.button("Evaluate Prompt"):

    if prompt_input and reference_input:

        result = evaluate_prompt(prompt_input, reference_input)

        st.subheader("Model Response")
        st.write(result["response"])

        st.subheader("Evaluation Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Similarity", round(result["similarity"],3))
        col2.metric("Evidence", round(result["evidence"],3))
        col3.metric("Consistency", round(result["consistency"],3))
        col4.metric("Final Score", round(result["final"],3))

        st.write("Latency:", round(result["latency"],2),"seconds")

        # Hallucination indicator
        if result["final"] > 0.7:
            st.success("Low Hallucination Risk")

        elif result["final"] > 0.4:
            st.warning("Moderate Hallucination Risk")

        else:
            st.error("High Hallucination Risk")

    else:
        st.warning("Please provide both prompt and reference answer.")
