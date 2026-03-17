
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
