import streamlit as st

st.title("GridTune: Hyperparameter Tuning Visualizer")  # PRD objectives

st.sidebar.header("Selections")  # PRD UI: Sidebar
dataset = st.sidebar.selectbox("Dataset", ["Iris", "Wine"])  # FR: Dataset Selection
model = st.sidebar.selectbox("Model", ["Logistic Regression", "SVC"])  # FR: Model Selection

st.write(f"Selected: {dataset}, {model}")