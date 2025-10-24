import streamlit as st
from data_handler.data_handler import DataHandler

st.title("GridTune: Hyperparameter Tuning Visualizer")  # PRD objectives

st.sidebar.header("Selections")  # PRD UI: Sidebar
dataset = st.sidebar.selectbox("Dataset", ["Iris", "Wine"])  # FR: Dataset Selection
model = st.sidebar.selectbox("Model", ["Logistic Regression", "SVC"])  # FR: Model Selection

@st.cache_data
def load_and_split(dataset):
    X, y = DataHandler.load_data(dataset)
    return DataHandler.split_data(X, y)

if st.sidebar.button("Load Data"):
    X_train, X_test, y_train, y_test = load_and_split(dataset)
    st.session_state['X_train'] = X_train
    st.session_state['y_train'] = y_train
    st.write(f"Data Loaded: {dataset} with {X_train.shape[0]} training samples.")
