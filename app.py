import streamlit as st
from data_handler.data_handler import DataHandler

st.title("GridTune: Hyperparameter Tuning Visualizer")  # PRD objectives

st.sidebar.header("Selections")  # PRD UI: Sidebar
dataset = st.sidebar.selectbox("Dataset", ["Iris", "Wine"])  # FR: Dataset Selection
model = st.sidebar.selectbox("Model", ["Logistic Regression", "SVC"])  # FR: Model Selection
if st.sidebar.button("Load Data"):
    X, y = DataHandler.load_data(dataset)
    X_train, X_test, y_train, y_test = DataHandler.split_data(X,y)
    st.session.state['X_train'] = X_train
    st.session.state['y_train'] = y_train
    st.write(f"Data Loaded: {dataset} with {X_train.shape[0]} training samples.")

st.write(f"Selected: {dataset}, {model}")