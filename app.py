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

from model_trainer import ModelTrainer

param_options = {
    "Logistic Regression": {
        'C': [ 0.1, 1],
        'penalty': [ 'l2']
    },
    "SVC": {
        'C': [0.1, 1],
        'kernel': ['linear']
    }
}

param_grid = param_options[model]

if 'X_train' in st.session_state and st.sidebar.button("Train Model"):
    with st.spinner("tunning hyperparameters..."):
        grid , best_params, cv_results = ModelTrainer.train(
            st.session_state['X_train'], 
            st.session_state['y_train'], 
            model, 
            param_grid
        )
        st.session_state['grid'] = grid
        st.session_state['best_params'] = best_params
        st.session_state['cv_results'] = cv_results
        st.success("Training complete!")



