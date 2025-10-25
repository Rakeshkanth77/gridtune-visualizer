import streamlit as st
from data_handler.data_handler import DataHandler
from model_trainer.model_trainer import ModelTrainer
import pandas as pd 
from visualizer.visualizer import visualizer
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


if model == "Logistic Regression":
    c_values = st.sidebar.multiselect("C values", [0.1,1,10])
    param_grid = {'C': c_values, 'penalty': ['l2']}
elif model == "SVC":
    c_values = st.sidebar.multiselect("C values", [0.1,1,10])
    kernels = st.sidebar.multiselect("Kernels", ['linear', 'rbf'])
    param_grid = {'C': c_values, 'kernel': kernels}

try:
    if 'X_train' in st.session_state and st.sidebar.button("Train Model"):
        with st.spinner("tunning hyperparameters..."):
            grid , best_params, results = ModelTrainer.train(
                st.session_state['X_train'], 
                st.session_state['y_train'], 
                model, 
                param_grid
            )
            st.session_state.grid = grid
            st.session_state.best_params = best_params
            st.session_state.results = results
            st.success("Training complete!")
except ValueError as e:
    st.error(f"Invalid params: {e}. Adjust grid")
except Exception as e:
    st.error(f"Training failed: {e}. Try smaller grid")


if "results" in st.session_state:
    results_df = pd.DataFrame({
        'Params': [str(params) for params in st.session_state.results['params']],
        'Mean Score' : st.session_state.results['mean_test_score'],
        'Std Score' : st.session_state.results['std_test_score'],
    })
    st.subheader("Results Table")
    st.dataframe(results_df)

    st.subheader("Best Hyperparameters")
    best_score= st.session_state.grid.best_score_
    st.metric("Best Score", f"{best_score:.4f}")
    st.json(st.session_state.best_params)


    visualizer.plot_scores(st.session_state.results)

    df = pd.DataFrame(st.session_state.results)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", csv, "grid_results.csv")

