import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

class visualizer:
    @staticmethod
    def plot_scores(results):
        score = results['mean_test_score']
        params = [str(p) for p in results['params']]
        fig, ax = plt.subplots()
        ax.bar(range(len(score)), score)
        ax.set_xlabel('Parameter Combinations')
        ax.set_ylabel('Mean Score')
        ax.set_title('Grid Search Scores')
        st.pyplot(fig)

    @staticmethod
    def display_best_params(best_params):
        st.bar_chart(best_params.values())


