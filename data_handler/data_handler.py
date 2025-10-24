import pandas as pd 
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

class DataHandler:
    @staticmethod
    def load_data(dataset_name):
        if dataset_name == "Iris":
            data = load_iris()
        elif dataset_name == "Wine":
            data = load_wine()
        else:
            raise ValueError("Unsupported dataset")

        X = pd.DataFrame(data.data, columns= data.feature_names)
        y = pd.Series(data.target)
        return X, y

    @staticmethod
    def split_data(X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
