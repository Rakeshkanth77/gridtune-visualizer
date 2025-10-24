from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    @staticmethod
    def train(X_train, y_train, model_name, param_grid):
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_name == "SVC":
            model = SVC()
        else:
            raise ValueError("Unsupported model")
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        return grid, grid.best_params_, grid.cv_results_