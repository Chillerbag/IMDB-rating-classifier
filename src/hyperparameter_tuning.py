
from sklearn.model_selection import GridSearchCV
def tune(model, grid, X, y):
    grid_model = GridSearchCV(model, grid, cv=5)
    grid_model.fit(X, y)
    best = grid_model.best_estimator_

    return best, grid_model

