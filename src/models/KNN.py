
from src import hyperparameter_tuning
from src import model_analyser
from sklearn.neighbors import KNeighborsClassifier

def KNN_analysis(train_DF_label, train_DF_features):
    print("KNN classifier --")
    # setup
    X = train_DF_features
    y = train_DF_label.values.ravel()

    # initialise the model. Pass to hyperparameter_tuning.

    # uses the best splitter as the default for this one
    clf = KNeighborsClassifier()

    # hyperparameter tuning
    param_grid_knn = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

    best_knn, grid_knn = hyperparameter_tuning.tune(clf, param_grid_knn, X, y)

    # the tuning uses 5 fold cross validation, so this is used to test our results
    print("Best KNN parameters:", best_knn, "\n")

    final_results = model_analyser.analysis(grid_knn, X, y)

    return final_results, best_knn
