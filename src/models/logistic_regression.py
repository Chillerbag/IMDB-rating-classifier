from sklearn.linear_model import LogisticRegression
from src import hyperparameter_tuning
from src import model_analyser

def logistic_regression_analysis(train_DF_label, train_DF_features):
    print("logisitic regression classifier --")

    # setup
    X = train_DF_features
    y = train_DF_label.values.ravel()

    # initialise the model. Pass to hyperparameter_tuning.
    clf = LogisticRegression(max_iter=50000)

    # hyperparameter tuning
    param_grid_lr = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}

    best_lr, grid_lr = hyperparameter_tuning.tune(clf, param_grid_lr, X, y)

    # the tuning uses 5-fold cross validation, so this is used to test our results
    print("Best Logistic regression parameters:", best_lr, "\n")

    final_results = model_analyser.analysis(grid_lr, X, y)


    return final_results, best_lr
