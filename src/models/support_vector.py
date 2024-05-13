# this is a meta classifier of decision trees. Might be more useful to do another model in the stacking
# seeing as we use a decision tree already


from sklearn.svm import SVC
from src import hyperparameter_tuning
from src import model_analyser


def SVC_analysis(train_DF_label, train_DF_features):
    print("Support vector classifier --")
    # setup
    X = train_DF_features
    y = train_DF_label.values.ravel()

    # initialise the model. Pass to hyperparameter_tuning.

    # uses the best splitter as the default for this one
    clf = SVC(kernel="rbf")

    # hyperparameter tuning
    param_grid_svc = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}

    best_svc, grid_svc = hyperparameter_tuning.tune(clf, param_grid_svc, X, y)

    # the tuning uses 5 fold cross validation, so this is used to test our results
    print("Best SVC parameters:", best_svc, "\n")

    final_results = model_analyser.analysis(grid_svc, X, y)

    return final_results, best_svc
