from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src import hyperparameter_tuning
from src import model_analyser

def decision_tree_analysis(train_DF_label, train_DF_features):
    print("Decision tree classifier --")
    model_name = "Decision Tree Classifier"
    # setup
    X = train_DF_features
    y = train_DF_label.values.ravel()

    # initialise the model. Pass to hyperparameter_tuning.
    clf = DecisionTreeClassifier()

    # hyperparameter tuning
    param_grid_dt = {'criterion': ["gini", "entropy", "log_loss"], 'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}

    best_dt, grid_dt = hyperparameter_tuning.tune(clf, param_grid_dt, X, y)

    # the tuning uses 5-fold cross validation, so this is used to test our results
    print("Best Decision tree parameters:", best_dt, "\n")

    final_results = model_analyser.analysis(grid_dt, X, y, model_name)
    return final_results, best_dt
