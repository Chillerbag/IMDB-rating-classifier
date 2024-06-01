# this is a meta classifier of decision trees. Might be more useful to do another model in the stacking
# seeing as we use a decision tree already


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src import hyperparameter_tuning
from src import model_analyser


def random_forest_analysis(train_DF_label, train_DF_features):
    print("random forest classifier --")
    model_name = "Random Forest Classifier"

    # setup
    X = train_DF_features
    y = train_DF_label.values.ravel()
    clf = RandomForestClassifier()

    # hyperparameter tuning
    param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}

    best_rf, grid_rf = hyperparameter_tuning.tune(clf, param_grid_rf, X, y)

    # the tuning uses 5-fold cross validation, so this is used to test our results
    print("Best Random Forest parameters:", best_rf, "\n")

    final_results = model_analyser.analysis(grid_rf, X, y, model_name)

    return final_results, best_rf
