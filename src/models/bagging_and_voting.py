from src import hyperparameter_tuning
from src import model_analyser
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, VotingClassifier

def bagging_voting_analysis(train_DF_label, train_DF_features, estimators):
    print("Bagging classifier --")
    model_name = "bagging and voting"
    # setup
    X = train_DF_features
    y = train_DF_label.values.ravel()

    param_grid_bg = {'estimator': [estimators[0], estimators[1], estimators[2]]}

    bagging_clf = BaggingClassifier(n_estimators=50, random_state=42)
    best_bg, grid_bg = hyperparameter_tuning.tune(bagging_clf, param_grid_bg, X, y)

    print("Best Bagging classifier parameters:", best_bg, "\n")
    final_results = model_analyser.analysis(grid_bg, X, y,model_name)
    bagged_estimators = [best_bg] + estimators
    named_estimators = [('estimator{}'.format(i), estimator) for i, estimator in enumerate(bagged_estimators)]

    print("Voting classifier w/ bagging as a param --")
    voting_clf = VotingClassifier(estimators=named_estimators, voting='hard')
    scores = cross_val_score(voting_clf, X, y, cv=5)
    average_accuracy = scores.mean()
    print("cross validation accuracy average for voting after bagging", round(average_accuracy, 4), "\n")

    return bagging_clf, voting_clf, average_accuracy
