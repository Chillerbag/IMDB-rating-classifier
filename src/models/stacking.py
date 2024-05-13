from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score

def stacking_analysis(train_DF_label, train_DF_features, estimators):
    print("Stacking classifier --")

    # setup
    X = train_DF_features
    y = train_DF_label.values.ravel()
    named_estimators = [('estimator{}'.format(i), estimator) for i, estimator in enumerate(estimators)]
    # uses the best splitter as the default for this one
    #TODO hyperaparmeter tuning here
    clf = StackingClassifier(estimators=named_estimators, final_estimator=LogisticRegression(C=10, max_iter=10000))
    scores = cross_val_score(clf, X, y, cv=5)
    average_accuracy = scores.mean()
    print("cross validation accuracy average for stacking:", round(average_accuracy, 4), "\n")
    print("models used: ", named_estimators, "\n")

    return clf, average_accuracy
