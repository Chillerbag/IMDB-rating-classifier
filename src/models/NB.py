from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def NaiveBayes_analysis(train_DF_label, train_DF_features, n_splits=5):
    print("Naive Bayes classifier --")
    # Setup
    X = train_DF_features
    y = train_DF_label.values.ravel()

    # Initialize the model
    clf = GaussianNB()

    # Perform k-fold cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=n_splits)

    # Analyze cross-validation scores
    print("Cross-validation accuracy average for Naive Bayes:", cv_scores.mean(), "\n")


    return cv_scores.mean()
