import pandas as pd

# functions from other files
import preprocessing
from models import logistic_regression
from models import decision_tree
from models import random_forest
from models import support_vector
from models import KNN
from models import NB
from models import stacking
from models import bagging_and_voting
import matplotlib.pyplot as plt
import os

def main():
    # setup storage of results
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # get the file path for each dataset
    file_path_train = 'data/train_dataset.csv'
    file_path_test = 'data/test_dataset.csv'
    # set up dataframes of all data
    train_DF = pd.read_csv(file_path_train, sep=',')
    test_DF = pd.read_csv(file_path_test, sep=',')
    test_ids = test_DF["id"]

    train_DF_label, train_DF_features, test_DF = preprocessing.preprocess(train_DF, test_DF)

    print("--- GENERAL MODEL PERFORMANCE ON DATASET--- \n")
    # perform analysis on all desired models. Find which one performs best on K fold
    final_results_lr, best_lr = logistic_regression.logistic_regression_analysis(train_DF_label, train_DF_features)
    final_results_dt, best_dt = decision_tree.decision_tree_analysis(train_DF_label, train_DF_features)
    final_results_rf, best_rf = random_forest.random_forest_analysis(train_DF_label, train_DF_features)
    final_results_svc, best_svc = support_vector.SVC_analysis(train_DF_label, train_DF_features)
    final_results_knn, best_knn = KNN.KNN_analysis(train_DF_label, train_DF_features)
    final_results_nb = NB.NaiveBayes_analysis(train_DF_label, train_DF_features, 5)

    top_accuracies = [(final_results_lr[0][0], best_lr), (final_results_dt[0][0], best_dt), (final_results_rf[0][0], best_rf), (final_results_svc[0][0], best_svc), (final_results_knn[0][0], best_knn)]
    sorted_accuracies = sorted(top_accuracies, reverse=True)
    # use the best results from these to get the estimators for combination.

    estimators = [sorted_accuracies[0][1], sorted_accuracies[1][1], sorted_accuracies[2][1]]
    print("estimators being used:", estimators)
    stacking_clf, stacking_avg_acc = stacking.stacking_analysis(train_DF_label, train_DF_features, estimators)
    bagging_clf, voting_clf, bgvote_avg_acc = bagging_and_voting.bagging_voting_analysis(train_DF_label, train_DF_features, estimators)

    # decide between bagging + voting, stacking, or random forest:
    best_models = [(final_results_rf[0][0], "rf"),  (stacking_avg_acc, "stk"), (bgvote_avg_acc, "bgv")]
    top_model = max(best_models, key=lambda x: x[0])
    top_model_id = top_model[1]

    match top_model_id:
        case "rf":
            clf = best_rf
        case "stk":
            clf = stacking_clf
        case "bgv":
            clf = voting_clf
    print("the best model, chosen for use, was:", clf, "\n")
    clf.fit(train_DF_features, train_DF_label)
    preds = clf.predict(test_DF)
    preds_df = pd.Series(preds, name="imdb_score_binned")
    submission = pd.concat([test_ids, preds_df], axis=1)
    submission.to_csv(f"{results_dir}/submission_final.csv", index=False)



    # Collect accuracy scores of all models
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC', 'KNN', 'Naive Bayes', 'Stacking', 'Bagging and Voting']
    accuracies = [final_results_lr[0][0], final_results_dt[0][0], final_results_rf[0][0], final_results_svc[0][0],
                  final_results_knn[0][0], final_results_nb, stacking_avg_acc, bgvote_avg_acc]

    # graph to show all accuracies:
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Performance of Different Models without feature selection')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for accuracy
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate the bars with the accuracy values
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center')

    plt.tight_layout()
    plt.show()

    # Save results for stacking and bagging/voting
    combined_models = {
        'Stacking': (stacking_clf, stacking_avg_acc),
        'Bagging_and_Voting': (voting_clf, bgvote_avg_acc)
    }

    for model_name, (model, accuracy) in combined_models.items():
        print(f"Fitting and predicting with {model_name}")
        model.fit(train_DF_features, train_DF_label)
        preds = model.predict(test_DF)
        preds_df = pd.Series(preds, name="imdb_score_binned")
        output = pd.concat([test_ids, preds_df], axis=1)
        output.to_csv(f"{results_dir}/{model_name}_results.csv", index=False)

    # for each other model
    models = {
        'Logistic_Regression': (best_lr, final_results_lr[0][0]),
        'Decision_Tree': (best_dt, final_results_dt[0][0]),
        'Random_Forest': (best_rf, final_results_rf[0][0]),
        'SVC': (best_svc, final_results_svc[0][0]),
        'KNN': (best_knn, final_results_knn[0][0]),
        'Naive_Bayes': (None, final_results_nb)
    }

    # Fit, predict, and save results for each model
    for model_name, (model, accuracy) in models.items():
        print(f"Fitting and predicting with {model_name}")
        if model is not None:
            model.fit(train_DF_features, train_DF_label)
            preds = model.predict(test_DF)
            preds_df = pd.Series(preds, name="imdb_score_binned")
            submission = pd.concat([test_ids, preds_df], axis=1)
            submission.to_csv(f"{results_dir}/{model_name}_results.csv", index=False)

    print("finished")
    return 1

if __name__ == "__main__":
    main()
