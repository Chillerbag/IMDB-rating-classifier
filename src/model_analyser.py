from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
def analysis(grid_model, X, y):
    # Access the cross-validated results
    cv_results = grid_model.cv_results_

    # Extract mean test scores and standard deviations for each parameter combination
    mean_test_scores = cv_results['mean_test_score']

    # You can also access other information such as parameters used for each combination
    params = cv_results['params']

    final_results = zip(mean_test_scores, params)
    final_results_list = list(final_results)
    final_results_list_sorted = sorted(final_results_list, key=lambda x: x[0], reverse=True)

    print("all tested hyperparameters and 5 fold cross validation average accuracies:")
    print(final_results_list_sorted, "\n")
    print("Top accuracy of all hyperparameters (5 fold CV) was: ", round(final_results_list_sorted[0][0], 4), "\n")

    # Perform nested cross-validation to calculate evaluation metrics
    y_pred = cross_val_predict(grid_model.best_estimator_, X, y, cv=5)

    # Calculate evaluation metrics
    confusion_matrix_result = confusion_matrix(y, y_pred)
    classification_report_result = classification_report(y, y_pred)
    mse_score_result = mean_squared_error(y, y_pred)
    r2_score_result = r2_score(y, y_pred)

    # Print or use the evaluation metrics as needed
    print("Confusion Matrix:\n", confusion_matrix_result)
    print("Classification Report:\n", classification_report_result)
    print("Mean Squared Error:", mse_score_result)
    print("R-squared Score:", r2_score_result)

    return final_results_list_sorted
