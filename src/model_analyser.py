from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def analysis(grid_model, X, y, model_name):
    # access the cross-validated results
    cv_results = grid_model.cv_results_

    # extract mean test scores and standard deviations for each parameter combination
    mean_test_scores = cv_results['mean_test_score']

    params = cv_results['params']

    final_results = zip(mean_test_scores, params)
    final_results_list = list(final_results)
    final_results_list_sorted = sorted(final_results_list, key=lambda x: x[0], reverse=True)

    print("all tested hyperparameters and 5 fold cross validation average accuracies:")
    print(final_results_list_sorted, "\n")
    print("Top accuracy of all hyperparameters (5 fold CV) was: ", round(final_results_list_sorted[0][0], 4), "\n")

    # perform nested cross-validation to calculate evaluation metrics
    y_pred = cross_val_predict(grid_model.best_estimator_, X, y, cv=5)

    # calculate evaluation metrics
    confusion_matrix_result = confusion_matrix(y, y_pred)
    classification_report_result = classification_report(y, y_pred)
    mse_score_result = mean_squared_error(y, y_pred)
    r2_score_result = r2_score(y, y_pred)

    # print or use the evaluation metrics as needed
    print("Confusion Matrix:\n", confusion_matrix_result)
    print("Classification Report:\n", classification_report_result)
    print("Mean Squared Error:", mse_score_result)
    print("R-squared Score:", r2_score_result)

    y_true = y
    labels = np.unique(y_true)
    df_cm = pd.DataFrame(confusion_matrix_result, columns=labels, index=labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt = 'g')  # font size
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
    return final_results_list_sorted
