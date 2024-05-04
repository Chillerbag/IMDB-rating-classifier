import pandas as pd

# functions from other files
import preprocessing


"""
--------------TODO-----------------
1) make the countvec features dense
2) see if scaling is possible. Try multiple methods
3) port mutual info over
4) look into possible feature engineering. 
5) set up grid search for each method
6) feed grid search params into models
7) set up code for each model + k fold testing & some graphs
8) best performing is used in main to make the submission.csv 
----------------------------------
"""



def main():
    # get the file path for each dataset
    file_path_train = 'data/train_dataset.csv'
    file_path_test = 'data/test_dataset.csv'

    # set up dataframes of all data
    train_DF = pd.read_csv(file_path_train, sep=',')
    test_DF = pd.read_csv(file_path_test, sep=',')

    train_DF_label, train_DF_features, test_DF = preprocessing.preprocess(train_DF, test_DF)
    return 0


