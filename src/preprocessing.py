import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import feature_engineering
from matplotlib import pyplot as plt


def preprocess(train_DF, test_DF):
    # run each function of preprocessing in this file
    print("Data imbalance: ", train_DF['imdb_score_binned'].value_counts())

    # feature engineering
    train_DF_engineered, test_DF_engineered = feature_engineering.feature_engineering(train_DF, test_DF)

    # initialisation
    train_DF_features, test_DF = initialise(train_DF_engineered, test_DF_engineered)

    # make a histogram
    plt.figure(figsize=(8, 5))
    plt.scatter(train_DF_features['imdb_score_binned'], train_DF_features['num_voted_users'], alpha=0.5)
    plt.title("histogram of train data before scaling and outlier removal")
    plt.xlabel('imdb_score_binned')
    plt.xticks([0, 1, 2, 3, 4])
    plt.ylabel('num_voted_users')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(train_DF_features['movie_facebook_likes'], train_DF_features['num_voted_users'], alpha=0.5)
    plt.title("scatterplot of movie_facebook_likes against num_voted_users before scaling and outlier removal")
    plt.xlabel('movie_facebook_likes')
    plt.ylabel('num_voted_users')
    plt.show()

    # removing outliers
    train_DF_features_no_outliers = remove_outliers(train_DF_features)
    print("after removing outliers, train_df is shape: ", train_DF_features_no_outliers.shape)


    # end of initialisation, get the label
    train_DF_features_cleaned = train_DF_features_no_outliers.drop("imdb_score_binned", axis=1)
    train_DF_label = train_DF_features_no_outliers["imdb_score_binned"]
    train_DF_label_df = train_DF_label.to_frame()

    # imputing
    train_DF_features_clean, test_DF_cleaned = clean_data_mean(train_DF_features_cleaned, test_DF)
    # feature_selection
    train_DF_features_selected, test_DF_selected = feature_selection(train_DF_label_df, train_DF_features_clean, test_DF_cleaned)

    # scaling
    scaled_train_DF, scaled_test_DF = scaling_max_abs(train_DF_features_clean, test_DF_cleaned)

    # weighting
    # weighted_train_DF, weighted_test_DF = apply_weighting(scaled_train_DF,train_DF_label_df, scaled_test_DF)

    # PCA
    # pca_train_DF, pca_test_DF = apply_pca(scaled_train_DF, scaled_test_DF, n_components=)

    # handling class imbalance
    # smote = SMOTE(sampling_strategy='auto', random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(scaled_train_DF, train_DF_label_df)

   #  print("\nAfter handling class imbalance with SMOTE, the train data shape was: ", X_resampled.shape)

    # make a histogram after all preprocessing
    # make a histogram
    plt.figure(figsize=(8, 5))
    plt.scatter(train_DF_label_df, scaled_train_DF['num_voted_users'], alpha=0.5)
    plt.title("histogram of train data after scaling and outlier removal")
    plt.xlabel('imdb_score_binned')
    plt.xticks([0, 1, 2, 3, 4])
    plt.ylabel('num_voted_users')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(scaled_train_DF['movie_facebook_likes'], scaled_train_DF['num_voted_users'], alpha=0.5)
    plt.title("scatterplot of movie_facebook_likes against num_voted_users after scaling and outlier removal")
    plt.xlabel('movie_facebook_likes')
    plt.ylabel('num_voted_users')
    plt.show()

    return train_DF_label_df, scaled_train_DF, scaled_test_DF


# this doesnt work with sparse matrices
def scaling_standard(train_DF_features_selected, test_DF_selected):
    scaler = StandardScaler()
    train_DF_features_scaled = scaler.fit_transform(train_DF_features_selected)
    scaled_train_DF = pd.DataFrame(train_DF_features_scaled, columns=train_DF_features_selected.columns)
    test_DF_scaled = scaler.transform(test_DF_selected)
    scaled_test_DF = pd.DataFrame(test_DF_scaled, columns=test_DF_selected.columns)

    return scaled_train_DF, scaled_test_DF


# in order to be sensitive to the sparse matrices
def scaling_max_abs(train_DF_features_selected, test_DF_selected):
    scaler = MaxAbsScaler()
    train_DF_features_scaled = scaler.fit_transform(train_DF_features_selected)
    scaled_train_DF = pd.DataFrame(train_DF_features_scaled, columns=train_DF_features_selected.columns)
    test_DF_scaled = scaler.transform(test_DF_selected)
    scaled_test_DF = pd.DataFrame(test_DF_scaled, columns=test_DF_selected.columns)

    return scaled_train_DF, scaled_test_DF


def apply_weighting(train_DF_features, train_DF_label, test_DF):
    # Fit logistic regression with L1 regularization
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(train_DF_features, train_DF_label)

    # Get feature weights
    feature_weights = model.coef_[0]

    # Apply weights to features
    weighted_train_DF = train_DF_features * feature_weights
    weighted_test_DF = test_DF * feature_weights

    return weighted_train_DF, weighted_test_DF
def feature_selection(train_DF_label, train_DF_features, test_DF):
    num_top_features = 25
    feature_scores = {}

    for feature in train_DF_features:
        mi_score = mutual_info_classif(train_DF_features[[feature]], train_DF_label["imdb_score_binned"])
        feature_scores[feature] = mi_score
    print("\n Mutual information scores:", feature_scores, "\n")
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n Sorted features by mutual information scores:", sorted_features, "\n")
    top_features = [feature[0] for feature in sorted_features[:num_top_features]]

    features_to_keep = []
    for feature in train_DF_features.columns.tolist():
        keep = False
        for top in top_features:
            if feature == top:
                keep = True
                break  # No need to continue checking if the feature is found
        if keep:
            features_to_keep.append(feature)

    # Filter the DataFrames outside the loop
    train_DF_features_selected = train_DF_features[features_to_keep]
    test_DF_selected = test_DF[features_to_keep]
    return train_DF_features_selected, test_DF_selected


def apply_pca(train_DF_features_selected, test_DF_selected, n_components):
    pca = PCA(n_components=n_components)
    pca_train_DF_features = pca.fit_transform(train_DF_features_selected)
    pca_test_DF_features = pca.transform(test_DF_selected)
    pca_train_DF = pd.DataFrame(pca_train_DF_features, columns=[f'PCA_{i + 1}' for i in range(n_components)])
    pca_test_DF = pd.DataFrame(pca_test_DF_features, columns=[f'PCA_{i + 1}' for i in range(n_components)])
    return pca_train_DF, pca_test_DF

def clean_data_knn(data_train, data_test):

    # make a copy of the original data to avoid modifying it directly
    data_cleaned_train = data_train.copy()
    data_cleaned_test = data_test.copy()

    # identify columns where zeros are not to be replaced
    columns_to_exclude = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]

    # Replace zeros with NaNs only in non-genre columns
    non_genre_columns = [col for col in data_cleaned_train.columns if col not in columns_to_exclude]

    data_cleaned_train[non_genre_columns] = data_cleaned_train[non_genre_columns].replace(0, np.nan)
    data_cleaned_test[non_genre_columns] = data_cleaned_test[non_genre_columns].replace(0, np.nan)

    # Impute NaNs using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    data_imputed_train = imputer.fit_transform(data_cleaned_train)
    data_cleaned_train_final = pd.DataFrame(data_imputed_train, columns=data_cleaned_train.columns)

    data_imputed_test = imputer.transform(data_cleaned_test)
    data_cleaned_test_final = pd.DataFrame(data_imputed_test, columns=data_cleaned_train.columns)


    return data_cleaned_train_final, data_cleaned_test_final

def clean_data_mean(data_train, data_test):

    # make a copy of the original data to avoid modifying it directly
    data_cleaned_train = data_train.copy()
    data_cleaned_test = data_test.copy()

    # identify columns where zeros are not to be replaced
    columns_to_exclude = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]

    # Replace zeros with NaNs only in non-genre columns
    non_genre_columns = [col for col in data_cleaned_train.columns if col not in columns_to_exclude]

    # Replace zeros with NaNs
    data_cleaned_train[non_genre_columns] = data_cleaned_train[non_genre_columns].replace(0, np.nan)
    data_cleaned_test[non_genre_columns] = data_cleaned_test[non_genre_columns].replace(0, np.nan)

    # Impute NaNs using SimpleImputer with mean strategy
    imputer = SimpleImputer(strategy='mean')
    data_imputed_train = imputer.fit_transform(data_cleaned_train)
    data_cleaned_train_final = pd.DataFrame(data_imputed_train, columns=data_cleaned_train.columns)

    data_imputed_test = imputer.transform(data_cleaned_test)
    data_cleaned_test_final = pd.DataFrame(data_imputed_test, columns=data_cleaned_train.columns)

    return data_cleaned_train_final, data_cleaned_test_final
def remove_outliers(data):
    # Calculate Z-score for each column
    z_scores = (data - data.mean()) / data.std()

    threshold = 8

    # Find indices of outliers
    outlier_indices = (z_scores > threshold).any(axis=1)

    # Remove outliers from the data
    data_no_outliers = data[~outlier_indices]

    return data_no_outliers

def initialise(train_DF, test_DF):
    # getting engineered features from file

    doc2vec_plot_keywords_feature_train = np.load('data/features_doc2vec/train_doc2vec_features_plot_keywords.npy')
    doc2vec_plot_keywords_feature_test = np.load('data/features_doc2vec/test_doc2vec_features_plot_keywords.npy')

    fasttext_title_embeddings_feature_train = np.load('data/features_fasttext/train_fasttext_title_embeddings.npy')
    fasttext_title_embeddings_feature_test = np.load('data/features_fasttext/test_fasttext_title_embeddings.npy')

    # replacing test and train data with their respective engineered sets
    train_DF["plot_keywords"] = doc2vec_plot_keywords_feature_train
    test_DF["plot_keywords"] = doc2vec_plot_keywords_feature_test

    train_DF["title_embedding"] = fasttext_title_embeddings_feature_train
    test_DF["title_embedding"] = fasttext_title_embeddings_feature_test

    # additional preprocessing

    # content rating
    label_encoder = LabelEncoder()
    train_DF["content_rating"] = label_encoder.fit_transform(train_DF["content_rating"])
    test_DF["content_rating"] = label_encoder.transform(test_DF["content_rating"])

    # language
    combined_values = pd.concat([train_DF["language"], test_DF["language"]])

    label_encoder = LabelEncoder()
    label_encoder.fit(combined_values)

    train_DF["language"] = label_encoder.transform(train_DF["language"])
    test_DF["language"] = label_encoder.transform(test_DF["language"])

    # country
    combined_values = pd.concat([train_DF["country"], test_DF["country"]])

    label_encoder = LabelEncoder()
    label_encoder.fit(combined_values)

    train_DF["country"] = label_encoder.transform(train_DF["country"])
    test_DF["country"] = label_encoder.transform(test_DF["country"])

    # movie name has no statisical significance
    train_DF = train_DF.drop("movie_title", axis=1)
    test_DF = test_DF.drop("movie_title", axis=1)

    # this column has little statistical signifiance
    train_DF = train_DF.drop("actor_3_name", axis=1)
    test_DF = test_DF.drop("actor_3_name", axis=1)

    # temp fix, deal with this later
    train_DF = train_DF.drop("Film-Noir", axis=1)

    return train_DF, test_DF


def check_categorical_columns(data):
    categorical_columns = data.select_dtypes(include=['category']).columns
    print("Categorical columns:")
    print(categorical_columns)

