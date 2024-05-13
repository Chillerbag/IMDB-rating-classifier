import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import feature_engineering


def preprocess(train_DF, test_DF):
    # run each function of preprocessing in this file

    # feature engineering
    train_DF_engineered, test_DF_engineered = feature_engineering.feature_engineering(train_DF, test_DF)

    # initialisation
    train_DF_label, train_DF_features, test_DF = initialise(train_DF_engineered, test_DF_engineered)

    train_DF_features_clean = clean_data(train_DF_features)
    test_DF_cleaned = clean_data(test_DF)

    # feature_selection
    #train_DF_features_selected, test_DF_selected = feature_selection(train_DF_label, train_DF_features, test_DF)
    #print(train_DF_features_selected.columns.tolist())

    # scaling
    scaled_train_DF, scaled_test_DF = scaling_2(train_DF_features_clean, test_DF_cleaned)

    # weighting
    # weighted_train_DF, weighted_test_DF = apply_weighting(scaled_train_DF,train_DF_label, scaled_test_DF)

    # PCA
    # pca_train_DF, pca_test_DF = apply_pca(scaled_train_DF, scaled_test_DF, n_components=)

    # handling class imbalance
    # X_resampled, y_resampled = RandomOverSampler(sampling_strategy='auto').fit_resample(scaled_train_DF, train_DF_label)

    return train_DF_label, scaled_train_DF, scaled_test_DF


# this doesnt work with sparse matrices
def scaling_1(train_DF_features_selected, test_DF_selected):
    scaler = StandardScaler()
    train_DF_features_scaled = scaler.fit_transform(train_DF_features_selected)
    scaled_train_DF = pd.DataFrame(train_DF_features_scaled, columns=train_DF_features_selected.columns)
    test_DF_scaled = scaler.transform(test_DF_selected)
    scaled_test_DF = pd.DataFrame(test_DF_scaled, columns=test_DF_selected.columns)

    return scaled_train_DF, scaled_test_DF


# in order to be sensitive to the sparse matrices
def scaling_2(train_DF_features_selected, test_DF_selected):
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

    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
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

def clean_data(data):
    data.replace(0, np.nan, inplace=True)
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data)
    data_cleaned = pd.DataFrame(data_imputed, columns=data.columns)
    return data_cleaned
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
    train_DF["language"] = label_encoder.fit_transform(train_DF["language"])
    # need to do this because russian is in test but not train
    test_DF["language"] = label_encoder.fit_transform(test_DF["language"])

    # country
    train_DF["country"] = label_encoder.fit_transform(train_DF["country"])
    # need to do this because Peru is in test but not in train
    test_DF["country"] = label_encoder.fit_transform(test_DF["country"])

    # movie name has no statisical significance
    train_DF = train_DF.drop("movie_title", axis=1)
    test_DF = test_DF.drop("movie_title", axis=1)

    # this column has little statistical signifiance
    train_DF = train_DF.drop("actor_3_name", axis=1)
    test_DF = test_DF.drop("actor_3_name", axis=1)

    # end of initialisation
    train_DF_label = train_DF["imdb_score_binned"]
    train_DF_label_df = train_DF_label.to_frame()

    train_DF_features = train_DF.drop("imdb_score_binned", axis=1)

    # temp fix, deal with this later
    train_DF_features = train_DF_features.drop("Film-Noir", axis=1)

    return train_DF_label_df, train_DF_features, test_DF
