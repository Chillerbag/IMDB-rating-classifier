import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def preprocess(train_DF, test_DF):
    # run each function of preprocessing in this file

    # initialisation
    train_DF_label, train_DF_features, test_DF = initialise(train_DF, test_DF)

    # scaling

    # feature_selection
    return train_DF_label, train_DF_features, test_DF


def initialise(train_DF, test_DF):
    # getting engineered features from file
    countVec_actor_1_feature_train = np.load('data/features_countvec/train_countvec_features_actor_1_name.npy')
    countVec_actor_1_feature_test = np.load('data/features_countvec/test_countvec_features_actor_1_name.npy')

    countVec_actor_2_feature_train = np.load('data/features_countvec/train_countvec_features_actor_2_name.npy')
    countVec_actor_2_feature_test = np.load('data/features_countvec/test_countvec_features_actor_2_name.npy')

    countVec_director_feature_train = np.load('data/features_countvec/train_countvec_features_director_name.npy')
    countVec_director_feature_test = np.load('data/features_countvec/test_countvec_features_director_name.npy')

    doc2vec_plot_keywords_feature_train = np.load('data/features_doc2vec/train_doc2vec_features_plot_keywords.npy')
    doc2vec_plot_keywords_feature_test = np.load('data/features_doc2vec/test_doc2vec_features_plot_keywords.npy')

    # dont want this.
    # doc2vec_genre_feature_train = np.load('data/features_doc2vec/train_doc2vec_features_genre.npy')
    # doc2vec_genre_feature_test = np.load('data/features_doc2vec/test_doc2vec_features_genre.npy')

    fasttext_title_embeddings_feature_train = np.load('data/features_fasttext/train_fasttext_title_embeddings.npy')
    fasttext_title_embeddings_feature_test = np.load('data/features_fasttext/test_fasttext_title_embeddings.npy')

    # replacing test and train data with their respective engineered sets
    # TODO: COMBINE THIS INTO 2 LINES
    train_DF["actor_1_name"] = countVec_actor_1_feature_train
    test_DF["actor_1_name"] = countVec_actor_1_feature_test

    train_DF["actor_2_name"] = countVec_actor_2_feature_train
    test_DF["actor_2_name"] = countVec_actor_2_feature_test

    train_DF["director_name"] = countVec_director_feature_train
    test_DF["director_name"] = countVec_director_feature_test

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
    test_DF["language"] = label_encoder.transform(test_DF["language"])

    # country
    train_DF["country"] = label_encoder.fit_transform(train_DF["country"])
    test_DF["country"] = label_encoder.transform(test_DF["country"])

    # movie name #TODO: figure out what to do with this column
    train_DF = train_DF.drop("movie_title", axis=1)
    test_DF = test_DF.drop("movie_title", axis=1)

    # actor 3 #TODO: figure otu what to do with this column
    train_DF = train_DF.drop("actor_3_name", axis=1)
    test_DF = test_DF.drop("actor_3_name", axis=1)

    # genres!
    genres_train = train_DF["genres"].str.get_dummies(sep='|')
    genres_test = test_DF["genres"].str.get_dummies(sep='|')

    train_DF = pd.concat([train_DF.drop("genres", axis=1), genres_train], axis=1)
    test_DF = pd.concat([test_DF.drop("genres", axis=1), genres_test], axis=1)

    # end of initialisation
    train_DF_label = train_DF["imdb_score_binned"]
    train_DF_label_df = train_DF_label.to_frame()
    print(train_DF_label_df["imdb_score_binned"])

    train_DF_features = train_DF.drop("imdb_score_binned", axis=1)

    # temp fix, deal with this later
    train_DF_features = train_DF_features.drop("Film-Noir", axis=1)

    return train_DF_label, train_DF_features, test_DF
