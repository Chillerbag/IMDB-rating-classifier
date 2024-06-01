import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cpi
from sklearn.preprocessing import LabelEncoder


def feature_engineering(train_DF, test_DF):

    # this attempt caused massive overfitting
    """
    # Calculate director reputation score
    director_avg_rating = train_DF.groupby('director_name')['imdb_score_binned'].mean()

    # Replace director names with their reputation scores in the training set
    train_DF['director_reputation'] = train_DF['director_name'].map(director_avg_rating)

    # Handle directors not in the training set
    test_DF['director_reputation'] = test_DF['director_name'].map(director_avg_rating).fillna(0)

    # Calculate actor reputation score
    actor_avg_rating = train_DF.groupby('actor_1_name')['imdb_score_binned'].mean()

    # Replace actor names with their reputation scores in the training set
    train_DF['actor_reputation'] = train_DF['actor_1_name'].map(actor_avg_rating)

    # Handle actors not in the training set
    test_DF['actor_reputation'] = test_DF['actor_1_name'].map(actor_avg_rating).fillna(0)
    """


    # this column has fairly high correlation.
    director_name_frequency = train_DF['director_name'].value_counts()
    actor1_name_frequency = train_DF['actor_1_name'].value_counts()
    actor2_name_frequency = train_DF['actor_2_name'].value_counts()

    # Replace director names with their frequencies in the training set
    train_DF['director_frequency'] = train_DF['director_name'].map(director_name_frequency)
    # Handle directors not in training set
    test_DF['director_frequency'] = test_DF['director_name'].map(director_name_frequency).fillna(0)

    train_DF['actor1_frequency'] = train_DF['actor_1_name'].map(actor1_name_frequency)
    test_DF['actor1_frequency'] = test_DF['actor_1_name'].map(actor1_name_frequency).fillna(0)

    train_DF['actor2_frequency'] = train_DF['actor_2_name'].map(actor2_name_frequency)
    test_DF['actor2_frequency'] = test_DF['actor_2_name'].map(actor2_name_frequency).fillna(0)

    # Drop the original director_name column
    train_DF.drop(['director_name'], axis=1, inplace=True)
    test_DF.drop(['director_name'], axis=1, inplace=True)

    train_DF.drop(['actor_1_name'], axis=1, inplace=True)
    test_DF.drop(['actor_1_name'], axis=1, inplace=True)

    train_DF.drop(['actor_2_name'], axis=1, inplace=True)
    test_DF.drop(['actor_2_name'], axis=1, inplace=True)

    # ------------- DURATION ------------
    # plot a histogram
    plt.hist(train_DF['duration'], bins=3)
    plt.title('Histogram of duration')
    plt.xlabel('duration of movie')
    plt.ylabel('frequency')
    plt.show()
    plt.clf()

    # binning
    bins = [0, 90, 120, 150, 190, 220, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    # train_DF['duration_bins'] = pd.cut(train_DF['duration'], bins=bins, labels=labels, right=False)
    # test_DF['duration_bins'] = pd.cut(test_DF['duration'], bins=bins, labels=labels, right=False)
    # train_DF.drop(['duration'], axis=1, inplace=True)
    # test_DF.drop(['duration'], axis=1, inplace=True)

    # ----------- GROSS --------------

    # used this for inspo https://pieriantraining.com/exploring-inflation-data-with-python/
    # first, lets adjust for inflation.
    start_year = 1913
    end_year = 2023
    cpi_values = {year: cpi.get(year) for year in range(start_year, end_year + 1)}

    base_year = 2023  # year we are adjusting for

    train_DF['inflation_factor'] = train_DF['title_year'].map(lambda year: cpi_values[base_year] / cpi_values[year])
    test_DF['inflation_factor'] = test_DF['title_year'].map(lambda year: cpi_values[base_year] / cpi_values[year])

    # adjust
    train_DF['adjusted_gross'] = train_DF['gross'] * train_DF['inflation_factor']
    test_DF['adjusted_gross'] = test_DF['gross'] * test_DF['inflation_factor']
    # plot a scatterplot
    value_counts = train_DF['adjusted_gross'].value_counts()

    x = value_counts.index
    y = value_counts.values

    plt.scatter(x, y)

    # this graph will always be awful due to scale, but it shows that high budgets are rare.
    plt.title('scatterplot of gross revenue')
    plt.xlabel('gross revenue')
    plt.ylabel('frequency')
    plt.show()
    plt.clf()

    # print("the minimum value of the gross revenue:", min(train_DF['adjusted_gross']))
    # 208
    # print("the maximum value of the gross revenue:", max(train_DF['adjusted_gross']))
    # 4,354,723,778
    # print("the median value of the gross revenue:", (train_DF['adjusted_gross']).median())
    # 49,406,833

    bins = [0, 500000, 1000000, 2500000, 6250000, 12500000, 25000000, 50000000, 500000000, 1000000000, float('inf')]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # train_DF['gross_bins'] = pd.cut(train_DF['adjusted_gross'], bins=bins, labels=labels, right=False)
    # test_DF['gross_bins'] = pd.cut(test_DF['adjusted_gross'], bins=bins, labels=labels, right=False)
    train_DF.drop(['gross', 'inflation_factor'], axis=1, inplace=True)
    test_DF.drop(['gross', 'inflation_factor'], axis=1, inplace=True)

    # -------------- FACEBOOK LIKES -------------
    # all of these values are minorly significant metrics, so combine them all into one.
    # train_DF['cast_likes'] = train_DF['director_facebook_likes'] + train_DF['actor_3_facebook_likes'] + train_DF['actor_2_facebook_likes'] + train_DF['actor_1_facebook_likes'] + train_DF['cast_total_facebook_likes']
    # test_DF['cast_likes'] = test_DF['director_facebook_likes'] + test_DF['actor_3_facebook_likes'] + test_DF['actor_2_facebook_likes'] + test_DF['actor_1_facebook_likes'] + test_DF['cast_total_facebook_likes']
   #  train_DF.drop(['director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes', 'cast_total_facebook_likes'], axis=1, inplace=True)
    # test_DF.drop(
    #     ['director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes', 'cast_total_facebook_likes'],
    #     axis=1, inplace=True)
    # now we bin
    # plt.hist(train_DF['cast_likes'], bins=10)
    # plt.title('Histogram of cast_likes')
    # plt.xlabel('cast_likes')
    # plt.ylabel('frequency')
    # plt.show()
    # plt.clf()

    # print("the minimum value of the cast_likes:", min(train_DF['cast_likes']))
    # 5
    # print("the maximum value of the cast_likes:", max(train_DF['cast_likes']))
    # 655,285
    # print("the median value of the cast_likes:", (train_DF['cast_likes']).median())
    # 2924

    bins = [0, 500, 1000, 2000, 3000, 5000, 10000, 50000, 100000,  float('inf')]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # train_DF['cast_likes_bins'] = pd.cut(train_DF['cast_likes'], bins=bins, labels=labels, right=False)
    # test_DF['cast_likes_bins'] = pd.cut(test_DF['cast_likes'], bins=bins, labels=labels, right=False)
    # train_DF.drop(['cast_likes'], axis=1, inplace=True)
    # test_DF.drop(['cast_likes'], axis=1, inplace=True)

    # -------------- GENRES -------------
    genres_train = train_DF["genres"].str.get_dummies(sep='|')
    genres_test = test_DF["genres"].str.get_dummies(sep='|')

    train_DF = pd.concat([train_DF.drop("genres", axis=1), genres_train], axis=1)
    test_DF = pd.concat([test_DF.drop("genres", axis=1), genres_test], axis=1)

    # -------------- NUM USER FOR REVIEWS -------------
    # plt.hist(train_DF['num_user_for_reviews'], bins=5)
    # plt.title('Histogram of num_user_for_reviews')
    # plt.xlabel('num_user_for_reviews')
    # plt.ylabel('frequency')
    # plt.show()
    # plt.clf()

    # print("the minimum value of num_user_for_reviews:", min(train_DF['num_user_for_reviews']))
    # 4
    # print("the maximum value num_user_for_reviews:", max(train_DF['num_user_for_reviews']))
    # 5060
    # print("the median value of num_user_for_reviews:", (train_DF['num_user_for_reviews']).median())
    # 208

    # bins = [0, 100, 200, 500, 1000, 2500, 4000, float('inf')]
    # labels = [0, 1, 2, 3, 4, 5, 6]
    # train_DF['num_user_for_reviews_bins'] = pd.cut(train_DF['num_user_for_reviews'], bins=bins, labels=labels, right=False)
    # test_DF['num_user_for_reviews_bins'] = pd.cut(test_DF['num_user_for_reviews'], bins=bins, labels=labels, right=False)
    # train_DF.drop(['num_user_for_reviews'], axis=1, inplace=True)
    # test_DF.drop(['num_user_for_reviews'], axis=1, inplace=True)

    # -------------- NUM VOTED USER -------------
    # plt.hist(train_DF['num_voted_users'], bins=5)
    # plt.title('Histogram of num_voted_users')
    # plt.xlabel('num_voted_users')
    # plt.ylabel('frequency')
    # plt.show()
    # plt.clf()

    # print("the minimum value of num_voted_users:", min(train_DF['num_voted_users']))
    # 91
    # print("the maximum value num_voted_users:", max(train_DF['num_voted_users']))
    # 1689764
    # print("the median value of num_voted_users:", (train_DF['num_voted_users']).median())
    # 53874

    # bins = [0, 13468, 26937, 53874, 500000, 1000000, float('inf')]
    # labels = [0, 1, 2, 3, 4, 5]
    # train_DF['num_voted_users_bins'] = pd.cut(train_DF['num_voted_users'], bins=bins, labels=labels,
     #                                              right=False)
    # test_DF['num_voted_users_bins'] = pd.cut(test_DF['num_voted_users'], bins=bins, labels=labels,
    #                                              right=False)
    # train_DF.drop(['num_voted_users'], axis=1, inplace=True)
    # test_DF.drop(['num_voted_users'], axis=1, inplace=True)

    # -------------- NUM CRITIC FOR REVIEWS -------------
    # plt.hist(train_DF['num_critic_for_reviews'], bins=3)
    # plt.title('Histogram of num_critic_for_reviews')
    # plt.xlabel('num_critic_for_reviews')
    # plt.ylabel('frequency')
    # plt.show()
    # plt.clf()

    # print("the minimum value of num_critic_for_reviews:", min(train_DF['num_critic_for_reviews']))
    # 2
    # print("the maximum value num_critic_for_reviews:", max(train_DF['num_critic_for_reviews']))
    # 813
    # print("the median value of num_critic_for_reviews:", (train_DF['num_critic_for_reviews']).median())
    # 137
    # bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, float('inf')]
    # labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # train_DF['num_critic_for_reviews_bins'] = pd.cut(train_DF['num_critic_for_reviews'], bins=bins, labels=labels,
    #                                          right=False)
    # test_DF['num_critic_for_reviews_bins'] = pd.cut(test_DF['num_critic_for_reviews'], bins=bins, labels=labels,
    #                                         right=False)
    # train_DF.drop(['num_critic_for_reviews'], axis=1, inplace=True)
    # test_DF.drop(['num_critic_for_reviews'], axis=1, inplace=True)


    # ------------------ YEAR --------------------------------
    # just labelencoder it
    combined_values = pd.concat([train_DF["title_year"], test_DF["title_year"]])

    label_encoder = LabelEncoder()
    label_encoder.fit(combined_values)

    train_DF["title_year"] = label_encoder.transform(train_DF["title_year"])
    test_DF["title_year"] = label_encoder.transform(test_DF["title_year"])

    # -------------- Movie facebook likes -------------
    # plt.hist(train_DF['movie_facebook_likes'], bins=3)
    plt.title('Histogram of num_critic_for_reviews')
    plt.xlabel('movie_facebook_likes')
    plt.ylabel('frequency')
    # plt.show()
    # plt.clf()

    # print("the minimum value of num_critic_for_reviews:", min(train_DF['num_critic_for_reviews']))
    # 2
    # print("the maximum value num_critic_for_reviews:", max(train_DF['num_critic_for_reviews']))
    # 813
    # print("the median value of num_critic_for_reviews:", (train_DF['num_critic_for_reviews']).median())
    # 137
    # bins = [0, 1000, 5000, 10000, 15000, 20000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000, float('inf')]
    # labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # train_DF['movie_likes_bins'] = pd.cut(train_DF['movie_facebook_likes'], bins=bins, labels=labels,
    #                                                 right=False)
    # test_DF['movie_likes_bins'] = pd.cut(test_DF['movie_facebook_likes'], bins=bins, labels=labels,
    #                                                right=False)
    # train_DF.drop(['movie_facebook_likes'], axis=1, inplace=True)
    # test_DF.drop(['movie_facebook_likes'], axis=1, inplace=True)

    train_DF_engineered, test_DF_engineered = convert_binned_categorical_to_numeric(train_DF, test_DF)

    return train_DF_engineered, test_DF_engineered

def convert_binned_categorical_to_numeric(train_DF, test_DF):
    # Convert binned categorical variables to numeric
    #train_DF['duration_bins'] = train_DF['duration_bins'].astype('int')
    # test_DF['duration_bins'] = test_DF['duration_bins'].astype('int')

    # train_DF['gross_bins'] = train_DF['gross_bins'].astype('int')
    # test_DF['gross_bins'] = test_DF['gross_bins'].astype('int')

    # train_DF['cast_likes_bins'] = train_DF['cast_likes_bins'].astype('int')
    # test_DF['cast_likes_bins'] = test_DF['cast_likes_bins'].astype('int')

    # train_DF['num_user_for_reviews_bins'] = train_DF['num_user_for_reviews_bins'].astype('int')
    # test_DF['num_user_for_reviews_bins'] = test_DF['num_user_for_reviews_bins'].astype('int')

    # train_DF['num_voted_users_bins'] = train_DF['num_voted_users_bins'].astype('int')
    # test_DF['num_voted_users_bins'] = test_DF['num_voted_users_bins'].astype('int')

    # rain_DF['num_critic_for_reviews_bins'] = train_DF['num_critic_for_reviews_bins'].astype('int')
    # test_DF['num_critic_for_reviews_bins'] = test_DF['num_critic_for_reviews_bins'].astype('int')

    # train_DF['movie_likes_bins'] = train_DF['movie_likes_bins'].astype('int')
    # test_DF['movie_likes_bins'] = test_DF['movie_likes_bins'].astype('int')

    return train_DF, test_DF