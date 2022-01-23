from UserRecommender import UserRecommender
from ItemRecommender import ItemRecommender
from icecream import ic
from time import sleep

# Create the recommender (also reads in the movie and user data)
k = 3
user_r = UserRecommender()
item_r = ItemRecommender()

# Read in the ratings and splits 80/20 using 2 as the random number seed
#r.load_ratings("ratings.txt", 20, 2)


# Prepare the experiment
#est_set = r.test_ratings


# For each rating in the test set, make a prediction using a 
# user-based KNN with k = 3
"""for test in user_r.test_ratings:
    user_id = test['user_id']
    movie_id = test['movie_id']
    rating = test['rating']
    predicted_rating = user_r.predict_rating_user_based_nn_wtd(user_id, movie_id, k)
    print(user_id, movie_id, rating, round(predicted_rating, 1))
    sleep(1)"""


# For each rating in the test set, make a prediction using an 
# item-based KNN with k = 3
for test in item_r.test_ratings:
    user_id = test['user_id']
    movie_id = test['movie_id']
    rating = test['rating']
    predicted_rating = item_r.predict_rating_item_based_nn_wtd(user_id, movie_id, k)
    print(user_id, movie_id, rating, round(predicted_rating, 1))
    sleep(1)
