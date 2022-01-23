"""

"""


from icecream import ic
from time import sleep
from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender


# For each rating in the test set, make a prediction using a 
# user-based KNN with k = 3

def run_experiment(k, which):
    if "u" in which:
        user_r = UserRecommender(k)
        
        for test in user_r.test_ratings:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = user_r.predict_rating_user_based_nn_wtd(user_id, movie_id)
            
            print(user_id, movie_id, rating, round(predicted_rating, 1))
            print("\n")
            #sleep(1)
            break


    # For each rating in the test set, make a prediction using an 
    # item-based KNN with k = 3
    if "i" in which:
        item_r = ItemRecommender(k)
        
        for test in item_r.test_ratings:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = item_r.predict_rating_item_based_nn_wtd(user_id, movie_id)
            
            print(user_id, movie_id, rating, round(predicted_rating, 1))
            #sleep(1)
            break
