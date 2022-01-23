"""

"""


from icecream import ic
from time import sleep
from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender
from recommender.PearlPuRecommender import PearlPuRecommender
from recommender.BootstrapRecommender import BootstrapRecommender
from recommender.CoRecRecommender import CoRecRecommender



def run_experiment(k: int, which: str) -> None:
    
    if "u" in which:
        # For each rating in the test set, make a prediction using a 
        # user-based KNN with k = 3
        user_r = UserRecommender(k)
        
        for test in user_r.test_ratings:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = user_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
            
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
            
            predicted_rating = item_r.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
            
            print(user_id, movie_id, rating, round(predicted_rating, 1))
            print("\n")
            #sleep(1)
            break
        
    
    if "r" in which:
        bs_r = BootstrapRecommender(k)
        
        for test in bs_r.test_ratings:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = bs_r.predict_rating_item_based_nn_wtd(user_id, movie_id)
            
            print(user_id, movie_id, rating, round(predicted_rating, 1))
            print("\n")
            #sleep(1)
            break
        

    if "p" in which:
        pp_r = PearlPuRecommender(k)
        
        for test in pp_r.test_ratings:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = pp_r.predict_rating_item_based_nn_wtd(user_id, movie_id)
            
            print(user_id, movie_id, rating, round(predicted_rating, 1))
            print("\n")
            #sleep(1)
            break
        
    if "c" in which:
        cr_r = CoRecRecommender(k)
        
        for test in cr_r.test_ratings:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = cr_r.predict_rating_item_based_nn_wtd(user_id, movie_id)
            
            print(user_id, movie_id, rating, round(predicted_rating, 1))
            print("\n")
            #sleep(1)
            break
    