"""

"""


from icecream import ic
from time import sleep
from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender
from recommender.PearlPuRecommender import PearlPuRecommender
from recommender.BootstrapRecommender import BootstrapRecommender
from recommender.CoRecRecommender import CoRecRecommender
from evaluation.Evaluation import Evaluation



def run_experiment(k: int, which: str) -> None:
    if "u" in which:
        # For each rating in the test set, make a prediction using a 
        # user-based KNN with k = 3
        user_r = UserRecommender(k)
        
        mae = 0
        c = 0
        
        for test in user_r.test_ratings:
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = user_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
                
                prediction_error = abs(predicted_rating - rating)
                mae += prediction_error
                c += 1
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("\n")
                #sleep(1)
                break
            
        mae = mae/c
        ic(c)
        ic(mae)


    # For each rating in the test set, make a prediction using an 
    # item-based KNN with k = 3
    if "i" in which:
        item_r = ItemRecommender(k)
        
        mae = 0
        c = 0
        
        for test in item_r.test_ratings:
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = item_r.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
                
                prediction_error = abs(predicted_rating - rating)
                mae += prediction_error
                c += 1
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("\n")
                #sleep(1)
                break
            
        mae = mae/c
        ic(c)
        ic(mae)
        
    
    if "r" in which:
        bs_r = BootstrapRecommender(k, iterations = 5, additions = 10)
        bs_r.enrich()
        
        mae = 0
        c = 0
        
        print("\n")
        
        for test in bs_r.test_ratings:
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = bs_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)

                prediction_error = abs(predicted_rating - rating)
                mae += prediction_error
                c += 1
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("Stopping\n")
                break
            
        mae = mae/c
        ic(c)
        ic(mae)
        

    if "p" in which:
        pp_r = PearlPuRecommender(k)
                
        for i, test in enumerate(pp_r.test_ratings):
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = pp_r.recursive_prediction(user_id, movie_id)
                
                test["pred_rating"] = predicted_rating
                pp_r.add_prediction(test)
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
                
            except KeyboardInterrupt:
                print("\nStopping\n")
                ic(i)
                break
            #sleep(1)
            
        mae_from_pearl_pu = Evaluation.mean_absolute_error(pp_r.predictions)
        ic(mae_from_pearl_pu)
        
        
    if "c" in which:
        cr_r = CoRecRecommender(k)
        
        mae = 0
        c = 0
        
        '''
        for test in cr_r.test_ratings:
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = cr_r.predict_rating_item_based_nn_wtd(user_id, movie_id)
                
                prediction_error = abs(predicted_rating - rating)
                mae += prediction_error
                c += 1
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("\n")
                #sleep(1)
                break
                
                
        mae = mae/c
        ic(c)
        ic(mae)
        '''
                
    