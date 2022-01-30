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

ic.disable()
#ic.configureOutput(includeContext=True)



def run_experiment(k: int, which: str) -> None:
    if "u" in which:
        # For each rating in the test set, make a prediction using a 
        # user-based KNN with k = 3
        user_r = UserRecommender(k)
        
        print("\nRunning User Recommender\n")

        for i, test in enumerate(user_r.test_ratings):
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = user_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
                test["pred_rating"] = predicted_rating
                user_r.add_prediction(test)
                              
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("\nStopping\n")
                print(i)
                #sleep(1)
                break
            
        mae_from_user_r = Evaluation.mean_absolute_error(user_r.predictions)
        print(mae_from_user_r)


    # For each rating in the test set, make a prediction using an 
    # item-based KNN with k = 3
    if "i" in which:
        item_r = ItemRecommender(k)
        
        print("\nRunning Item Recommender\n")

        for i, test in enumerate(item_r.test_ratings):
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = item_r.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
                test["pred_rating"] = predicted_rating
                item_r.add_prediction(test)
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("\nStopping\n")
                print(i)
                #sleep(1)
                break
            
        mae_from_item_r = Evaluation.mean_absolute_error(item_r.predictions)
        print(mae_from_item_r)
        
    
    if "b" in which:
        bs_r = BootstrapRecommender(k, iterations = 5, additions = 10)
        print("\nEnriching Bootstrap Recommender\n")

        bs_r.enrich()

        print("\nRunning Bootstrap Recommender\n")
        
        for i, test in enumerate(bs_r.test_ratings):
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = bs_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
                test["pred_rating"] = predicted_rating
                bs_r.add_prediction(test)
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("\nStopping\n")
                print(i)
                break
            
        mae_from_bs_r = Evaluation.mean_absolute_error(bs_r.predictions)
        print(mae_from_bs_r)
        

    if "p" in which:
        pp_r = PearlPuRecommender(k)

        print("\nRunning Pearl Pu Recommender\n")

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
                print(i)
                break
            #sleep(1)
            
        mae_from_pearl_pu = Evaluation.mean_absolute_error(pp_r.predictions)
        print(mae_from_pearl_pu)
        
        
    if "c" in which:
        cr_r = CoRecRecommender(k)
    
        print("\nRunning Co Rec Recommender\n")

        '''
        for i, test in enumerate(cr_r.test_ratings):
            try:
                user_id = test['user_id']
                movie_id = test['movie_id']
                rating = test['rating']
                
                predicted_rating = cr_r.predict_rating_item_based_nn_wtd(user_id, movie_id)
                
                test["pred_rating"] = predicted_rating
                cr_r.add_prediction(test)
                
                #print(user_id, movie_id, rating, round(predicted_rating, 1))
            except KeyboardInterrupt:
                print("\nStopping\n")
                print(i)
                #sleep(1)
                break
                
        mae_from_cr_r = Evaluation.mean_absolute_error(cr_r.predictions)
        print(mae_from_cr_r)
        '''
                
    