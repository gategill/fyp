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
import time
import os
ic.disable()
#ic.configureOutput(includeContext=True)

current_timestamp = int(time.time())

def run_experiment(k: int, which: str, save_results: bool = False) -> None:
    save_path = "./results/{}".format(current_timestamp)
    os.mkdir(save_path)
    
    # For each rating in the test set, make a prediction using a 
    # user-based KNN with k = 3
    lines_result = "algorithm, mae\n"
    if "u" in which:
        u, mae = run_user_rec_experiment(k)
        print(u, mae)
        lines_result += "u_rec_k={}, {}\n".format(k, mae)

    # For each rating in the test set, make a prediction using an 
    # item-based KNN with k = 3
    if "i" in which:
        u, mae = run_item_rec_experiment(k)
        print(u, mae)
        lines_result += "i_rec_k={}, {}\n".format(k, mae)

    if "b" in which:
        u, mae = run_bootstrap_rec_experiment(k)
        print(u, mae)
        lines_result += "bs_rec_k={}, {}\n".format(k, mae)

    if "p" in which:
        u, mae = run_pearlpu_rec_experiment(k)
        print(u, mae)
        lines_result += "pp_rec_k={}, {}\n".format(k, mae)
        
    if "c" in which:
        u, mae = run_corec_rec_experiment(k)
        print(u, mae)
        lines_result += "corec_rec_k={}, {}\n".format(k, mae)
        
    if save_results:
        saved_file = "{}/{}.txt".format(save_path, which)
        
        with open(saved_file, "w") as f:
            f.write(lines_result)


def run_user_rec_experiment(k):
    user_r = UserRecommender(k)
    
    print("\nRunning User Recommender\n")
    print(len(user_r.test_ratings))

    for i, test in enumerate(user_r.test_ratings):
        try:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = user_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
            test["pred_rating"] = predicted_rating
            user_r.add_prediction(test)
            
            #if i > 100:
            #    break
        
            #print(user_id, movie_id, rating, round(predicted_rating, 1))
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            #sleep(1)
            break
        
    mae = Evaluation.mean_absolute_error(user_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    #print(test)
    #print(mae)
    
    return test, mae     


def run_item_rec_experiment(k):
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
            
            #if i > 107:
            #    break
        
            #print(user_id, movie_id, rating, round(predicted_rating, 1))
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            #sleep(1)
            break
        
    mae = Evaluation.mean_absolute_error(item_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    #print(test)
    #print(mae)
    
    return test, mae        

        
def run_bootstrap_rec_experiment(k):
    bs_r = BootstrapRecommender(k, iterations = 3 , additions = 10) # was 1,6
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
            
            #if i > 10:
            #    break
        
            #print(user_id, movie_id, rating, round(predicted_rating, 1))
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            #sleep(1)
            break
        
    mae = Evaluation.mean_absolute_error(bs_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    #print(test)
    #print(mae)
    
    return test, mae     


def run_pearlpu_rec_experiment(k):
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
            
            #if i > 17:
            #    break
        
            #print(user_id, movie_id, rating, round(predicted_rating, 1))
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            #sleep(1)
            break
        
    mae = Evaluation.mean_absolute_error(pp_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    #print(test)
    #print(mae)
    
    return test, mae     
    
def run_corec_rec_experiment(k):
    cr_r = CoRecRecommender(k)

    print("\nRunning Co Rec Recommender\n")

    for i, test in enumerate(cr_r.test_ratings):
        try:
            user_id = test['user_id']
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = cr_r.predict_rating_item_based_nn_wtd(user_id, movie_id)
            
            test["pred_rating"] = predicted_rating
            cr_r.add_prediction(test)
            
            #if i > 11:
            #    break
        
            #print(user_id, movie_id, rating, round(predicted_rating, 1))
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            #sleep(1)
            break
        
    mae = Evaluation.mean_absolute_error(cr_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    #print(test)
    #print(mae)
    
    return test, mae     
