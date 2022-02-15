"""

"""


from sqlite3 import Timestamp
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
import boto3


#session = boto3.Session()
#s3 = session.resource('s3')
s3 = boto3.client('s3')




ic.disable()
#ic.configureOutput(includeContext=True)
s3_resource = boto3.resource('s3')




def save_in_s3_function(da, which, current_timestamp):
    #s3_resource.Bucket("fyp-w9797878").upload_file(Filename=filename, Key=key_c)
    #txt_data = da
    #object = s3.Object(BucketName ="fyp-w9797878", File_Key = str(current_timestamp) + "/"+ which + '.txt')
    #result = object.put(Body=da)
    s3.put_object(Body = da, Bucket = "fyp-w9797878", Key = str(current_timestamp) + "/"+ which + '.txt')

    
    
def run_experiment(k: int, which: str, save_results: bool, kfolds: int, save_in_s3: bool) -> None:
    
    save_in_s3 = False
    for i in range(kfolds):
        
        current_timestamp = int(time.time())
        save_path = "./results/{}".format(current_timestamp)
        os.mkdir(save_path)
        
        # For each rating in the test set, make a prediction using a 
        # user-based KNN with k = 3
        lines_result = "algorithm, mae\n"
        if "u" in which:
            u, mae = run_user_rec_experiment(k)
            print(u, mae)
            lines_result += "u_rec_k={}, {}\n".format(k, mae)
            
            #print(save_in_s3)
            if save_in_s3:
            #saved_file = "{}/{}.txt".format(save_path, which)
                save_in_s3_function("u_rec_k={}, {}\n".format(k, mae), "u", current_timestamp)


        # For each rating in the test set, make a prediction using an 
        # item-based KNN with k = 3
        if "i" in which:
            u, mae = run_item_rec_experiment(k)
            print(u, mae)
            lines_result += "i_rec_k={}, {}\n".format(k, mae)
            
            if save_in_s3:
            #saved_file = "{}/{}.txt".format(save_path, which)
                save_in_s3_function("i_rec_k={}, {}\n".format(k, mae), "i", current_timestamp)


        if "b" in which:
            u, mae = run_bootstrap_rec_experiment(k)
            print(u, mae)
            lines_result += "bs_rec_k={}, {}\n".format(k, mae)
            
            if save_in_s3:
            #saved_file = "{}/{}.txt".format(save_path, which)
                save_in_s3_function("bs_rec_k={}, {}\n".format(k, mae), "b", current_timestamp)


        if "p" in which:
            u, mae = run_pearlpu_rec_experiment(k)
            print(u, mae)
            lines_result += "pp_rec_k={}, {}\n".format(k, mae)
            
            if save_in_s3:
            #saved_file = "{}/{}.txt".format(save_path, which)
                save_in_s3_function("pp_rec_k={}, {}\n".format(k, mae), "p", current_timestamp)
                
            
        if "c" in which:
            u, mae = run_corec_rec_experiment(k)
            print(u, mae)
            lines_result += "corec_rec_k={}, {}\n".format(k, mae)
            
            if save_in_s3:
            #saved_file = "{}/{}.txt".format(save_path, which)
                save_in_s3_function("corec_rec_k={}, {}\n".format(k, mae), "c", current_timestamp)
            
        if save_results:
            saved_file = "{}/{}.txt".format(save_path, which)
            
            with open(saved_file, "w") as f:
                f.write(lines_result)
                
        if save_in_s3:
            #saved_file = "{}/{}.txt".format(save_path, which)
            save_in_s3_function(lines_result, which, current_timestamp)



def run_user_rec_experiment(k):
    user_r = UserRecommender(k)
    
    print("\nRunning User Recommender\n")
    print(len(user_r.test_ratings))

    for i, test in enumerate(user_r.test_ratings):
        try:
            user_id = test['user_id']
            item_id = test['item_id']
            rating = test['rating']
            
            predicted_rating = user_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
            
            
              
            if predicted_rating < 1.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
                
                
            test["pred_rating"] = predicted_rating
            user_r.add_prediction(test)
            
            if i > 100:
                break
        
            #print(user_id, item_id, rating, round(predicted_rating, 1))
            
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
            item_id = test['item_id']
            rating = test['rating']
            
            predicted_rating = item_r.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
            
              
            if predicted_rating < 1.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
                
            test["pred_rating"] = predicted_rating
            item_r.add_prediction(test)
            
            if i > 100:
                break
        
            #print(user_id, item_id, rating, round(predicted_rating, 1))
            
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
            item_id = test['item_id']
            rating = test['rating']
            #print(bs_r.dataset.num_ratings)

            predicted_rating = bs_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
            
            
            if predicted_rating < 1.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
                
            test["pred_rating"] = predicted_rating
            bs_r.add_prediction(test)
            
            if i > 30:
                break
        
            #print(user_id, item_id, rating, round(predicted_rating, 1))
            
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
            item_id = test['item_id']
            rating = test['rating']
            
            predicted_rating = pp_r.recursive_prediction(user_id, item_id)
            #assert predicted_rating <= 5
            #assert predicted_rating >= 0
            
            if predicted_rating < 1.0:
                print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5.0:
                print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
            test["pred_rating"] = predicted_rating
            pp_r.add_prediction(test)
            
            if i > 30:
                break
        
            #print(user_id, item_id, rating, round(predicted_rating, 1))
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            #sleep(1)
            break
        
        
        except AssertionError:
            print("The rating is beyond the range: {}".format(predicted_rating))
            continue
        
    mae = Evaluation.mean_absolute_error(pp_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    #print(test)
    #print(mae)
    
    return test, mae     


def run_corec_rec_experiment(k):
    co_rec_r = CoRecRecommender(k)

    print("\nRunning Co Rec Recommender\n")
    
    # train
    co_rec_r.train_co_rec()

    # predict and test
    for i, test in enumerate(co_rec_r.test_ratings):
        try:
            user_id = test['user_id']
            item_id = test['item_id']
            rating = test['rating']
            
            user_predicted_rating = co_rec_r.predict_co_rec_user(user_id, item_id)
            item_predicted_rating = co_rec_r.predict_co_rec_item(user_id, item_id)
            
            test["user_pred_rating"] = user_predicted_rating
            test["item_pred_rating"] = item_predicted_rating
            co_rec_r.add_prediction(test)

            if i > 10:
                break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
        
    mae_user, mae_item = Evaluation.mean_absolute_error(co_rec_r.predictions)
    
    return test, [mae_user, mae_item]