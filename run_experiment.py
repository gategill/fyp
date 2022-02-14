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

    
    
def run_experiment(k: int, which: str, save_results: bool = True, save_in_s3 = True, kfolds = 5) -> None:
    
    
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
            movie_id = test['movie_id']
            rating = test['rating']
            
            predicted_rating = user_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
            
            
              
            if predicted_rating < 1.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
                
                
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
            
              
            if predicted_rating < 1.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
                
            test["pred_rating"] = predicted_rating
            item_r.add_prediction(test)
            
            #if i > 100:
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
            #print(bs_r.dataset.num_ratings)

            predicted_rating = bs_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_movie_id = movie_id)
            
            
            if predicted_rating < 1.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
                
            test["pred_rating"] = predicted_rating
            bs_r.add_prediction(test)
            
            #if i > 50:
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
            #assert predicted_rating <= 5
            #assert predicted_rating >= 0
            
            if predicted_rating < 1.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5.0:
                #print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
            test["pred_rating"] = predicted_rating
            pp_r.add_prediction(test)
            
            #if i > 30:
            #    break
        
            #print(user_id, movie_id, rating, round(predicted_rating, 1))
            
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


"""def run_corec_rec_experiment(k):
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
from sklearn.model_selection import train_test_split
from dataset.Dataset import Dataset
from sklearn.base import ClusterMixin

class myUserRec(ClusterMixin):
    def __init__(self, *args, **args):
        self.model = UserRecommender(*args, **args)
        
    def fit_predict(self, X):
        self.model.predict_rating_user_based_nn_wtd
        
        
def something():
    ds = Dataset()
    print(ds.get_ratings_as_df())
    
    
    #X_train, X_test, y_train, y_test = train_test_split(ds.user_training_ratings, ds.user_test_ratings, test_size=0.2, random_state=2)

    # add entire to dataset
    # how to resolve the item and user view of the data in sklearn???
    
#something()"""