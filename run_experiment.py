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
import yaml
import random
import shutil

#ic.disable()
s3 = boto3.client('s3')
#s3_resource = boto3.resource('s3')


def save_in_s3_function(data, which_model, current_timestamp):
    s3.put_object(Body = data, Bucket = "fyp-w9797878", Key = str(current_timestamp) + "/"+ which_model + '.txt')


def read_in_yaml_file(config_path):
    with open(config_path) as f:
        config_data = yaml.load(f, Loader = yaml.FullLoader)
    return config_data


def run_experiment_yaml(config_path: str):
    kwargs = read_in_yaml_file(config_path)
    kwargs["config_path"] = config_path
    
    #print(config_data)
    
    run_experiment(**kwargs)


def run_experiment(**kwargs) -> None:
    random.seed(kwargs["exp_config"]["seed"])
    save_in_s3 = kwargs["exp_config"]["save_in_s3"]
    save_results = kwargs["exp_config"]["save_results"]
    all_models =  "_".join(list(kwargs["models"].keys()))
    
    for i in range(kwargs["exp_config"]["kfolds"]):
        
        current_timestamp = int(time.time())
        save_path = "./results/{}".format(current_timestamp)
        os.mkdir(save_path)
        
        models = kwargs["models"]
        
        # For each rating in the test set, make a prediction using a 
        # user-based KNN with k = 3
        lines_result = "algorithm, mae, time_elapsed\n"
        if "UserKNN" in models:

            try:
                tic = time.time()
                u, mae = run_user_rec_experiment(**kwargs)
                toc = time.time()
                time_elapsed = round(toc - tic, 3)
                
                print(u, mae, time_elapsed)
                experiment_result = "UserKNN, {}, {} \n".format(mae, time_elapsed)
                lines_result += experiment_result
                
                if save_in_s3:
                    save_in_s3_function(experiment_result, "UserKNN", current_timestamp)
                    
            except Exception as e:
                line_error = "error performing experiment, UserKNN, error = {}".format(e)
                print(line_error)
                
                if save_in_s3:
                    save_in_s3_function(line_error, "UserKNN", current_timestamp)

        # For each rating in the test set, make a prediction using an 
        # item-based KNN with k = 3
        if "ItemKNN" in models:
            try:
                tic = time.time()
                u, mae = run_item_rec_experiment(**kwargs)
                toc = time.time()
                time_elapsed = round(toc - tic, 3)
                
                print(u, mae, time_elapsed)
                experiment_result = "ItemKNN, {}, {} \n".format(mae, time_elapsed)
                lines_result += experiment_result
                
                if save_in_s3:
                    save_in_s3_function(experiment_result, "ItemKNN", current_timestamp)

            except Exception as e:
                line_error = "error performing experiment, ItemKNN, error = {}".format(e)
                print(line_error)
                
                if save_in_s3:
                    save_in_s3_function(line_error, "ItemKNN", current_timestamp)

        if "Bootstrap" in models:
            try:
                tic = time.time()
                u, mae = run_bootstrap_rec_experiment(**kwargs)
                toc = time.time()
                time_elapsed = round(toc - tic, 3)
                
                print(u, mae, time_elapsed)
                experiment_result = "Bootstrap, {}, {} \n".format(mae, time_elapsed)
                lines_result += experiment_result
                
                if save_in_s3:
                    save_in_s3_function(experiment_result, "Bootstrap", current_timestamp)
                        
            except Exception as e:
                line_error = "error performing experiment, Bootstrap, error = {}".format(e)
                print(line_error)
                
                if save_in_s3:
                    save_in_s3_function(line_error, "Bootstrap", current_timestamp)



        if "PearlPu" in models:
            try:
                tic = time.time()
                u, mae = run_pearlpu_rec_experiment(**kwargs)
                toc = time.time()
                time_elapsed = round(toc - tic, 3)
                
                print(u, mae, time_elapsed)
                experiment_result = "PearlPu, {}, {} \n".format(mae, time_elapsed)
                lines_result += experiment_result
                
                if save_in_s3:
                    save_in_s3_function(experiment_result, "PearlPu", current_timestamp)
                
            except Exception as e:
                line_error = "error performing experiment, PearlPu, error = {}".format(e)
                print(line_error)
                
                if save_in_s3:
                    save_in_s3_function(line_error, "PearlPu", current_timestamp)


        if "CoRec" in models:
            #try:
            tic = time.time()
            u, mae = run_corec_rec_experiment(**kwargs)
            toc = time.time()
            time_elapsed = round(toc - tic, 3)
            
            print(u, mae, time_elapsed)
            experiment_result = "CoRec, {}, {} \n".format(mae, time_elapsed)
            lines_result += experiment_result
            
            if save_in_s3:
                save_in_s3_function(experiment_result, "CoRec", current_timestamp)
                    
            #except Exception as e:
            #    line_error = "error performing experiment, CoRec, error = {}".format(e)
            #    print(line_error)
                
            #    if save_in_s3:
            #        save_in_s3_function(line_error, "CoRec", current_timestamp)
            
            
        if save_results:
            saved_file_results = "{}/{}.txt".format(save_path, all_models)
            
            with open(saved_file_results, "w") as f:
                f.write(lines_result)
                
            #saved_file_config = "{}/{}.yml".format(save_path, kwargs["config_path"])
            src = kwargs["config_path"]
            dst = save_path + "/config.yml"
            shutil.copyfile(src, dst)           
                 
        if save_in_s3:
            save_in_s3_function(lines_result, all_models, current_timestamp)
            with open(kwargs["config_path"], "rb") as f:
                s3.upload_fileobj(f, Bucket = "fyp-w9797878",  Key = str(current_timestamp) + "/config_file.yml")


def run_user_rec_experiment(**kwargs):
    kwargs["run_params"] = kwargs["models"]["UserKNN"]
    user_r = UserRecommender(**kwargs)
        
    print("\nRunning User Recommender\n")
    print(len(user_r.test_ratings))

    for i, test in enumerate(user_r.test_ratings):
        try:
            user_id = int(test['user_id'])
            item_id = int(test['item_id'])
            
            predicted_rating = user_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
              
            if predicted_rating < 1.0:
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                predicted_rating = 5.0
                
            test["pred_rating"] = predicted_rating
            user_r.add_prediction(test)
            
            if kwargs["exp_config"]["early_stop"]:
                if i > 100:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae = Evaluation.mean_absolute_error(user_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    return test, mae     


def run_item_rec_experiment(**kwargs):
    kwargs["run_params"] = kwargs["models"]["ItemKNN"]
    item_r = ItemRecommender(**kwargs)
        
    print("\nRunning Item Recommender\n")

    for i, test in enumerate(item_r.test_ratings):
        try:
            user_id = int(test['user_id'])
            item_id = int(test['item_id'])
            
            predicted_rating = item_r.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
            
            if predicted_rating < 1.0:
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                predicted_rating = 5.0
                
            test["pred_rating"] = predicted_rating
            item_r.add_prediction(test)
            
            if kwargs["exp_config"]["early_stop"]:
                if i > 100:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            #sleep(1)
            break
        
    mae = Evaluation.mean_absolute_error(item_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    return test, mae        

        
def run_bootstrap_rec_experiment(**kwargs):
    kwargs["run_params"] = kwargs["models"]["Bootstrap"]
    bs_r = BootstrapRecommender(**kwargs) # was 1,6
    print("\nEnriching Bootstrap Recommender\n")

    bs_r.enrich()

    print("\nRunning Bootstrap Recommender\n")
    
    for i, test in enumerate(bs_r.test_ratings):
        try:
            user_id = int(test['user_id'])
            item_id = int(test['item_id'])

            predicted_rating = bs_r.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
            
            if predicted_rating < 1.0:
                predicted_rating = 1.0
                
            if predicted_rating > 5:
                predicted_rating = 5.0
                
            test["pred_rating"] = predicted_rating
            bs_r.add_prediction(test)
            
            if kwargs["exp_config"]["early_stop"]:
                if i > 30:
                    break
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae = Evaluation.mean_absolute_error(bs_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)

    return test, mae     


def run_pearlpu_rec_experiment(**kwargs):
    kwargs["run_params"] = kwargs["models"]["PearlPu"]

    pp_r = PearlPuRecommender(**kwargs)

    print("\nRunning Pearl Pu Recommender\n")

    for i, test in enumerate(pp_r.test_ratings):
        try:
            user_id = int(test['user_id'])
            item_id = int(test['item_id'])
            
            predicted_rating = pp_r.recursive_prediction(user_id, item_id)
            
            if predicted_rating < 1.0:
                print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 1.0
                
            if predicted_rating > 5.0:
                print("The rating is beyond the range: {}".format(predicted_rating))
                predicted_rating = 5.0
                
            test["pred_rating"] = predicted_rating
            pp_r.add_prediction(test)
            
            if kwargs["exp_config"]["early_stop"]:
                if i > 30:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae = Evaluation.mean_absolute_error(pp_r.predictions)
    mae = round(mae, 5)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    return test, mae     


def run_corec_rec_experiment(**kwargs):
    kwargs["run_params"] = kwargs["models"]["CoRec"]
    co_rec_r = CoRecRecommender(**kwargs)

    print("\nRunning CoRec Recommender\n")
    
    # train
    co_rec_r.train_co_rec()

    # predict and test
    for i, test in enumerate(co_rec_r.test_ratings):
        try:
            user_id = int(test['user_id'])
            item_id = int(test['item_id'])
            
            user_predicted_rating = co_rec_r.predict_co_rec_for_users(user_id, item_id)
            item_predicted_rating = co_rec_r.predict_co_rec_for_items(user_id, item_id)
            
            test["user_pred_rating"] = user_predicted_rating
            test["item_pred_rating"] = item_predicted_rating
            co_rec_r.add_prediction(test)

            if kwargs["exp_config"]["early_stop"]:
                if i > 100:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae_user, mae_item = Evaluation.mean_absolute_error(co_rec_r.predictions)
    
    return test, [mae_user, mae_item]