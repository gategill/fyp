"""

"""


from icecream import ic
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
# cold start < 20 ratings

s3 = boto3.client('s3')

def save_in_s3_function(data, which_model, current_timestamp):
    s3.put_object(Body = data, Bucket = "fyp-w9797878", Key = str(current_timestamp) + "/"+ which_model + '.txt')


def read_in_yaml_file(config_path):
    with open(config_path) as f:
        kwargs = yaml.load(f, Loader = yaml.FullLoader)

    if "experiment_config" not in kwargs:
        raise KeyError("missing experiment_config in kwargs")
    if "seed" not in kwargs["experiment_config"]:
        raise KeyError("missing seed in experiment_config")
    if "save_in_s3" not in kwargs["experiment_config"]:
        raise KeyError("missing save_in_s3 in experiment_config")
    if "save_results" not in kwargs["experiment_config"]:
        raise KeyError("missing save_results in experiment_config")
    if "kfolds" not in kwargs["experiment_config"]:
        raise KeyError("missing kfolds in experiment_config")
    if "early_stop" not in kwargs["experiment_config"]:
        raise KeyError("missing early_stop in experiment_config")
    if "disable_ic" not in kwargs["experiment_config"]:
        raise KeyError("missing disable_ic in experiment_config")
    
    if (type(kwargs["experiment_config"]["seed"]) != int):
        raise TypeError("seed should be an integer")
    if type(kwargs["experiment_config"]["save_in_s3"]) != bool:
        raise TypeError("save_in_s3 should be boolean")
    if type(kwargs["experiment_config"]["save_results"]) != bool:
        raise TypeError("save_results should be boolean")
    if type(kwargs["experiment_config"]["kfolds"]) != int:
        raise TypeError("kfolds should be an integer")
    if type(kwargs["experiment_config"]["early_stop"]) != bool:
        raise TypeError("early_stop should be boolean")
    if type(kwargs["experiment_config"]["disable_ic"]) != bool:
        raise TypeError("disable_ic should be boolean")
    
    
    if "dataset_config" not in kwargs:
        raise KeyError("missing dataset_config in kwargs")
    if "dataset_path" not in kwargs["dataset_config"]:
        raise KeyError("missing dataset_path in dataset_config")
    if "prefiltering" not in kwargs["dataset_config"]:
        raise KeyError("missing prefiltering in dataset_config")
    if "test_splitting_ratio" not in kwargs["dataset_config"]:
        raise KeyError("missing test_splitting_ratio in dataset_config")
    
    if type(kwargs["dataset_config"]["dataset_path"]) != str:
        raise TypeError("dataset_path should be a string")
    if type(kwargs["dataset_config"]["prefiltering"]) != dict:
        raise TypeError("prefiltering should be a dictionary")
    if type(kwargs["dataset_config"]["test_splitting_ratio"]) != float:
        raise TypeError("test_splitting_ratio should be a float")
    
    
    if "models" not in kwargs:
        raise KeyError("missing models in kwargs")
    if len(kwargs["models"]) == 0:
        raise ValueError("no models provided in kwargs[models]")
    
    if "UserKNN" in kwargs["models"]:
        if "neighbours" not in kwargs["models"]["UserKNN"]:
            raise KeyError("missing neighbours in UserKNN")
        if "similarity" not in kwargs["models"]["UserKNN"]:
            raise KeyError("missing similarity in UserKNN")
        
        if type(kwargs["models"]["UserKNN"]["neighbours"]) != int:
            raise TypeError("UserKNN.neighbours should be an integer")
        if type(kwargs["models"]["UserKNN"]["similarity"]) != str:
            raise TypeError("UserKNN.similarity should be an string")
        
        if kwargs["models"]["UserKNN"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid UserKNN.similarity measure\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
        
        
    if "ItemKNN" in kwargs["models"]:
        if "neighbours" not in kwargs["models"]["ItemKNN"]:
            raise KeyError("missing neighbours in ItemKNN")
        if "similarity" not in kwargs["models"]["ItemKNN"]:
            raise KeyError("missing similarity in ItemKNN")

        if type(kwargs["models"]["ItemKNN"]["neighbours"]) != int:
            raise TypeError("ItemKNN.neighbours should be an integer")
        if type(kwargs["models"]["ItemKNN"]["similarity"]) != str:
            raise TypeError("ItemKNN.similarity should be an string")
        
        if kwargs["models"]["ItemKNN"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid ItemKNN.similarity\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
        
        
    if "Bootstrap" in kwargs["models"]:
        if "neighbours" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing neighbours in Bootstrap")
        if "similarity" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing similarity in Bootstrap")
        if "iterations" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing iterations in Bootstrap")
        if "additions" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing additions in Bootstrap")
        
        if type(kwargs["models"]["Bootstrap"]["neighbours"]) != int:
            raise TypeError("Bootstrap.neighbours should be an integer")
        if type(kwargs["models"]["Bootstrap"]["similarity"]) != str:
            raise TypeError("Bootstrap.similarity should be an string")
        if type(kwargs["models"]["Bootstrap"]["iterations"]) != int:
            raise TypeError("Bootstrap.iterations should be an integer")
        if type(kwargs["models"]["Bootstrap"]["additions"]) != int:
            raise TypeError("Bootstrap.additions should be an integer")
        
        if kwargs["models"]["Bootstrap"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid Bootstrap.similarity\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
        
        
    if "PearlPu" in kwargs["models"]:
        if "neighbours" not in kwargs["models"]["PearlPu"]:
            raise KeyError("missing neighbours in PearlPu")
        if "similarity" not in kwargs["models"]["PearlPu"]:
            raise KeyError("missing similarity in PearlPu")
        if "weight_threshold" not in kwargs["models"]["PearlPu"]:
            raise KeyError("missing weight_threshold in PearlPu")
        if "recursion_threshold" not in kwargs["models"]["PearlPu"]:
            raise KeyError("missing recursion_threshold in PearlPu")
        if "phi" not in kwargs["models"]["PearlPu"]:
            raise KeyError("missing phi in PearlPu")
        if "k_prime" not in kwargs["models"]["PearlPu"]:
            raise KeyError("missing k_prime in PearlPu")
        if "baseline" not in kwargs["models"]["PearlPu"]:
            raise KeyError("missing baseline in PearlPu")
        
        if type(kwargs["models"]["PearlPu"]["neighbours"]) != int:
            raise TypeError("PearlPu.neighbours should be an integer")
        if type(kwargs["models"]["PearlPu"]["similarity"]) != str:
            raise TypeError("PearlPu.similarity should be an string")
        if type(kwargs["models"]["PearlPu"]["weight_threshold"]) != float:
            raise TypeError("PearlPu.weight_threshold should be a float")
        if type(kwargs["models"]["PearlPu"]["recursion_threshold"]) != int:
            raise TypeError("PearlPu.recursion_threshold should be an integer")
        if type(kwargs["models"]["PearlPu"]["phi"]) != int:
            raise TypeError("PearlPu.phi should be an integer")
        if type(kwargs["models"]["PearlPu"]["k_prime"]) != int:
            raise TypeError("PearlPu.k_prime should be an integer")
        if type(kwargs["models"]["PearlPu"]["baseline"]) != str:
            raise TypeError("PearlPu.baseline should be a string")
        
        if kwargs["models"]["PearlPu"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid PearlPu.similarity\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
        if kwargs["models"]["PearlPu"]["baseline"] not in ["bs", "bs+", "ss", "cs", "cs+"]:
            raise ValueError("invalid PearlPu.baseline\navailable baselines are: [bs, bs+, ss, cs, cs+]")
        
        
    if "CoRec" in kwargs["models"]:
        if "neighbours" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing neighbours in CoRec")
        if "similarity" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing similarity in CoRec")
        if "additions" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing additions in CoRec")
        if "top_m" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing top_m in CoRec")

        if type(kwargs["models"]["CoRec"]["neighbours"]) != int:
            raise TypeError("CoRec.neighbours should be an integer")
        if type(kwargs["models"]["CoRec"]["similarity"]) != str:
            raise TypeError("CoRec.similarity should be an string")
        if type(kwargs["models"]["CoRec"]["additions"]) != int:
            raise TypeError("CoRec.additions should be an integer")
        if type(kwargs["models"]["CoRec"]["top_m"]) != int:
            raise TypeError("CoRec.top_m should be an integer")
        if kwargs["models"]["CoRec"]["top_m"] < kwargs["models"]["CoRec"]["additions"]:
            raise ValueError("CoRec.top_m should be larger than additions (I think)")

        if kwargs["models"]["CoRec"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid CoRec.similarity\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
            
    return kwargs


def run_experiment(config_path) -> None:
    kwargs = read_in_yaml_file(config_path)
    kwargs["config_path"] = config_path
    
    save_in_s3 = kwargs["experiment_config"]["save_in_s3"]
    save_results = kwargs["experiment_config"]["save_results"]
    kfolds = kwargs["experiment_config"]["kfolds"]
    a_seed = kwargs["experiment_config"]["seed"]
    
    all_models =  "_".join(list(kwargs["models"].keys()))
    
    if a_seed != -1:
        random.seed(a_seed)
    
    if kwargs["experiment_config"]["disable_ic"]: 
        ic.disable()
    
    for i in range(kfolds):
        current_timestamp = int(time.time())
        save_path = "./results/{}".format(current_timestamp)
        os.mkdir(save_path)
        
        models = kwargs["models"]
        
        # For each rating in the test set, make a prediction using a 
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
            try:
                tic = time.time()
                u, mae = run_corec_rec_experiment(**kwargs)
                toc = time.time()
                time_elapsed = round(toc - tic, 3)
                
                print(u, mae, time_elapsed)
                experiment_result = "CoRec, {}, {} \n".format(mae, time_elapsed)
                lines_result += experiment_result
                
                if save_in_s3:
                    save_in_s3_function(experiment_result, "CoRec", current_timestamp)
                        
            except Exception as e:
                line_error = "error performing experiment, CoRec, error = {}".format(e)
                print(line_error)
                
                if save_in_s3:
                    save_in_s3_function(line_error, "CoRec", current_timestamp)
            
            
        if save_results:
            saved_file_results = "{}/{}.txt".format(save_path, all_models)
            
            with open(saved_file_results, "w") as f:
                f.write(lines_result)
                
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
            
            if kwargs["experiment_config"]["early_stop"]:
                if i > 50:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae = Evaluation.mean_absolute_error(user_r.predictions)
    mae = round(mae, 3)
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
            
            if kwargs["experiment_config"]["early_stop"]:
                if i > 50:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae = Evaluation.mean_absolute_error(item_r.predictions)
    mae = round(mae, 3)
    test["pred_rating"] = round(test["pred_rating"], 2)
    
    return test, mae        

        
def run_bootstrap_rec_experiment(**kwargs):
    kwargs["run_params"] = kwargs["models"]["Bootstrap"]
    bs_r = BootstrapRecommender(**kwargs) # was 1, 6
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
            
            if kwargs["experiment_config"]["early_stop"]:
                if i > 10:
                    break
            
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae = Evaluation.mean_absolute_error(bs_r.predictions)
    mae = round(mae, 3)
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
            
            if kwargs["experiment_config"]["early_stop"]:
                if i > 10:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae = Evaluation.mean_absolute_error(pp_r.predictions)
    mae = round(mae, 3)
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

            if kwargs["experiment_config"]["early_stop"]:
                if i > 30:
                    break
                    
        except KeyboardInterrupt:
            ic("\nStopping\n")
            ic(i)
            break
        
    mae_user, mae_item = Evaluation.mean_absolute_error(co_rec_r.predictions)
    
    return test, [mae_user, mae_item]