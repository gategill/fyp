"""

"""


from recommender.MatrixFactorisationRecommender import MatrixFactorisationRecommender
from recommender.UserRecursiveKNNRecommender import UserRecursiveKNNRecommender
from recommender.ItemRecursiveKNNRecommender import ItemRecursiveKNNRecommender
from recommender.UserBootstrapRecommender import UserBootstrapRecommender
from recommender.ConfidentUserBootstrapRecommender import ConfidentUserBootstrapRecommender
from recommender.ItemBootstrapRecommender import ItemBootstrapRecommender
from recommender.ConfidentItemBootstrapRecommender import ConfidentItemBootstrapRecommender
from recommender.ItemKNNRecommender import ItemKNNRecommender
from recommender.UserKNNRecommender import UserKNNRecommender
from recommender.MostPopRecommender import MostPopRecommender
from recommender.RandomRecommender import RandomRecommender
from recommender.CoRecRecommender import CoRecRecommender
from recommender.MeanRecommender import MeanRecommender
from dataset.Dataset import Dataset

from scipy.stats import ttest_rel
from icecream import ic
import numpy as np
import YAMLHandler
import traceback
import time
import boto3
import random
import shutil
import os

s3 = boto3.client('s3')

def run_experiment(config_path) -> None:
    recommenders = {"UserKNN" : UserKNNRecommender, 
                       "ItemKNN" : ItemKNNRecommender,
                       "UserBootstrap" : UserBootstrapRecommender,
                       "ConfidentUserBootstrap" : ConfidentUserBootstrapRecommender,
                       "ItemBootstrap" : ItemBootstrapRecommender,
                       "ConfidentItemBootstrap" : ConfidentItemBootstrapRecommender,
                       "UserRecursiveKNN" : UserRecursiveKNNRecommender, 
                       "ItemRecursiveKNN" : ItemRecursiveKNNRecommender, 
                       "CoRec" : CoRecRecommender,
                       "MatrixFactorisation" : MatrixFactorisationRecommender,
                       "MostPop" : MostPopRecommender,
                       "Random" : RandomRecommender,
                       "Mean" : MeanRecommender,
                       }
            
        
    kwargs = YAMLHandler.read_in_yaml_file(config_path)
    # pass some agruments down
    kwargs["config_path"] = config_path
    kwargs["dataset_config"]["kfolds"] = kwargs["experiment_config"]["kfolds"]
    

    save_in_s3 = kwargs["experiment_config"]["save_in_s3"]
    kfolds = kwargs["experiment_config"]["kfolds"]
    if "seed" in kwargs["experiment_config"]:
        a_seed = kwargs["experiment_config"]["seed"]
        random.seed(a_seed)
        
    current_timestamp = int(time.time())
    save_path = "./results/{}".format(current_timestamp)
    os.mkdir(save_path)
    os.mkdir(save_path + "/all")
    os.mkdir(save_path + "/model")
    os.mkdir(save_path + "/model_k")
    os.mkdir(save_path + "/single")

    
    src = kwargs["config_path"]
    dst = save_path + "/config.yml"
    shutil.copyfile(src, dst) 
    
    if save_in_s3:
        with open(kwargs["config_path"], "rb") as f:
            s3.upload_fileobj(f, Bucket = "fyp-w9797878",  Key = str(current_timestamp) + "/config_file.yml")

    all_models =  "_".join(list(kwargs["models"].keys()))

    if kwargs["experiment_config"]["disable_ic"]: 
        ic.disable()
        
    dataset = Dataset(**kwargs["dataset_config"])
    
    results_header = "algorithm, k, mae, time_elapsed_s, fold_num\n"
    results_header += (len(results_header) * "-") + "\n"
    all_results = results_header
    
    for model in kwargs["models"]:
        #print(kwargs)
        model_results = results_header            
        print("MODEL = {}".format(model))
        
        #for K in kwargs["experiment_config"]["neighbours"]:    
        for K in kwargs["models"][model]["neighbours"]:
  
            print("NEIGHBOURS = {}".format(K))
            model_k_mae = []
            #model_k_rmse = []
            model_k_results = results_header

            for fold_num in range(kfolds):
                single_results = results_header
                
                print("FOLD NUMBER = {}/{}\n".format(fold_num + 1, kfolds))
            
                dataset.load_ratings(fold_num)
        
                try:                
                    print("Running {} Recommender".format(model))
                    kwargs["models"][model]["neighbours"] = K
                    kwargs["models"][model]["similarity"] = kwargs["models"][model]["similarity"]
                    kwargs["run_params"] = kwargs["models"][model]
                    
                    tic = time.time()
                    a_recommender = recommenders[model](dataset, **kwargs)
                    a_recommender.train()
                    test = a_recommender.get_predictions()
                    toc = time.time()
                    time_elapsed = round(toc - tic, 3)
                    
                    mae = a_recommender.evaluate_predictions("MAE")
                    #rmse = a_recommender.evaluate_predictions("RMSE")
                        
                    del a_recommender

                    print(test, mae)
                    
                    model_k_mae.append(mae)
                    #model_k_rmse.append(rmse)
                    
                    experiment_result = "{}, {}, {}, {}, {}\n".format(model, K, mae, time_elapsed, fold_num + 1)
                    all_results += experiment_result
                    model_results += experiment_result
                    model_k_results += experiment_result
                    single_results += experiment_result
                    
                    # saving singles
                    with open("{}/single/single-{}-{}-{}.txt".format(save_path, model, K, fold_num + 1), "w") as f:
                        f.write(single_results)
                    
                    if save_in_s3:
                        s3_name = "{}/single/single-{}-{}-{}.txt".format(current_timestamp,model, K, fold_num)
                        s3.put_object(Body = single_results, Bucket = "fyp-w9797878", Key = s3_name)
                
                except Exception as e:
                    line_error = "error performing experiment, {}, error = {}".format(model, e)
                    print(traceback.print_exc())
                    
                    # saving errors
                    with open("{}/single/single-{}-{}-{}.txt".format(save_path, model, K, fold_num), "w") as f:
                        f.write(line_error)
                    
                    if save_in_s3:
                        s3_name = "{}/single/single-{}-{}-{}.txt".format(current_timestamp,model, K, fold_num)
                        s3.put_object(Body = line_error, Bucket = "fyp-w9797878", Key = s3_name)
                
                
            
            if model == "CoRec":
                user_mae = [i [0] for i in model_k_mae]
                item_mae = [i [1] for i in model_k_mae]
                
                model_k_user_mean_mae = round(np.mean(user_mae), 5) 
                model_k_item_mean_mae = round(np.mean(item_mae), 5) 
                
                model_k_mean_mae = [model_k_user_mean_mae, model_k_item_mean_mae]
                
                #user_rmse = [i [0] for i in model_k_rmse]
                #item_rmse = [i [1] for i in model_k_rmse]
                
                #model_k_user_mean_rmse = round(np.mean(user_rmse), 5) 
                #model_k_item_mean_rmse = round(np.mean(item_rmse), 5) 
                
                model_k_mean_mae = [model_k_user_mean_mae, model_k_item_mean_mae]
                #model_k_mean_rmse = [model_k_user_mean_rmse, model_k_item_mean_rmse]
                
            else:
                model_k_mean_mae = round(np.mean(model_k_mae), 5) 
                #model_k_mean_rmse = round(np.mean(model_k_rmse), 5) 
            
            model_k_results += "{}_{}: Averaged_MAE = {}\n".format(model, K, model_k_mean_mae)
            #model_k_results += "{}_{}: Averaged_RMSE = {}\n".format(model, K, model_k_mean_rmse)
            
            model_results += "{}_{}: Averaged_MAE = {}\n".format(model, K, model_k_mean_mae)
            #model_results += "{}_{}: Averaged_RMSE = {}\n".format(model, K, model_k_mean_rmse)
            
            all_results += "{}_{}: Averaged_MAE = {}\n".format(model, K, model_k_mean_mae)            
            #all_results += "{}_{}: Averaged_RMSE = {}\n".format(model, K, model_k_mean_rmse)            
            
            # saving model_k
            with open("{}/model_k/model_k-{}-{}.txt".format(save_path, model, K), "w") as f:
                f.write(model_k_results)
        
            if save_in_s3:
                s3_name = "{}/model_k/model_k-{}-{}.txt".format(current_timestamp, model, K)
                s3.put_object(Body = model_k_results, Bucket = "fyp-w9797878", Key = s3_name)
             
        
        # saving model   
        with open("{}/model/model-{}.txt".format(save_path, model), "w") as f:
            f.write(model_results)
        
        if save_in_s3:
            s3_name = "{}/model/model-{}.txt".format(current_timestamp, model)
            s3.put_object(Body = model_results, Bucket = "fyp-w9797878", Key = s3_name)
       
    # saving all         
    with open("{}/all/all-{}.txt".format(save_path, all_models), "w") as f:
        f.write(all_results)
           
    if save_in_s3:
        s3_name = "{}/all/all-{}-{}-{}.txt".format(current_timestamp,model, all_models)
        s3.put_object(Body = single_results, Bucket = "fyp-w9797878", Key = s3_name)
        
