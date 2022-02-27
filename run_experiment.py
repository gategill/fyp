"""

"""


from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender
from recommender.PearlPuRecommender import PearlPuRecommender
from recommender.BootstrapRecommender import BootstrapRecommender
from recommender.CoRecRecommender import CoRecRecommender
from yaml_handler import read_in_yaml_file
from dataset.Dataset import Dataset
import traceback
import time
import os
import boto3
import random
import shutil
from icecream import ic
import numpy as np

s3 = boto3.client('s3')

def run_experiment(config_path) -> None:
    recommenders = {"UserKNN" : UserRecommender, 
                       "ItemKNN" : ItemRecommender,
                       "BootstrapKNN" : BootstrapRecommender,
                       "PearlPu" : PearlPuRecommender, 
                       "CoRec" : CoRecRecommender}
            
        
    kwargs = read_in_yaml_file(config_path)
    # pass some agruments down
    kwargs["config_path"] = config_path
    kwargs["dataset_config"]["kolds"] = kwargs["experiment_config"]["kolds"]
    

    save_in_s3 = kwargs["experiment_config"]["save_in_s3"]
    kolds = kwargs["experiment_config"]["kolds"]
    a_seed = kwargs["experiment_config"]["seed"]
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
    
    if a_seed != -1:
        random.seed(a_seed)
    
    if kwargs["experiment_config"]["disable_ic"]: 
        ic.disable()
        
    dataset = Dataset(**kwargs["dataset_config"])
    
    results_header = "algorithm, k, mae, time_elapsed_s, fold_num\n"
    all_results = results_header
    
    for model in kwargs["models"]:
        model_results = results_header            
        print("MODEL = {}".format(model))
        
        for K in kwargs["experiment_config"]["neighbours"]:    
            print("NEIGHBOURS = {}".format(K))
            model_k_mae = []
            model_k_results = results_header

            for fold_num in range(kolds):
                single_results = results_header
                
                print("FOLD NUMBER = {}/{}\n".format(fold_num + 1, kolds))
            
                dataset.load_ratings(fold_num)
        
                try:                
                    print("Running {} Recommender".format(model))
                    kwargs["models"][model]["neighbours"] = K
                    kwargs["run_params"] = kwargs["models"][model]
                    
                    tic = time.time()
                    a_recommender = recommenders[model](dataset, **kwargs)
                    a_recommender.train()
                    test = a_recommender.get_predictions()
                    toc = time.time()
                    time_elapsed = round(toc - tic, 3)
                    
                    mae = a_recommender.evaluate_predictions()
                        
                    del a_recommender

                    print(test, mae)
                    
                    model_k_mae.append(mae)
                    
                    experiment_result = "{}, {}, {}, {}, {}\n".format(model, K, mae, time_elapsed, fold_num + 1)
                    all_results += experiment_result
                    model_results += experiment_result
                    model_k_results += experiment_result
                    single_results += experiment_result
                    
                    with open("{}/single/single-{}-{}-{}.txt".format(save_path, model, K, fold_num + 1), "w") as f:
                        f.write(single_results)
                    
                    if save_in_s3:
                        s3_name = "{}/single/single-{}-{}-{}.txt".format(current_timestamp,model, K, fold_num)
                        s3.put_object(Body = single_results, Bucket = "fyp-w9797878", Key = s3_name)
                
                except Exception as e:
                    line_error = "error performing experiment, {}, error = {}".format(model, e)
                    print(traceback.print_exc())
                    
                    with open("{}/single/single-{}-{}-{}.txt".format(save_path, model, K, fold_num), "w") as f:
                        f.write(line_error)
                    
                    if save_in_s3:
                        s3_name = "{}/single/single-{}-{}-{}.txt".format(current_timestamp,model, K, fold_num)
                        s3.put_object(Body = line_error, Bucket = "fyp-w9797878", Key = s3_name)
                
                
            model_k_num_singles = len(model_k_mae)
            model_k_mean_mae = round(np.mean(model_k_mae), 5) 
            model_k_std = round(np.std(model_k_mae), 5)
            
            model_k_results += "{}_{}: Number of Folds = {}\n".format(model, K, model_k_num_singles)
            model_k_results += "{}_{}: Mean MAE = {}\n".format(model, K, model_k_mean_mae)
            model_k_results += "{}_{}: Standard Deviation = {}\n\n".format(model, K, model_k_std)
            
            model_results += "{}_{}: Number of Folds = {}\n".format(model, K, model_k_num_singles)
            model_results += "{}_{}: Mean MAE = {}\n".format(model, K, model_k_mean_mae)
            model_results += "{}_{}: Standard Deviation = {}\n\n".format(model, K, model_k_std)
            
            all_results += "{}_{}: Number of Folds = {}\n".format(model, K, model_k_num_singles)
            all_results += "{}_{}: Mean MAE = {}\n".format(model, K, model_k_mean_mae)
            all_results += "{}_{}: Standard Deviation = {}\n\n".format(model, K, model_k_std)
            
            
            with open("{}/model_k/model_k-{}-{}.txt".format(save_path, model, K), "w") as f:
                f.write(model_k_results)
        
            if save_in_s3:
                s3_name = "{}/model_k/model_k-{}-{}.txt".format(current_timestamp, model, K)
                s3.put_object(Body = model_k_results, Bucket = "fyp-w9797878", Key = s3_name)
             
             
        with open("{}/model/model-{}.txt".format(save_path, model), "w") as f:
            f.write(model_results)
        
        if save_in_s3:
            s3_name = "{}/model/model-{}.txt".format(current_timestamp, model)
            s3.put_object(Body = model_results, Bucket = "fyp-w9797878", Key = s3_name)
                
    with open("{}/all/all-{}.txt".format(save_path, all_models), "w") as f:
        f.write(all_results)
           
    if save_in_s3:
        s3_name = "{}/all/all-{}-{}-{}.txt".format(current_timestamp,model, all_models)
        s3.put_object(Body = single_results, Bucket = "fyp-w9797878", Key = s3_name)
        
