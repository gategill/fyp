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
    if "k_folds" in  kwargs["testing_strategy"]["method"]:
        kwargs["dataset_config"]["k_folds"] = k_folds = kwargs["testing_strategy"]["method"]["k_fold"]
        
    elif "train_ratio" in  kwargs["testing_strategy"]["method"]:
        kwargs["dataset_config"]["train_ratio"] = train_ratio = kwargs["testing_strategy"]["method"]["train_ratio"]
        k_folds = 1
        
    elif "validation" in  kwargs["testing_strategy"]["method"]:
        kwargs["dataset_config"]["validation"] = validation = True
        kwargs["dataset_config"]["train_ratio"]  = kwargs["testing_strategy"]["method"]["validation"][0]
        kwargs["dataset_config"]["validation_ratio"]  = kwargs["testing_strategy"]["method"]["validation"][1]
        kwargs["dataset_config"]["test_ratio"] = kwargs["testing_strategy"]["method"]["validation"][2]
        k_folds = 1
        
    else:
        raise ValueError("invalid method in testing_strategy")

    save_in_s3 = kwargs["experiment_config"]["save_in_s3"]
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
    results_header += (len(results_header) * "-") + "\n"
    all_results = results_header
    
    for model in kwargs["models"]:
        model_mae = []
        best_recommender = []
        #print(kwargs)
        model_results = results_header            
        print("MODEL = {}".format(model))
        
        #for K in kwargs["experiment_config"]["neighbours"]:    
        for K in kwargs["models"][model]["neighbours"]: # for param_set in param_space
  
            print("NEIGHBOURS = {}".format(K))
            model_k_mae = []
            #model_k_rmse = []
            model_k_results = results_header

            for fold_num in range(k_folds):
                single_results = results_header
                
                print("FOLD NUMBER = {}/{}\n".format(fold_num + 1, k_folds))
            
                dataset.load_ratings(fold_num) if k_folds > 1 else dataset.load_ratings()
        
                try:                
                    print("running {} recommender".format(model))
                    #kwargs["models"][model]["neighbours"] = K
                    #kwargs["models"][model]["similarity"] = kwargs["experiment_config"]["similarity"]
                    kwargs["run_params"] = kwargs["models"][model]
                    kwargs["run_params"]["neighbours"] = K
                    
                    tic = time.time()
                    a_recommender = recommenders[model](dataset, **kwargs)
                    a_recommender.train()
                    which = "validation" if validation else None
                    a_prediction = a_recommender.predict(which = which) 
                    toc = time.time()
                    time_elapsed = round(toc - tic, 3)
                    
                    mae = a_recommender.evaluate_predictions()
                        
                    print(a_prediction, mae)
                    
                    model_k_mae.append(mae)
                    model_mae.append(mae)
                    
                    best_recommender = [K, a_recommender] if mae <= min(model_mae) else best_recommender
                    
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
                
            else:
                model_k_mean_mae = round(np.mean(model_k_mae), 5) 
            
            model_k_results += "{}_{}: Averaged_MAE = {}\n".format(model, K, model_k_mean_mae)
            model_results += "{}_{}: Averaged_MAE = {}\n".format(model, K, model_k_mean_mae)
            all_results += "{}_{}: Averaged_MAE = {}\n".format(model, K, model_k_mean_mae)            
            
            # saving model_k
            with open("{}/model_k/model_k-{}-{}.txt".format(save_path, model, K), "w") as f:
                f.write(model_k_results)
        
            if save_in_s3:
                s3_name = "{}/model_k/model_k-{}-{}.txt".format(current_timestamp, model, K)
                s3.put_object(Body = model_k_results, Bucket = "fyp-w9797878", Key = s3_name)
             
        print("RUnning VALIDATION")
        validated_model = best_recommender[1]
        best_k = best_recommender[0]
        validated_model.prepare_for_validation()
        validated_prediction = validated_model.predict()
        validated_mae = validated_model.evaluate_predictions()
    
        print(validated_prediction, validated_mae)
        
        model_results += "{}_{}: Validated_MAE = {}\n".format(model, best_k, validated_mae)
        all_results += "{}_{}: Validated_MAE = {}\n".format(model, best_k, validated_mae)           

                    
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
        s3.put_object(Body = all_results, Bucket = "fyp-w9797878", Key = s3_name)
        
