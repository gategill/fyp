from recommender.UserRecursiveKNNRecommender import UserRecursiveKNNRecommender
from recommender.ItemRecursiveKNNRecommender import ItemRecursiveKNNRecommender
from recommender.UserBootstrapRecommender import UserBootstrapRecommender
from recommender.ItemBootstrapRecommender import ItemBootstrapRecommender
from recommender.ItemKNNRecommender import ItemKNNRecommender
from recommender.UserKNNRecommender import UserKNNRecommender
from recommender.CoRecRecommender import CoRecRecommender
from dataset.Dataset import Dataset

from icecream import ic
import YAMLHandler
import itertools
import random
import shutil
import boto3
import time
import os

s3 = boto3.client('s3')

def run_experiment(config_path) -> None:
    recommenders = {"UserKNN" : UserKNNRecommender, 
                       "ItemKNN" : ItemKNNRecommender,
                       "UserBootstrap" : UserBootstrapRecommender,
                       "ItemBootstrap" : ItemBootstrapRecommender,
                       "UserRecursiveKNN" : UserRecursiveKNNRecommender, 
                       "ItemRecursiveKNN" : ItemRecursiveKNNRecommender, 
                       "CoRec" : CoRecRecommender
                       }
            
    print(u''' .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |  _________   | || |   _____      | || |  ____  ____  | || |     ____     | || |  _________   | |
| | |_   ___  |  | || |  |_   _|     | || | |_  _||_  _| | || |   .'    `.   | || | |  _   _  |  | |
| |   | |_  \_|  | || |    | |       | || |   \ \  / /   | || |  /  .--.  \  | || | |_/ | | \_|  | |
| |   |  _|  _   | || |    | |   _   | || |    \ \/ /    | || |  | |    | |  | || |     | |      | |
| |  _| |___/ |  | || |   _| |__/ |  | || |    _|  |_    | || |  \  `--'  /  | || |    _| |_     | |
| | |_________|  | || |  |________|  | || |   |______|   | || |   `.____.'   | || |   |_____|    | |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------' ''')
    time.sleep(1)
        
    kwargs = YAMLHandler.read_in_yaml_file(config_path)
    # pass some agruments down
    kwargs["config_path"] = config_path
    kwargs["dataset_config"]["kfolds"] = kwargs["experiment_config"]["kfolds"]
    kwargs["dataset_config"]["validation"] = kwargs["experiment_config"]["validation"]

    save_in_s3 = kwargs["experiment_config"]["save_in_s3"]
    kfolds = kwargs["experiment_config"]["kfolds"]
    if "seed" in kwargs["experiment_config"]:
        a_seed = kwargs["experiment_config"]["seed"]
        random.seed(a_seed)
        
    current_timestamp = int(time.time())
    print(current_timestamp)
    save_name = kwargs["experiment_config"]["save_name"]
    save_path = "./results/{}-{}".format(current_timestamp, save_name)
    if not os.path.exists("./results"):
        os.mkdir("./results")
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
        try:
            model_mae = []

            model_results = results_header            
            print("MODEL = {}".format(model))
            
            parameter_space = get_parameter_space(kwargs["models"][model])
            for parameter_set in parameter_space:
                print("\nPARAMETERS = {}".format(parameter_set))
                all_param_val =[str(v) for v in parameter_set.values()]
                all_param_val = "_".join(all_param_val)
                model_k_mae = []
                model_k_results = results_header

                for fold_num in range(kfolds):
                    single_results = results_header
                    
                    print("FOLD NUMBER = {}/{}\n".format(fold_num + 1, kfolds))
                
                    dataset.load_ratings(fold_num) if kfolds > 1 else dataset.load_ratings()
                            
                    print("Running {} Recommender".format(model))

                    kwargs["run_params"] = parameter_set
                    
                    tic = time.time()
                    a_recommender = recommenders[model](dataset, **kwargs)
                    a_recommender.train()

                    print("\nGetting Predictions\n")
                    test = a_recommender.get_predictions()
                    toc = time.time()
                    time_elapsed = round(toc - tic, 3)
                    
                    mae = a_recommender.evaluate_predictions()
                        
                    print(test, mae)
                    
                    model_k_mae.append(mae)
                    model_mae.append(mae)
                                        
                    experiment_result = "{}, {}, {}, {}, {}\n".format(model, mae, time_elapsed, fold_num + 1, all_param_val)
                    all_results += experiment_result
                    model_results += experiment_result
                    model_k_results += experiment_result
                    single_results += experiment_result
                    
                    # saving singles
                    with open("{}/single/single-{}-{}-{}.txt".format(save_path, model, fold_num + 1, all_param_val), "w") as f:
                        f.write(single_results)
                    
                    if save_in_s3:
                        s3_name = "{}/single/single-{}-{}-{}.txt".format(current_timestamp,model, fold_num, all_param_val)
                        s3.put_object(Body = single_results, Bucket = "fyp-w9797878", Key = s3_name)       
                
                # saving model_k
                with open("{}/model_k/model_k-{}-{}.txt".format(save_path, model, all_param_val), "w") as f:
                    f.write(model_k_results)
            
                if save_in_s3:
                    s3_name = "{}/model_k/model_k-{}-{}.txt".format(current_timestamp, model, all_param_val)
                    s3.put_object(Body = model_k_results, Bucket = "fyp-w9797878", Key = s3_name)
                        
            # saving model   
            with open("{}/model/model-{}.txt".format(save_path, model), "w") as f:
                f.write(model_results)
            
            if save_in_s3:
                s3_name = "{}/model/model-{}.txt".format(current_timestamp, model)
                s3.put_object(Body = model_results, Bucket = "fyp-w9797878", Key = s3_name)
       
        except KeyboardInterrupt:
            break 
        
    # saving all         
    with open("{}/all/all-{}.txt".format(save_path, all_models), "w") as f:
        f.write(all_results)
           
    if save_in_s3:
        s3_name = "{}/all/all-{}.txt".format(current_timestamp, all_models)
        s3.put_object(Body = all_results, Bucket = "fyp-w9797878", Key = s3_name)
        
        
def get_parameter_space(model_params):
    output_list = []
    
    # turn parameters to lists
    for k, v in model_params.items():
        if type(v) != list:
            l = list()
            l.append(v)
            model_params[k] = l

    param_value_list = [v for v in model_params.values()]
    all_combinations = list(itertools.product(*param_value_list))

    for param_set in all_combinations:
        dict_of_params = {}
        for i, k in enumerate(list(model_params.keys())):
            dict_of_params[k] = param_set[i]
        output_list.append(dict_of_params)
        
    return output_list
        