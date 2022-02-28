from numpy import append
import yaml


def read_in_yaml_file(config_path):
    with open(config_path) as f:
        kwargs = yaml.load(f, Loader = yaml.FullLoader)

    if "experiment_config" not in kwargs:
        raise KeyError("missing experiment_config in kwargs")
    if "seed" not in kwargs["experiment_config"]:
        raise KeyError("missing seed in experiment_config")
    if "save_in_s3" not in kwargs["experiment_config"]:
        raise KeyError("missing save_in_s3 in experiment_config")
    if "kolds" not in kwargs["experiment_config"]:
        raise KeyError("missing kolds in experiment_config")
    if "early_stop" not in kwargs["experiment_config"]:
        raise KeyError("missing early_stop in experiment_config")
    if "disable_ic" not in kwargs["experiment_config"]:
        raise KeyError("missing disable_ic in experiment_config")
    if "weighted_ratings" not in kwargs["experiment_config"]:
        raise KeyError("missing weighted_ratings in experiment_config")
    if "evaluation_metrics" not in kwargs["experiment_config"]:
        raise KeyError("missing evaluation_metrics in experiment_config")
    if "neighbours" not in kwargs["experiment_config"]:
        raise KeyError("missing neighbours in experiment_config")
    if "similarity" not in kwargs["experiment_config"]:
            raise KeyError("missing similarity in experiment_config")
    
    
    if type(kwargs["experiment_config"]["neighbours"]) not in [int, list]:
        raise TypeError("neighbours should be a list of integers list")
    
        
    if type(kwargs["experiment_config"]["neighbours"]) == list:
        for nn in kwargs["experiment_config"]["neighbours"]:
            if type(nn) != int:
                raise TypeError("neighbours should be integers in a list")
            
    if type(kwargs["experiment_config"]["neighbours"]) == int:
        li = []
        li.append(kwargs["experiment_config"]["neighbours"])
        kwargs["experiment_config"]["neighbours"] = li


    if (type(kwargs["experiment_config"]["seed"]) != int):
        raise TypeError("seed should be an integer")
    if type(kwargs["experiment_config"]["save_in_s3"]) != bool:
        raise TypeError("save_in_s3 should be boolean")
    if type(kwargs["experiment_config"]["kolds"]) != int:
        raise TypeError("kolds should be an integer")
    if type(kwargs["experiment_config"]["early_stop"]) != bool:
        raise TypeError("early_stop should be boolean")
    if type(kwargs["experiment_config"]["disable_ic"]) != bool:
        raise TypeError("disable_ic should be boolean")
    if type(kwargs["experiment_config"]["evaluation_metrics"]) != (list or str) :
        raise TypeError("evaluation_metrics should be string or list")
    if type(kwargs["experiment_config"]["similarity"]) != str:
            raise TypeError("similarity should be an string") 
    if kwargs["experiment_config"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
        raise ValueError("invalid similarity measure\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")

    
    if "dataset_config" not in kwargs:
        raise KeyError("missing dataset_config in kwargs")
    if "dataset_path" not in kwargs["dataset_config"]:
        raise KeyError("missing dataset_path in dataset_config")
    if "prefiltering" not in kwargs["dataset_config"]:
        raise KeyError("missing prefiltering in dataset_config")
    
    if type(kwargs["dataset_config"]["dataset_path"]) != str:
        raise TypeError("dataset_path should be a string")
    if type(kwargs["dataset_config"]["prefiltering"]) != dict:
        raise TypeError("prefiltering should be a dictionary")
 
    if "models" not in kwargs:
        raise KeyError("missing models in kwargs")
    if len(kwargs["models"]) == 0:
        raise ValueError("no models provided in kwargs[models]")
        
    if "Bootstrap" in kwargs["models"]:
        if "enrichments" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing enrichments in Bootstrap")
        if "additions" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing additions in Bootstrap")

        if type(kwargs["models"]["Bootstrap"]["enrichments"]) != int:
            raise TypeError("Bootstrap.enrichments should be an integer")
        if type(kwargs["models"]["Bootstrap"]["additions"]) != int:
            raise TypeError("Bootstrap.additions should be an integer")
        
        
    if "RecursiveKNN" in kwargs["models"]:
        if "weight_threshold" not in kwargs["models"]["RecursiveKNN"]:
            raise KeyError("missing weight_threshold in RecursiveKNN")
        if "recursion_threshold" not in kwargs["models"]["RecursiveKNN"]:
            raise KeyError("missing recursion_threshold in RecursiveKNN")
        if "phi" not in kwargs["models"]["RecursiveKNN"]:
            raise KeyError("missing phi in RecursiveKNN")
        if "k_prime" not in kwargs["models"]["RecursiveKNN"]:
            raise KeyError("missing k_prime in RecursiveKNN")
        if "baseline" not in kwargs["models"]["RecursiveKNN"]:
            raise KeyError("missing baseline in RecursiveKNN")

        if type(kwargs["models"]["RecursiveKNN"]["weight_threshold"]) != float:
            raise TypeError("RecursiveKNN.weight_threshold should be a float")
        if type(kwargs["models"]["RecursiveKNN"]["recursion_threshold"]) != int:
            raise TypeError("RecursiveKNN.recursion_threshold should be an integer")
        if type(kwargs["models"]["RecursiveKNN"]["phi"]) != int:
            raise TypeError("RecursiveKNN.phi should be an integer")
        if type(kwargs["models"]["RecursiveKNN"]["k_prime"]) != int:
            raise TypeError("RecursiveKNN.k_prime should be an integer")
        if type(kwargs["models"]["RecursiveKNN"]["baseline"]) != str:
            raise TypeError("RecursiveKNN.baseline should be a string")
        
        if kwargs["models"]["RecursiveKNN"]["baseline"] not in ["bs", "bs+", "ss", "cs", "cs+"]:
            raise ValueError("invalid RecursiveKNN.baseline\navailable baselines are: [bs, bs+, ss, cs, cs+]")
        
    if "CoRec" in kwargs["models"]:
        if "additions" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing additions in CoRec")
        if "top_m" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing top_m in CoRec")

        if type(kwargs["models"]["CoRec"]["additions"]) != int:
            raise TypeError("CoRec.additions should be an integer")
        if type(kwargs["models"]["CoRec"]["top_m"]) != int:
            raise TypeError("CoRec.top_m should be an integer")
        if kwargs["models"]["CoRec"]["top_m"] < kwargs["models"]["CoRec"]["additions"]:
            raise ValueError("CoRec.top_m should be larger than additions (I think)")

            
    return kwargs