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
    
    if "dataset_config" not in kwargs:
        raise KeyError("missing dataset_config in kwargs")
    if "dataset_path" not in kwargs["dataset_config"]:
        raise KeyError("missing dataset_path in dataset_config")
    if "prefiltering" not in kwargs["dataset_config"]:
        raise KeyError("missing prefiltering in dataset_config")
    #if "test_splitting_ratio" not in kwargs["dataset_config"]:
    #    raise KeyError("missing test_splitting_ratio in dataset_config")
    
    if type(kwargs["dataset_config"]["dataset_path"]) != str:
        raise TypeError("dataset_path should be a string")
    if type(kwargs["dataset_config"]["prefiltering"]) != dict:
        raise TypeError("prefiltering should be a dictionary")
    #if type(kwargs["dataset_config"]["test_splitting_ratio"]) != float:
    #    raise TypeError("test_splitting_ratio should be a float")
    
    
    if "models" not in kwargs:
        raise KeyError("missing models in kwargs")
    if len(kwargs["models"]) == 0:
        raise ValueError("no models provided in kwargs[models]")
    
    if "UserKNN" in kwargs["models"]:
        if "similarity" not in kwargs["models"]["UserKNN"]:
            raise KeyError("missing similarity in UserKNN")
        
        if type(kwargs["models"]["UserKNN"]["similarity"]) != str:
            raise TypeError("UserKNN.similarity should be an string")
        
        if kwargs["models"]["UserKNN"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid UserKNN.similarity measure\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
        
        
    if "ItemKNN" in kwargs["models"]:
        if "similarity" not in kwargs["models"]["ItemKNN"]:
            raise KeyError("missing similarity in ItemKNN")

        if type(kwargs["models"]["ItemKNN"]["similarity"]) != str:
            raise TypeError("ItemKNN.similarity should be an string")
        
        if kwargs["models"]["ItemKNN"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid ItemKNN.similarity\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
        
        
    if "Bootstrap" in kwargs["models"]:
        if "similarity" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing similarity in Bootstrap")
        if "fold_nums" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing fold_nums in Bootstrap")
        if "additions" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing additions in Bootstrap")

        if type(kwargs["models"]["Bootstrap"]["similarity"]) != str:
            raise TypeError("Bootstrap.similarity should be an string")
        if type(kwargs["models"]["Bootstrap"]["fold_nums"]) != int:
            raise TypeError("Bootstrap.fold_nums should be an integer")
        if type(kwargs["models"]["Bootstrap"]["additions"]) != int:
            raise TypeError("Bootstrap.additions should be an integer")
        
        if kwargs["models"]["Bootstrap"]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid Bootstrap.similarity\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
        
        
    if "PearlPu" in kwargs["models"]:
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
        if "similarity" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing similarity in CoRec")
        if "additions" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing additions in CoRec")
        if "top_m" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing top_m in CoRec")

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