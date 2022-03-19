from numpy import append
import yaml


def read_in_yaml_file(config_path):
    with open(config_path) as f:
        kwargs = yaml.load(f, Loader = yaml.FullLoader)

    if "experiment_config" not in kwargs:
        raise KeyError("missing experiment_config in kwargs")
    #if "seed" not in kwargs["experiment_config"]:
    #    raise KeyError("missing seed in experiment_config")
    if "save_in_s3" not in kwargs["experiment_config"]:
        raise KeyError("missing save_in_s3 in experiment_config")
    if "kfolds" not in kwargs["experiment_config"]:
        raise KeyError("missing kfolds in experiment_config")
    if "early_stop" not in kwargs["experiment_config"]:
        raise KeyError("missing early_stop in experiment_config")
    if "disable_ic" not in kwargs["experiment_config"]:
        raise KeyError("missing disable_ic in experiment_config")
    if "evaluation_metrics" not in kwargs["experiment_config"]:
        raise KeyError("missing evaluation_metrics in experiment_config")
    if "paired_t_test" not in kwargs["experiment_config"]:
            raise KeyError("missing paired_t_test in experiment_config")


    #if (type(kwargs["experiment_config"]["seed"]) != int):
    #    raise TypeError("seed should be an integer")
    if type(kwargs["experiment_config"]["save_in_s3"]) != bool:
        raise TypeError("save_in_s3 should be boolean")
    if type(kwargs["experiment_config"]["kfolds"]) != int:
        raise TypeError("kfolds should be an integer")
    if type(kwargs["experiment_config"]["early_stop"]) != bool:
        raise TypeError("early_stop should be boolean")
    if type(kwargs["experiment_config"]["disable_ic"]) != bool:
        raise TypeError("disable_ic should be boolean")
    if type(kwargs["experiment_config"]["paired_t_test"]) != bool:
        raise TypeError("paired_t_test should be boolean")
    if type(kwargs["experiment_config"]["evaluation_metrics"]) != str:
        raise TypeError("evaluation_metrics should be string")

    
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
    for model in kwargs["models"]:
        if model not in ["UserKNN", "ItemKNN", "UserBootstrap","ItemBootstrap", "UserRecursiveKNN", "ItemRecursiveKNN", "CoRec", "MatrixFactorisation", "MostPop", "Random", "Mean", "ConfidentUserBootstrap","ConfidentItemBootstrap"]:
            raise KeyError("invalid model\nvalid model are: [UserKNN, ItemKNN, UserBootstrap, ItemBootstrap, UserRecursiveKNN, ItemRecursiveKNN, CoRec, MatrixFactorisation, MostPop, Random, Mean, ConfidentUserBootstrap,ConfidentItemBootstrap]")
        
    for model in kwargs["models"]:
        if "neighbours" not in kwargs["models"][model]:
            raise KeyError("missing neighbours in {}".format(model))
        if "similarity" not in kwargs["models"][model]:
             raise KeyError("missing similarity in {}".format(model))
         
        if type(kwargs["models"][model]["neighbours"]) not in [int, list]:
            raise TypeError("neighbours should be a list of integers list")
            
        #if type(kwargs["models"][model]["neighbours"]) == list:
        #    for nn in kwargs["models"][model]:
        #        if type(nn) != int:
        #            raise TypeError("neighbours should be integers in a list")
                
        if type(kwargs["models"][model]["neighbours"]) == int:
            li = []
            li.append(kwargs["models"][model]["neighbours"])
            kwargs["models"][model]["neighbours"] = li
            
        if type(kwargs["models"][model]["similarity"]) != str:
                raise TypeError("similarity should be an string") 
            
        if kwargs["models"][model]["similarity"] not in ["sim_pearson", "sim_cosine", "sim_sim"]:
            raise ValueError("invalid similarity measure\navailable similarites are: [sim_pearson, sim_cosine, sim_sim]")
            
            
    if "Bootstrap" in kwargs["models"]:
        if "enrichments" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing enrichments in Bootstrap")
        if "additions" not in kwargs["models"]["Bootstrap"]:
            raise KeyError("missing additions in Bootstrap")

        if type(kwargs["models"]["Bootstrap"]["enrichments"]) != int:
            raise TypeError("Bootstrap.enrichments should be an integer")
        if type(kwargs["models"]["Bootstrap"]["additions"]) != int:
            raise TypeError("Bootstrap.additions should be an integer")
        
        
    if "UserRecursiveKNN" in kwargs["models"]:
        if "weight_threshold" not in kwargs["models"]["UserRecursiveKNN"]:
            raise KeyError("missing weight_threshold in UserRecursiveKNN")
        if "recursion_threshold" not in kwargs["models"]["UserRecursiveKNN"]:
            raise KeyError("missing recursion_threshold in UserRecursiveKNN")
        if "phi" not in kwargs["models"]["UserRecursiveKNN"]:
            raise KeyError("missing phi in UserRecursiveKNN")
        if "k_prime" not in kwargs["models"]["UserRecursiveKNN"]:
            raise KeyError("missing k_prime in UserRecursiveKNN")
        if "neighbour_selection" not in kwargs["models"]["UserRecursiveKNN"]:
            raise KeyError("missing neighbour_selection in UserRecursiveKNN")

        if type(kwargs["models"]["UserRecursiveKNN"]["weight_threshold"]) != float:
            raise TypeError("UserRecursiveKNN.weight_threshold should be a float")
        if type(kwargs["models"]["UserRecursiveKNN"]["recursion_threshold"]) != int:
            raise TypeError("UserRecursiveKNN.recursion_threshold should be an integer")
        if type(kwargs["models"]["UserRecursiveKNN"]["phi"]) != int:
            raise TypeError("UserRecursiveKNN.phi should be an integer")
        if type(kwargs["models"]["UserRecursiveKNN"]["k_prime"]) != int:
            raise TypeError("UserRecursiveKNN.k_prime should be an integer")
        if type(kwargs["models"]["UserRecursiveKNN"]["neighbour_selection"]) != str:
            raise TypeError("UserRecursiveKNN.neighbour_selection should be a string")
        
        if kwargs["models"]["UserRecursiveKNN"]["neighbour_selection"] not in ["bs", "bs+", "ss", "cs", "cs+"]:
            raise ValueError("invalid UserRecursiveKNN.neighbour_selection\navailable neighbour_selections are: [bs, bs+, ss, cs, cs+]")
        

    if "ItemRecursiveKNN" in kwargs["models"]:
        if "weight_threshold" not in kwargs["models"]["ItemRecursiveKNN"]:
            raise KeyError("missing weight_threshold in ItemRecursiveKNN")
        if "recursion_threshold" not in kwargs["models"]["ItemRecursiveKNN"]:
            raise KeyError("missing recursion_threshold in ItemRecursiveKNN")
        if "phi" not in kwargs["models"]["ItemRecursiveKNN"]:
            raise KeyError("missing phi in ItemRecursiveKNN")
        if "k_prime" not in kwargs["models"]["ItemRecursiveKNN"]:
            raise KeyError("missing k_prime in ItemRecursiveKNN")
        if "neighbour_selection" not in kwargs["models"]["ItemRecursiveKNN"]:
            raise KeyError("missing neighbour_selection in ItemRecursiveKNN")

        if type(kwargs["models"]["ItemRecursiveKNN"]["weight_threshold"]) != float:
            raise TypeError("ItemRecursiveKNN.weight_threshold should be a float")
        if type(kwargs["models"]["ItemRecursiveKNN"]["recursion_threshold"]) != int:
            raise TypeError("ItemRecursiveKNN.recursion_threshold should be an integer")
        if type(kwargs["models"]["ItemRecursiveKNN"]["phi"]) != int:
            raise TypeError("ItemRecursiveKNN.phi should be an integer")
        if type(kwargs["models"]["ItemRecursiveKNN"]["k_prime"]) != int:
            raise TypeError("ItemRecursiveKNN.k_prime should be an integer")
        if type(kwargs["models"]["ItemRecursiveKNN"]["neighbour_selection"]) != str:
            raise TypeError("ItemRecursiveKNN.neighbour_selection should be a string")
        
        if kwargs["models"]["ItemRecursiveKNN"]["neighbour_selection"] not in ["bs", "bs+", "ss", "cs", "cs+"]:
            raise ValueError("invalid ItemRecursiveKNN.neighbour_selection\navailable neighbour_selections are: [bs, bs+, ss, cs, cs+]")
        
        
    if "CoRec" in kwargs["models"]:
        if "additions" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing additions in CoRec")
        if "top_m" not in kwargs["models"]["CoRec"]:
            raise KeyError("missing top_m in CoRec")

        if type(kwargs["models"]["CoRec"]["additions"]) != int:
            raise TypeError("CoRec.additions should be an integer")
        #if type(kwargs["models"]["CoRec"]["top_m"]) != int:
        #    raise TypeError("CoRec.top_m should be an integer")
        #if kwargs["models"]["CoRec"]["top_m"] < kwargs["models"]["CoRec"]["additions"]:
        #    raise ValueError("CoRec.top_m should be larger than additions (I think)")
        
        
    if "MatrixFactorisation" in kwargs["models"]:
        if "R" not in kwargs["models"]["MatrixFactorisation"]:
            raise KeyError("missing R in MatrixFactorisation")
        if "P" not in kwargs["models"]["MatrixFactorisation"]:
            raise KeyError("missing P in MatrixFactorisation")
        if "Q" not in kwargs["models"]["MatrixFactorisation"]:
            raise KeyError("missing Q in MatrixFactorisation")
        if "K" not in kwargs["models"]["MatrixFactorisation"]:
            raise KeyError("missing K in MatrixFactorisation")
        if "steps" not in kwargs["models"]["MatrixFactorisation"]:
            raise KeyError("missing steps in MatrixFactorisation")
        if "alpha" not in kwargs["models"]["MatrixFactorisation"]:
            raise KeyError("missing alpha in MatrixFactorisation")
        if "beta" not in kwargs["models"]["MatrixFactorisation"]:
            raise KeyError("missing beta in MatrixFactorisation")

        
    if "MostPop" in kwargs["models"]:
        if "top_m" not in kwargs["models"]["MostPop"]:
            raise KeyError("missing top_m in MostPop")
        
        
    if "Mean" in kwargs["models"]:
        if "which" not in kwargs["models"]["Mean"]:
            raise KeyError("missing which in Mean")
        
        
    if "ConfidentUserBootstrap" in kwargs["models"]:
        if "enrichments" not in kwargs["models"]["ConfidentUserBootstrap"]:
            raise KeyError("missing enrichments in ConfidentUserBootstrap")
        if "additions" not in kwargs["models"]["ConfidentUserBootstrap"]:
            raise KeyError("missing additions in ConfidentUserBootstrap")
        if "top_m" not in kwargs["models"]["ConfidentUserBootstrap"]:
            raise KeyError("missing top_m in ConfidentUserBootstrap")

        if type(kwargs["models"]["ConfidentUserBootstrap"]["enrichments"]) != int:
            raise TypeError("ConfidentUserBootstrap.enrichments should be an integer")
        if type(kwargs["models"]["ConfidentUserBootstrap"]["additions"]) != int:
            raise TypeError("ConfidentUserBootstrap.additions should be an integer")
        if type(kwargs["models"]["ConfidentUserBootstrap"]["top_m"]) != int:
            raise TypeError("ConfidentUserBootstrap.top_m should be an integer")
        if kwargs["models"]["ConfidentUserBootstrap"]["top_m"] < kwargs["models"]["ConfidentUserBootstrap"]["additions"]:
            raise ValueError("ConfidentUserBootstrap.top_m should be larger than additions (I think)")
        
        
    if "ConfidentItemBootstrap" in kwargs["models"]:
        if "enrichments" not in kwargs["models"]["ConfidentItemBootstrap"]:
            raise KeyError("missing enrichments in ConfidentItemBootstrap")
        if "additions" not in kwargs["models"]["ConfidentItemBootstrap"]:
            raise KeyError("missing additions in ConfidentItemBootstrap")
        if "top_m" not in kwargs["models"]["ConfidentItemBootstrap"]:
            raise KeyError("missing top_m in ConfidentItemBootstrap")

        if type(kwargs["models"]["ConfidentItemBootstrap"]["enrichments"]) != int:
            raise TypeError("ConfidentItemBootstrap.enrichments should be an integer")
        if type(kwargs["models"]["ConfidentItemBootstrap"]["additions"]) != int:
            raise TypeError("ConfidentItemBootstrap.additions should be an integer")
        if type(kwargs["models"]["ConfidentItemBootstrap"]["top_m"]) != int:
            raise TypeError("ConfidentItemBootstrap.top_m should be an integer")
        if kwargs["models"]["ConfidentItemBootstrap"]["top_m"] < kwargs["models"]["ConfidentItemBootstrap"]["additions"]:
            raise ValueError("ConfidentItemBootstrap.top_m should be larger than additions (I think)")
        
            
    return kwargs