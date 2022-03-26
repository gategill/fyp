"""

"""


import copy
from icecream import ic
from dataset.Dataset import Dataset
from evaluation.Evaluation import Evaluation
from recommender.Similarities import Similarities


class GenericRecommender:
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("gen_rec.__init__()")
        
        self.kwargs = kwargs        
        self.predictions = []
        self.ROUNDING = 1
        
        self.k = self.kwargs["run_params"]["neighbours"]
        self.similarity_function = self.get_similarity_function()
        
        self.dataset = copy.deepcopy(dataset)

        
    def train(self):
        # ic("gen_rec.train()")
        self.load_dataset(**self.kwargs["dataset_config"])
        # could be more training stage...
        
        
    def get_single_prediction(self, **kwargs):
        raise NotImplementedError("implement this method")

    
    #def predict(self, which = None):
    #    entries = self.validation_ratings if which == "validation" else self.test_ratings
    #    #print(entries)
    #    for i, entry in enumerate(entries):
    #        #print(entry)
    #        try:
    #            user_id = int(entry['user_id'])
    #            item_id = int(entry['item_id'])
    #            
    #            predicted_rating = self.get_single_prediction(active_user_id = user_id, candidate_item_id = item_id)
    #                
    #            entry["pred_rating"] = predicted_rating
    #            self.add_prediction(entry)
    #            
    #            if self.kwargs["experiment_config"]["early_stop"]:
    #                if i > 30:
    #                    break
    #                    
    #        except KeyboardInterrupt:
    #            break
    #    #print(self.predictions)
    #    return self.predictions[-1]


    def evaluate_predictions(self,):
        # ic("gen_rec.evaluate_predictions()")
        mae = Evaluation.mean_absolute_error(self.predictions)
        return round(mae, 5)
    

    def get_similarity_function(self):
        s = self.kwargs["run_params"]["similarity"]
        
        if s == "sim_sim":
            return Similarities.sim_sim
        
        if s == "sim_pearson":
            return Similarities.sim_pearson
        
        if s == "sim_cosine":
            return Similarities.sim_cosine
        
        
    def load_dataset(self, **kwargs) -> None:
        #ic("gen_load_dataset()")
        
        if self.dataset is None:
            self.dataset = Dataset(**kwargs)
        
        self.user_ids = self.dataset.get_user_ids()
        self.item_ids = self.dataset.get_item_ids()
        self.user_train_ratings = self.dataset.get_user_train_ratings()
        self.user_train_means = self.dataset.get_user_train_means()
        self.item_train_ratings = self.dataset.get_item_train_ratings()
        self.item_train_means = self.dataset.get_item_train_means()
        self.train_ratings = self.dataset.get_train_ratings()
        
        self.user_test_ratings = self.dataset.get_user_test_ratings()
        self.item_test_ratings = self.dataset.get_item_test_ratings()
        self.test_ratings = self.dataset.get_test_ratings()
        
        self.user_validation_ratings = self.dataset.get_user_validation_ratings()
        self.item_validation_ratings = self.dataset.get_item_validation_ratings()
        self.validation_ratings = self.dataset.get_validation_ratings()
        
        self.mean_train_rating = self.dataset.get_mean_train_rating()
                

    def calculate_avg_rating(self, neighbours: list) -> float:
        #ic("gen_rec.calculate_avg_rating()")
        # [{sim, user_id, rating}]
        
        if len(neighbours) == 0:
            return None
        numerator = 0.0
        denominator = len(neighbours)
        
        for u_s_r in neighbours:
            rating = u_s_r['rating']
            numerator = numerator + rating
            
        if denominator <= 0.0:
            return None
        
        rating = numerator / denominator
        
        if rating < 1.0:
            rating = 1.0
            
        if rating > 5:
            rating = 5.0
                
        return round(rating, self.ROUNDING)
    

    def calculate_wtd_avg_rating(self, neighbours: list) -> float:
        # weighted, introduces similarity
        # [{sim, user_id, rating}]
        #ic("gen_rec.calculate_wtd_avg_rating()")
        
        if len(neighbours) == 0:
            return None
        numerator = 0.0
        denominator = 0.0
        
        for u_s_r in neighbours:
            rating = u_s_r['rating']
            sim = u_s_r['sim']
            numerator = numerator + sim * rating
            denominator = denominator + sim
            
        if denominator <= 0.0:
            return None
        
        rating = numerator / denominator
    
        if rating < 1.0:
            rating = 1.0
            
        if rating > 5:
            rating = 5.0
                
        return round(rating, self.ROUNDING)
    
    
    def add_new_recommendations(self, new_recommendations: list) -> None:
        """recommendation*s* = list if dicts"""
        #ic("gen_rec.add_new_recommendations()")
        
        self.dataset.add_new_recommendations_to_trainset(new_recommendations)

    #def prepare_for_validation(self):
    #    self.reset_predictions()
    #    self.add_new_recommendations(self.validation_ratings) # might cause abnormalities in augmentation
    
    
    def add_prediction(self, prediction: dict) -> None:
        """a single dict entry is addod to a list"""
        self.predictions.append(prediction)
        
    def reset_predictions(self):
        self.predictions = []


    def __str__(self) -> str:
        return "{},\n{},\n{}".format(self.k, self.similarity_function, self.dataset)