"""

"""


from icecream import ic
from recommender.ItemKNNRecommender import ItemKNNRecommender
import copy
import time
from tqdm import tqdm

class ItemRecursiveKNNRecommender(ItemKNNRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("pp_rec.__init__()")

        super().__init__(dataset, **kwargs)
        self.weight_threshold = kwargs["run_params"]["weight_threshold"]
        self.recursion_threshold = kwargs["run_params"]["recursion_threshold"]
        self.phi = kwargs["run_params"]["phi"]
        self.k_prime = self.k #kwargs["run_params"]["k_prime"]
        self.neighbour_selection = kwargs["run_params"]["neighbour_selection"]
        self.hashed_predictions = {}
    
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        return self.recursive_prediction(active_user_id, candidate_item_id)
      
    def recursive_prediction(self, active_user: int, candidate_item: int, recursion_level: int = 0) -> float:
        """"""
        #ic("pp_rec.recursive_prediction()")
        active_user = int(active_user)
        candidate_item = int(candidate_item)
        # starts at 0
        if recursion_level > self.recursion_threshold:
            hashkey_key = str(active_user) + "-"+ str(candidate_item)
            if hashkey_key in self.hashed_predictions:
                #print("HIT!")
                return self.hashed_predictions[hashkey_key]
            else:
                pr = self.baseline_predictor(active_user, candidate_item)
                #print(pr)
                self.hashed_predictions[hashkey_key] = pr
                return pr

        # no item id, doesn't limit to just rated
        nns = self.nearest_neighbour_seletion(active_user, candidate_item)
        
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nns:
            neighbour_id = int(neighbour["item_id"])
            neighbour_item_rating = self.get_user_item_rating(user_id = active_user, item_id = neighbour_id) # should change?
            
            if neighbour_item_rating is not None: # has a rating
                sim_x_y = self.get_item_similarity(self.similarity_function, candidate_item, neighbour_id)
                mean_rating_for_neighbour = self.get_item_mean_rating(neighbour_id)
                
                alpha += (neighbour_item_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(active_user_id = active_user, candidate_item_id = neighbour_id, recursion_level = recursion_level + 1)
                sim_x_y = self.get_item_similarity(self.similarity_function, candidate_item, neighbour_id)
                mean_rating_for_neighbour = self.get_item_mean_rating(neighbour_id)
                
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_candidate_item = self.get_item_mean_rating(candidate_item)
        if beta == 0.0:
            
            return mean_rating_for_candidate_item
        else:
            prediction = mean_rating_for_candidate_item + (alpha/beta)
            
            if prediction < 1.0:
                prediction = 1.0
                
            if prediction > 5:
                prediction = 5.0
    
            return round(prediction, self.ROUNDING)
        
        
    def nearest_neighbour_seletion(self, active_user, candidate_item):
        if self.neighbour_selection == "bs":
            nns = self.get_k_nearest_items(self.similarity_function, self.k, candidate_item_id = candidate_item, active_user_id = active_user)
                        
        elif self.neighbour_selection == "bs+":
            nns = self.get_k_nearest_items_with_overlap(self.similarity_function, k = self.k, candidate_item_id = candidate_item, active_user_id = active_user, overlap = self.phi)

        elif self.neighbour_selection == "ss":
            nns = self.get_k_nearest_items(self.similarity_function, k = self.k_prime, candidate_item_id = candidate_item, active_user_id =  None)         
                
        elif self.neighbour_selection == "cs":
            nns1 = self.get_k_nearest_items(self.similarity_function, k = self.k, candidate_item_id = candidate_item, active_user_id = active_user)
            nns2 = self.get_k_nearest_items(self.similarity_function, k = self.k_prime, candidate_item_id = candidate_item, active_user_id = None)
            nns = nns1 + nns2
          
        elif self.neighbour_selection == "cs+":
            nns1 = self.get_k_nearest_items_with_overlap(self.similarity_function, k = self.k, candidate_item_id = candidate_item, active_user_id = active_user, overlap = self.phi)
            nns2 = self.get_k_nearest_items_with_overlap(self.similarity_function, k = self.k_prime, candidate_item_id = candidate_item, active_user_id = None, overlap = self.phi)
            nns = nns1 + nns2
            
        else:
            raise KeyError("Invalid neighbour_selection strategy")
        
        return nns
          
          
    def baseline_predictor(self, active_user, candidate_item):
        # baseline predictor = BS
        nns = self.get_k_nearest_items(self.similarity_function, self.k, active_user_id =  active_user, candidate_item_id = candidate_item)
        prediction = self.calculate_wtd_avg_rating(nns)
        
        if prediction:  
            return prediction
        else:
            prediction = self.get_item_mean_rating(candidate_item)
            
            if prediction:
                return prediction
            else:
                return self.mean_train_rating      