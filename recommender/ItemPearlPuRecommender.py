"""

"""


from icecream import ic
from recommender.ItemRecommender import ItemRecommender

# !!! UNDER CONSTRUCTION !!! 

class ItemPearlPuRecommender(ItemRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("pp_rec.__init__()")

        super().__init__(dataset, **kwargs)
        self.weight_threshold = kwargs["run_params"]["weight_threshold"]
        self.recursion_threshold = kwargs["run_params"]["recursion_threshold"]
        self.phi = kwargs["run_params"]["phi"]
        self.k_prime = kwargs["run_params"]["k_prime"]
        self.baseline = kwargs["run_params"]["baseline"]
    
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        return self.recursive_prediction(active_user_id, candidate_item_id)

        
    def recursive_prediction(self, active_user: int, candidate_item: int, recursion_level: int = 1) -> float:
        """"""
        #ic("pp_rec.recursive_prediction()")
        
        # starts at 1
        #print("RECURSION PROGERSS = {}/{}".format(recursion_level, self.recursion_threshold))
        if recursion_level > self.recursion_threshold:
            #ic("Reached Recursion Limit - Using Baseline")
            if self.baseline == "bs":
                return self.predict_rating_item_based_nn_wtd(active_user, candidate_item) # baseline # is this even right? does it make sense? TODO

        nns = self.get_k_nearest_items(self.similarity_function, self.k, candidate_item)  # no user id, doesn't limit to just rated
        
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nns:
            neighbour_id = neighbour["item_id"]
            neighbour_user_rating = self.get_item_user_rating(neighbour_id, candidate_item)
            
            if neighbour_user_rating is not None:
                sim_x_y = self.get_item_similarity(self.similarity_function, candidate_item, neighbour_id)
                mean_rating_for_neighbour = self.get_item_mean_rating(neighbour_id)
                
                alpha += (neighbour_user_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(neighbour_id, candidate_item, recursion_level + 1)
                sim_x_y = self.get_user_similarity(self.similarity_function, active_user, neighbour_id)
                mean_rating_for_neighbour = self.get_user_mean_rating(neighbour_id)
                
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_active_item = self.get_item_mean_rating(candidate_item)
        
        if beta == 0.0:
            return mean_rating_for_active_item
        else:
            prediction = mean_rating_for_active_item + (alpha/beta)
            
            if prediction < 1.0:
                prediction = 1.0
                
            if prediction > 5:
                prediction = 5.0
    
            return round(prediction, self.ROUNDING)