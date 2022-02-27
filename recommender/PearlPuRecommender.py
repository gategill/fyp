"""

"""


from icecream import ic
from recommender.UserRecommender import UserRecommender


class PearlPuRecommender(UserRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("pp_rec.__init__()")

        super().__init__(dataset, **kwargs)
        self.weight_threshold = kwargs["run_params"]["weight_threshold"]
        self.recursion_threshold = kwargs["run_params"]["recursion_threshold"]
        self.phi = kwargs["run_params"]["phi"]
        self.k_prime = kwargs["run_params"]["k_prime"]
        self.baseline = kwargs["run_params"]["baseline"]
        
        
    def recursive_prediction(self, active_user: int, candidate_moive: int, recursion_level: int = 1) -> float:
        """"""
        #ic("pp_rec.recursive_prediction()")
        
        # starts at 1
        if recursion_level > self.recursion_threshold:
            #ic("Reached Recursion Limit - Using Baseline")
            if self.baseline == "bs":
                return self.predict_rating_user_based_nn_wtd(active_user, candidate_moive) # baseline

        nns = self.get_k_nearest_users(self.similarity_function, self.k, active_user)  # no item id, doesn't limit to just rated
        
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nns:
            neighbour_id = neighbour["user_id"]
            neighbour_item_rating = self.get_user_item_rating(neighbour_id, candidate_moive)
            
            if neighbour_item_rating is not None:
                sim_x_y = self.get_user_similarity(self.similarity_function, active_user, neighbour_id)
                mean_rating_for_neighbour = self.get_user_mean_rating(neighbour_id)
                
                alpha += (neighbour_item_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(neighbour_id, candidate_moive, recursion_level + 1)
                sim_x_y = self.get_user_similarity(self.similarity_function, active_user, neighbour_id)
                mean_rating_for_neighbour = self.get_user_mean_rating(neighbour_id)
                
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_active_user = self.get_user_mean_rating(active_user)
        
        if beta == 0.0:
            return mean_rating_for_active_user
        else:
            prediction = mean_rating_for_active_user + (alpha/beta)
            
            if prediction < 1.0:
                prediction = 1.0
                
            if prediction > 5:
                prediction = 5.0
    
            return prediction