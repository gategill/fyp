"""

"""


from icecream import ic
from recommender.UserRecommender import UserRecommender
from recommender.Similarities import Similarities


class PearlPuRecommender(UserRecommender):
    def __init__(self, k: int = 5,  weight_threshold: float = 1.0, recursion_threshold: int = 3, dataset = None) -> None:
        ic("pp_rec.__init__()")

        super().__init__(k, dataset)
        self.weight_threshold = weight_threshold
        self.recursion_threshold = recursion_threshold
        #self.user_rec = UserRecommender(self.k)
        #self.test_ratings = self.user_rec.test_ratings
        
        
    def recursive_prediction(self, active_user: int, candidate_moive: int, recursion_level: int = 1) -> float:
        """"""
        ic("pp_rec.recursive_prediction()")
        ic(recursion_level)
        
        if recursion_level > self.recursion_threshold:
            ic("Reached Recursion Limit - Using Baseline")
            
            return self.predict_rating_user_based_nn_wtd(active_user, candidate_moive) # baseline

        nns = self.get_k_nearest_users(Similarities.sim_pearson, self.k, active_user)  # no item id, doesn't limit to just rated
        
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nns:
            neighbour_id = neighbour["user_id"]
            neighbour_item_rating = self.get_user_item_rating(neighbour_id, candidate_moive)
            
            if neighbour_item_rating is not None:
                sim_x_y = self.get_user_similarity(Similarities.sim_pearson, active_user, neighbour_id)
                mean_rating_for_neighbour = self.get_user_mean_rating(neighbour_id)
                
                #ic(sim_x_y)
                alpha += (neighbour_item_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(neighbour_id, candidate_moive, recursion_level + 1)
                sim_x_y = self.get_user_similarity(Similarities.sim_pearson, active_user, neighbour_id)
                #ic(sim_x_y)
                mean_rating_for_neighbour = self.get_user_mean_rating(neighbour_id)
                
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_active_user = self.get_user_mean_rating(active_user)
        
        #ic(alpha)
        #ic(beta)
        
        
        if beta == 0.0:
            return mean_rating_for_active_user
        else:
            return mean_rating_for_active_user + (alpha/beta)