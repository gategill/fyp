"""

"""


from icecream import ic
from recommender.UserRecommender import UserRecommender
from recommender.Similarities import Similarities


class PearlPuRecommender:
    def __init__(self, k: int = 5,  weight_threshold: float = 1.0, recursion_threshold: int = 3) -> None:
        #super().__init__(k)
        ic("pp_rec.__init__()")
        self.k = k
        self.weight_threshold = weight_threshold
        self.recursion_threshold = recursion_threshold
        self.user_rec = UserRecommender(self.k)
        self.test_ratings = self.user_rec.test_ratings
        
        
    def recursive_prediction(self, active_user: int, candidate_moive: int, recursion_level: int = 1) -> float:
        """"""
        ic("pp_rec.recursive_prediction()")
        ic(recursion_level)
        
        if recursion_level > self.recursion_threshold:
            ic("Reached Recursion Limit - Using Baseline")
            
            return self.user_rec.predict_rating_user_based_nn_wtd(active_user, candidate_moive) # baseline

        nns = self.user_rec.get_k_nearest_users(Similarities.sim_pearson, self.k, active_user)  # no movie id, doesn't limit to just rated
        
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nns:
            neighbour_id = neighbour["user_id"]
            neighbour_movie_rating = self.user_rec.get_user_movie_rating(neighbour_id, candidate_moive)
            
            if neighbour_movie_rating is not None:
                sim_x_y = self.user_rec.get_user_similarity(Similarities.sim_pearson, active_user, neighbour_id)
                mean_rating_for_neighbour = self.user_rec.get_user_mean_rating(neighbour_id)
                
                #ic(sim_x_y)
                alpha += (neighbour_movie_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(neighbour_id, candidate_moive, recursion_level + 1)
                sim_x_y = self.user_rec.get_user_similarity(Similarities.sim_pearson, active_user, neighbour_id)
                #ic(sim_x_y)
                mean_rating_for_neighbour = self.user_rec.get_user_mean_rating(neighbour_id)
                
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_active_user = self.user_rec.get_user_mean_rating(active_user)
        
        #ic(alpha)
        #ic(beta)
        
        
        if beta == 0.0:
            return mean_rating_for_active_user
        else:
            return mean_rating_for_active_user + (alpha/beta)