"""

"""


#from time import sleep
from icecream import ic
#from dataset.Dataset import Dataset
from recommender.UserRecommender import UserRecommender
from recommender.Similarities import Similarities


class PearlPuRecommender:
    def __init__(self, k: int = 5,  weight_threshold: float = 0.75, recursion_threshold: int = 3) -> None:
        ic("pp_rec.__init__()")
        self.k = k
        self.weight_threshold = weight_threshold
        self.recursion_threshold = recursion_threshold
        self.user_rec = UserRecommender(self.k)
        self.test_rating = self.user_rec.test_rating
        
        
    def recursive_prediction(self, active_user: int, candidate_moive: int, level: int = 2) -> float:
        """"""
        ic("pp.rec.recursive_prediction()")
        ic(level)
        
        if level > self.recursion_threshold:
            return self.get_baseline_prediction(active_user, candidate_moive)
        
        nearest_neighbours = self.select_neighbour(active_user) # no movie id, doesn't limit to just rated
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nearest_neighbours:
            shared_movie_rating = self.get_shared_movie_rating(neighbour, candidate_moive)
            
            if shared_movie_rating is not None:
                sim_x_y = self.get_similarity_between(active_user, neighbour)
                mean_rating_for_neighbour = self.user_rec.get_user_mean_rating(neighbour)
                
                alpha += (shared_movie_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(active_user, candidate_moive, level + 1)
                sim_x_y = self.get_similarity_between(active_user, neighbour)
                mean_rating_for_neighbour = self.user_rec.get_user_mean_rating(neighbour)
                    
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_active_user = self.user_rec.get_user_mean_rating(neighbour)

        return mean_rating_for_active_user + (alpha/beta)
                
                
         
    def get_baseline_prediction(self, active_user: int, candidate_moive: int) -> float:
        """"""
        ic("pp_rec.get_baseline_prediction()")

        baseline_prediction = self.user_rec.predict_rating_user_based_nn_wtd(active_user, candidate_moive)
        ic(baseline_prediction)
        
        return baseline_prediction
    
    
    def select_neighbour(self, active_user: int) -> list:
        """"""
        ic("pp_rec.select_neighbour()")
        
        nns = self.user_rec.get_k_nearest_users(Similarities.sim_pearson, self.k, active_user)
        ic(nns)
        
        return nns


    def get_shared_movie_rating(self, neighbour: int, caldidate_movie: int) -> float:
        """"""
        ic("pp_rec.get_shared_movie_rating()")
        
        if 1:
            return shared_movie_rating
        else:
            return None
    
    
    def get_similarity_between(self, active_user: int, neighbour: int) -> float:
        """"""
        ic("pp_rec.get_similarity_between()")
        
        return None
        

        