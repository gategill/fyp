"""

"""


#from time import sleep
from icecream import ic
#from dataset.Dataset import Dataset
from recommender.UserRecommender import UserRecommender

class PearlPuRecommender:
    def __init__(self, k: int = 5,  weight_threshold: float = 0.75, recursion_threshold: int = 3) -> None:
        ic("pp_rec.__init__()")
        self.k = k
        self.weight_threshold = weight_threshold
        self.recursion_threshold = recursion_threshold
        
        self.test_rating = self.get_test_rating()
        
    def recursive_prediction(self, active_user: int, candidate_moive: int, level: int = 2) -> float:
        """"""
        ic("pp.rec.recursive_prediction()")
        ic(level)
        
        if level > self.recursion_threshold:
            return self.get_baseline_predictor()
        
        nearest_neighbours = self.select_neighbour(active_user, candidate_moive)
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nearest_neighbours:
            shared_movie_rating = self.get_shared_movie_rating(neighbour, candidate_moive)
            
            if shared_movie_rating is not None:
                sim_x_y = self.get_similarity_between(active_user, neighbour)
                mean_rating_for_neighbour = self.get_mean_rating_for_user(neighbour)
                
                alpha += (shared_movie_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(active_user, candidate_moive, level + 1)
                sim_x_y = self.get_similarity_between(active_user, neighbour)
                mean_rating_for_neighbour = self.get_mean_rating_for_user(neighbour)
                    
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_active_user = self.get_mean_rating_for_user(active_user)

        return mean_rating_for_active_user + (alpha/beta)
                
                
         
    def get_baseline_predictor(self) -> float:
        """"""
        ic("pp_rec.get_baseline_predictor()")
        
        return None
    
    
    def select_neighbour(self, active_user: int, candidate_moive: int) -> int:
        """"""
        ic("pp_rec.select_neighbour()")
        
        return None


    def get_shared_movie_rating(self, neighbour: int, caldidate_movie: int) -> float:
        """"""
        ic("pp_rec.get_shared_movie_rating()")
        
        return None
    
    
    def get_similarity_between(self, active_user: int, neighbour: int) -> float:
        """"""
        ic("pp_rec.get_similarity_between()")
        
        return None
        

    def get_mean_rating_for_user(self, neighbour: int) -> float:
        """"""
        ic("pp_rec.get_mean_rating_for_user()")
        
        return None
        
        
    def get_test_rating(self) -> list:
        """"""
        ic("pp_rec.get_test_rating()")
        
        return None
