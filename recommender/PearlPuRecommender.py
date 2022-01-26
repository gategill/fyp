"""

"""


#from time import sleep
from icecream import ic
#from dataset.Dataset import Dataset
from recommender.UserRecommender import UserRecommender

class PearlPuRecommender:
    def __init__(self, k: int, recursion_threshold: int, weight_threshold: float) -> None:
        ic("pp_rec.__init__()")
        self.k = k
        self.weight_threshold = weight_threshold
        
        self.recursion_threshold = recursion_threshold
        
    def recursive_prediction(self, active_user, candidate_moive, level):
        """"""
        ic("pp.rec.recursive_prediction()")
        
        if level > self.recursion_threshold:
            return self.get_baseline_predictor()
        
        nearest_neighbours = self.select_neighbour(active_user, candidate_moive)
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nearest_neighbours:
            shared_movie_rating = self.get_shared_movie_rating(neighbour, candidate_moive)
            if shared_movie_rating is not None:
                
                
         
    def get_baseline_predictor(self):
        ic("pp_rec.get_baseline_predictor()")
        pass
    
    
    def select_neighbour(self):
        ic("pp_rec.select_neighbour()")
        pass


    def get_shared_movie_rating(self, neighbour, caldidate_movie):
        

            
