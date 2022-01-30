"""

"""


#from time import sleep
from icecream import ic
from dataset.Dataset import Dataset


class GenericRecommender:
    def __init__(self, k: int) -> None:
        ic("gen_rec.__init__()")
        
        self.k = k
        
        self.dataset = Dataset()
        self.load_dataset(self.dataset)
        

    def load_dataset(self, dataset) -> None:
        ic("gen_load_dataset()")
        
        #dataset = Dataset()
        
        self.user_ids = dataset.get_user_ids()
        self.movie_ids = dataset.get_movie_ids()
        
        self.user_training_ratings = dataset.get_user_training_ratings()
        self.user_training_means = dataset.get_user_training_means()
        self.movie_training_ratings = dataset.get_movie_training_ratings()
        self.movie_training_means = dataset.get_movie_training_means()
        
        self.user_test_ratings = dataset.get_user_test_ratings()
        self.movie_test_ratings = dataset.get_movie_test_ratings()
        self.test_ratings = dataset.get_test_ratings()
        
        self.mean_training_rating = dataset.get_mean_training_rating()
                

    def calculate_avg_rating(self, neighbours: list) -> float:
        ic("gen_rec.calculate_avg_rating()")
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
        
        return numerator / denominator


    def calculate_wtd_avg_rating(self, neighbours: list) -> float:
        # weighted, introduces similarity
        # [{sim, user_id, rating}]
        ic("gen_rec.calculate_wtd_avg_rating()")
        
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
        
        return numerator / denominator
    
    def add_new_recommendations(self, new_recommendations) -> None:
        # 
        ic("gen_rec.add_new_recommendations()")
        
        self.dataset.add_new_recommendations_to_dataset(new_recommendations)