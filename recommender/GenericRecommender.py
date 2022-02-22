"""

"""


#from time import sleep
from icecream import ic
from dataset.Dataset import Dataset


class GenericRecommender:
    def __init__(self, k: int, dataset = None) -> None:
        #ic("gen_rec.__init__()")
        
        self.k = k
        self.dataset = dataset
        self.load_dataset()
        self.predictions = []
        

    def load_dataset(self) -> None:
        #ic("gen_load_dataset()")
        
        if self.dataset is None:
            self.dataset = Dataset()
        
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
                
        return rating
    

    def calculate_wtd_avg_rating(self, neighbours: list) -> float:
        # weighted, introduces similarity
        # [{sim, user_id, rating}]
        ##ic("gen_rec.calculate_wtd_avg_rating()")
        
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
                
        return rating
    
    
    def add_new_recommendations(self, new_recommendations: list) -> None:
        """recommendation*s* = list if dicts"""
        #ic("gen_rec.add_new_recommendations()")
        
        self.dataset.add_new_recommendations_to_dataset(new_recommendations)

    
    def add_prediction(self, prediction: dict) -> None:
        """a single dict entry"""
        self.predictions.append(prediction)

