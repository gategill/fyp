#import random
#from tkinter.messagebox import NO
import types
from time import sleep
from icecream import ic

import Similarities as sm
import Datasets as ds


class GenericRecommender:
    def __init__(self):
        ic("gen_rec.__init__()")
        
        dataset = ds.Datasets()
        self.user_training_ratings = dataset.get_user_training_ratings()
        self.user_training_means = dataset.get_user_training_means()
        self.movie_training_ratings = dataset.get_movie_training_ratings()
        self.movie_training_means = dataset.get_movie_training_means()
        self.test_ratings = dataset.get_test_ratings()


    def calculate_avg_rating(self, neighbours): # GENERIC
        ic("gen_rec.calculate_avg_rating()")
        
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


    def calculate_wtd_avg_rating(self, neighbours): # weighted, introduces similarity # GENERIC
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
    

            
