"""

"""


#from time import sleep
from icecream import ic
from dataset.Dataset import Dataset
from recommender.GenericRecommender import GenericRecommender
from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender
from recommender.Similarities import Similarities
import random
import copy


class CoRecRecommender(GenericRecommender):
    def __init__(self, k: int, additions: int, top_m: int, dataset = None) -> None:
        ic("cr_rec.__init__()")
        
        super().__init__(k, dataset)
        
        self.additions = additions
        self.top_m = top_m
        
        #self.user_rec = UserRecommender(self.k, self.dataset)        
        #self.item_rec = ItemRecommender(self.k, self.dataset)        
        
        
    def co_rec_algorithm(self, X, M):
        items_unrated = {}
        # step 1
        for user_id in self.user_training_ratings.keys():
            items_unrated[user_id] = self.get_user_unrated_items(user_id, self.additions)
        
        # steps 2
        training_labelled_user = copy.deepcopy(self.user_training_ratings)
        training_labelled_item = copy.deepcopy(self.item_training_ratings)
        
        training_unlabelled_user = copy.deepcopy(items_unrated)
        training_unlabelled_item = copy.deepcopy(items_unrated) # need to switch
        
        while (not training_unlabelled_user) and (not training_unlabelled_item):     
               
            pass
        

    def enrich(self) -> None:
        
        new_recommendations = []
    
        for user_id in self.user_training_ratings.keys():
            items_unrated = self.get_user_unrated_items(user_id, self.additions)

            for mm in items_unrated:
                predicted_rating = self.predict_rating_user_based_nn_wtd(user_id, int(mm))
                r = round(predicted_rating, 2)
                new_recommendations.append({"user_id" : int(user_id) , "item_id" : int(mm) ,"rating" : float(r)})
            
        self.add_new_recommendations(new_recommendations)
            
            
    def get_user_unrated_items(self, user_id: int,  number: int) -> list:
        """"""
        ic("bs_rec.get_user_unrated_items()")
        
        value = self.user_training_ratings[user_id]
        items_rated_in_training = set(list(value.keys()))
        items_rated_in_test = set()
        
        if user_id in self.user_test_ratings:
            items_rated_in_test = set(list(self.user_test_ratings[user_id].keys()))
        
        items_rated = items_rated_in_training.intersection(items_rated_in_test)
        items_unrated = list(set(self.item_ids).difference(items_rated))
        items_unrated = random.sample(items_unrated, k = number)
        
        return items_unrated
        
                
    def get_confidence_measure(self, algorithm, user_id, item_id, prediction): 
        trustworthiness = self.get_measure_of_trustworthiness(algorithm, user_id, item_id, prediction)
        Nu = self.get_Nu(user_id)
        Ni = self.get_Ni(item_id)
        
        return trustworthiness*Nu*Ni
        
        
    def get_measure_of_trustworthiness(self, algorithm, user_id, item_id, prediction): 
        baseline_estimate =  self.get_baseline_estimate(user_id, item_id)
        
        return abs(1/(baseline_estimate - prediction))
        
        
    def get_baseline_estimate(self, user_id, item_id):
        return 3.0
    
    
    def get_Nu(self, user_id):
        return 10
    
    
    def get_Ni(self, item_id):
        return 10
    
