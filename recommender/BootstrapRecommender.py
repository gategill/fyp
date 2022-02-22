"""

"""


#from time import sleep
import random
from icecream import ic
#from dataset.Dataset import Dataset
#import copy
#from recommender.ItemRecommender import ItemRecommender
from recommender.UserRecommender import UserRecommender


class BootstrapRecommender(UserRecommender):
    def __init__(self, k: int, iterations: int = 5, additions: int = 10, dataset = None) -> None:
        #ic("bs_rec.__init__()")
        
        super().__init__(k, dataset)
        self.iterations = iterations
        self.additions = additions
        
        
    def enrich(self) -> None:
        #ic("bs_rec.enhance()")
        
        for iteration in range(1, self.iterations + 1):
            print("\n")
            ic(iteration)

            new_recommendations = []
        
            for user_id in self.user_train_ratings.keys():
                items_unrated = self.get_user_unrated_items(user_id, self.additions)

                for mm in items_unrated:
                    predicted_rating = self.predict_rating_user_based_nn_wtd(user_id, int(mm))
                    
                    if predicted_rating < 1.0:
                        predicted_rating = 1.0
                        
                    if predicted_rating > 5:
                        predicted_rating = 5.0
     
                    r = round(predicted_rating, 2)
                    new_recommendations.append({"user_id" : int(user_id) , "item_id" : int(mm) ,"rating" : float(r)})
                
            self.add_new_recommendations(new_recommendations)
            
            
    def get_user_unrated_items(self, user_id: int,  number: int) -> list:
        """"""
        #ic("bs_rec.get_user_unrated_items()")
        
        value = self.user_train_ratings[user_id]
        items_rated_in_train = set(list(value.keys()))
        items_rated_in_test = set()
        
        if user_id in self.user_test_ratings:
            items_rated_in_test = set(list(self.user_test_ratings[user_id].keys()))
        
        items_rated = items_rated_in_train.intersection(items_rated_in_test)
        items_unrated = list(set(self.item_ids).difference(items_rated))
        items_unrated = random.sample(items_unrated, k = number)
        
        return items_unrated
        
                
 
            
                

        
        



            
