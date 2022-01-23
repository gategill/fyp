"""

"""


from time import sleep
from icecream import ic
from dataset.Dataset import Dataset
import copy
from recommender.ItemRecommender import ItemRecommender
from recommender.UserRecommender import UserRecommender

# modle like mechanism to bootstrap
# Introduce copies for own datasets
# given iterations ite, 
# add data to augenmted dataset (Dataset)
# make sure not in training?

# look at ratings.txt
# get new recommendation 
# find 5 nearest neighoubs
# get score  
# ie add new movie rating to rating not in testing and not in training
# add new reccomendation to dataset

class BootstrapRecommender:
    def __init__(self, k: int) -> None:
        ic("bs_rec.__init__()")
        self.mode = "USER"
        self.k = k
        self.user_r = UserRecommender(k)
        self.enriched_ratings = copy.deepcopy(self.user_r.user_training_ratings)
        
    def enrich(self, iterations = 5):
        added_due_to_enrichment = []
        for i in range(iterations):
            ic("iteration: {}".format(i))
        
            for user_id, v  in self.enriched_ratings.items():
                #user_id = test['user_id']
                movies_I_rated = v.keys() # list
                
                th = []
                
                
                
                # do it randomly
                # get 10 movie_id I didn't watch and that are not in test
                # assign random rating
                # add to enriched_rating
                
                #rating = v['rating']
         
        
        



            
