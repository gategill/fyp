"""

"""


from time import sleep
from icecream import ic
from dataset.Dataset import Dataset
import copy
from recommender.ItemRecommender import ItemRecommender
from recommender.UserRecommender import UserRecommender
import random

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
        
        
    def enrich(self, iterations = 5):
        enriched_ratings = copy.deepcopy(self.user_r.user_training_ratings)
        added_due_to_enrichment = {}
        
        for i in range(iterations):
            ic("iteration: {}".format(i))
        
            for user_id, v in enriched_ratings.items():
                added_due_to_enrichment.setdefault(user_id, {})

                movies_I_rated_training = set(list(v.keys()))
                movies_I_rated_test = set(list(enriched_ratings[user_id].keys()))
                movies_I_rated = movies_I_rated_training.intersection(movies_I_rated_test)
                
                movies_I_didnt_see = set(self.user_r.movie_ids).difference(movies_I_rated)
                movies_I_didnt_see = list(movies_I_didnt_see)
                movies_I_didnt_see = random.sample(movies_I_didnt_see, k = 10)
                
                #predicted_rating = random.choice(range(5 + 1), 10)
                for mm in movies_I_didnt_see:
                    enriched_ratings[user_id][mm] = random.choice(range(0, 5 + 1))
                
                ic(enriched_ratings)
                ic(added_due_to_enrichment)
                
                break
                
                
            
                # assign random rating
                # add to enriched_rating
                
                #rating = v['rating']
         
        
        



            
