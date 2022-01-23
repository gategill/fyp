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
        
        
    def enrich(self, iterations: int = 5) -> None:
        ic("bs_rec.enhance()")
        
        enriched_ratings = copy.deepcopy(self.user_r.user_training_ratings)
        #added_due_to_enrichment = {}
        
        for iteration in range(iterations):
            ic(iteration)
        
            for user_id, v in enriched_ratings.items():
                #added_due_to_enrichment.setdefault(user_id, {})

                movies_I_rated_training = set(list(v.keys()))
                movies_I_rated_test = set(list(enriched_ratings[user_id].keys()))
                movies_I_rated = movies_I_rated_training.intersection(movies_I_rated_test)
                
                #ic(movies_I_rated)
                #ic(type(self.user_r.movie_ids[0]))
                movies_I_didnt_see = set(self.user_r.movie_ids).difference(movies_I_rated)
                movies_I_didnt_see = list(movies_I_didnt_see)
                #ic(type(movies_I_didnt_see[0]))

                movies_I_didnt_see = random.sample(movies_I_didnt_see, k = 10)
                
                #predicted_rating = random.choice(range(5 + 1), 10)
                for mm in movies_I_didnt_see:
                    r =  round(random.uniform(0, 5),1)
                    enriched_ratings[user_id][mm] = r
                    
                    to add it again, you need to recalculate fields in dataset
                    add 1, add multple
                    need to be updated at every iteraition
                    
                    #added_due_to_enrichment[user_id][mm] = r
                    
                #ic(user_id)
                #ic(self.user_r.user_training_ratings[user_id])
                #ic(enriched_ratings[user_id])
                #ic(added_due_to_enrichment)
                
                #break
            #break
                

        
        



            
