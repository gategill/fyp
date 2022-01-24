"""

"""


#from time import sleep
import random
from icecream import ic
#from dataset.Dataset import Dataset
#import copy
#from recommender.ItemRecommender import ItemRecommender
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

class BootstrapRecommender(UserRecommender):
    def __init__(self, k: int, iterations: int = 5, additions: int = 10) -> None:
        ic("bs_rec.__init__()")
        
        super().__init__(k)
        
        self.iterations = iterations
        self.additions = additions
        
        
    def enrich(self) -> None:
        ic("bs_rec.enhance()")
        
        #= copy.deepcopy(self.user_r.user_training_ratings)
        #added_due_to_enrichment = {}
        
        for iteration in range(1, self.iterations + 1):
            print("\n")
            ic(iteration)

            new_recommendations = []
        
            for user_id, v in self.user_training_ratings.items():
                #ic(user_id)
                #added_due_to_enrichment.setdefault(user_id, {})

                movies_rated_in_training = set(list(v.keys()))
                
                movies_rated_in_test = set()
                if user_id in self.user_test_ratings:
                    movies_rated_in_test = set(list(self.user_test_ratings[user_id].keys()))
                
                movies_rated = movies_rated_in_training.intersection(movies_rated_in_test)
                
                #ic(movies_I_rated)
                #ic(type(self.user_r.movie_ids[0]))
                movies_unrated = list(set(self.movie_ids).difference(movies_rated))
                #movies_unrated = list(movies_unrated)
                #ic(type(movies_unrated[0]))

                movies_unrated = random.sample(movies_unrated, k = self.additions)
                
                #predicted_rating = random.choice(range(5 + 1), 10)
                for mm in movies_unrated:
                    r =  round(random.uniform(0, 5), 1)
                    new_recommendations.append({"user_id" : user_id , "movie_id" : mm ,"rating" : r})
                    
                    #to add it again, you need to recalculate fields in dataset
                    #add 1, ad3d multple
                    #need to be updated at every iteraition
                    #work out how to add these to datasets and what to update
                    
                    #consider what you're adding, its format and for what? movie or user
                    #consider using __d_to_dlist()
                    
                    #added_due_to_enrichment[user_id][mm] = r
                    
                #ic(user_id)
                #ic(self.user_r.user_training_ratings[user_id])
                #ic(new_recommendations)
                #ic(added_due_to_enrichment)
                
            self.add_new_recommendations(new_recommendations)
                
 
            
                

        
        



            
