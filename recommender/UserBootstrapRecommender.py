"""

"""


import random
from icecream import ic
from recommender.UserRecommender import UserRecommender


class UserBootstrapRecommender(UserRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("bs_rec.__init__()")
        
        super().__init__(dataset, **kwargs)
        self.enrichments = kwargs["run_params"]["enrichments"]
        self.additions = kwargs["run_params"]["additions"]
        
        
    def train(self):
        self.load_dataset(**self.kwargs["dataset_config"])
        self.enrich()
        
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        return super().get_single_prediction(active_user_id, candidate_item_id)
    
        
    def enrich(self) -> None:
        #ic("bs_rec.enhance()")
        
        for enrichment_round in range(self.enrichments):
            print("\n")

            new_recommendations = []
        
            for user_id in self.user_train_ratings.keys():
                items_unrated = self.get_user_unrated_items(user_id, self.additions)

                for item_id in items_unrated:
                    predicted_rating = self.get_single_prediction(user_id, item_id)     
                    new_recommendations.append({"user_id" : int(user_id) , "item_id" : int(item_id) ,"rating" : float(predicted_rating)})
                
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
        items_unrated = random.sample(items_unrated, number)
        
        assert(items_unrated[0] not in items_rated)
        
        return items_unrated
        
                
 
            
                

        
        



            
