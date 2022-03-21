"""

"""


import random
from icecream import ic
from recommender.ItemKNNRecommender import ItemKNNRecommender


class ItemBootstrapRecommender(ItemKNNRecommender):
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
            print("enrichment round {}\n".format(enrichment_round + 1))

            new_recommendations = []
        
            for item_id in self.item_train_ratings.keys(): # does this even make sense?
                users_unrated = self.get_item_unrated_users(item_id, self.additions)

                for user_id in users_unrated:
                    predicted_rating = self.get_single_prediction(user_id, item_id)     
                    new_recommendations.append({"user_id" : int(user_id) , "item_id" : int(item_id) ,"rating" : float(predicted_rating)})
                
            self.add_new_recommendations(new_recommendations)
            
            
    def get_item_unrated_users(self, item_id: int,  number: int) -> list:
        """"""
        #ic("bs_rec.get_user_unrated_items()")
        
        value = self.item_train_ratings[item_id]
        users_rated_in_train = set(list(value.keys()))
        users_rated_in_test = set()
        
        if item_id in self.item_test_ratings:
            users_rated_in_test = set(list(self.item_test_ratings[item_id].keys()))
        
        users_rated = users_rated_in_train.intersection(users_rated_in_test)
        users_unrated = list(set(self.user_ids).difference(users_rated))
        users_unrated = random.sample(users_unrated, number)
        
        assert(users_unrated[0] not in users_rated)
        
        return users_unrated
        
                
 
            
                

        
        



            
