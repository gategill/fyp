"""

"""


import random
from icecream import ic
from recommender.UserKNNRecommender import UserKNNRecommender


class ConfidentUserBootstrapRecommender(UserKNNRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("bs_rec.__init__()")
        
        super().__init__(dataset, **kwargs)
        self.enrichments = kwargs["run_params"]["enrichments"]
        self.additions = kwargs["run_params"]["additions"]
        
        
    def train(self):
        self.load_dataset(**self.kwargs["dataset_config"])
        
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        return 0
        
                
 
            
                

        
        



            
