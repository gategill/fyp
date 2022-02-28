   
"""

"""


from icecream import ic
import numpy as np
from recommender.GenericRecommender import GenericRecommender


class RandomRecommender(GenericRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:

        super().__init__(dataset, **kwargs)
    
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        return  np.random.uniform(1.0, 5.0)


