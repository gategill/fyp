"""

"""


#from time import sleep
from icecream import ic
from dataset.Dataset import Dataset
from recommender.GenericRecommender import GenericRecommender
from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender
from recommender.Similarities import Similarities


class CoRecRecommender(GenericRecommender):
    def __init__(self, k: int, dataset = None) -> None:
        ic("cr_rec.__init__()")
        
        super().__init__(k, dataset)
        
        self.user_rec = UserRecommender(self.k, self.dataset)        
        self.item_rec = ItemRecommender(self.k, self.dataset)        
        
    def co_rec_algorithm(self):
        pass
        
        
