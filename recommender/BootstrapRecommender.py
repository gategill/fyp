"""

"""


import random
from icecream import ic
from recommender.UserRecommender import UserRecommender


class BootstrapRecommender(UserRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("bs_rec.__init__()")
        
        super().__init__(dataset, **kwargs)
        self.fold_nums = kwargs["run_params"]["fold_nums"]
        self.additions = kwargs["run_params"]["additions"]
        
        
    def enrich(self) -> None:
        #ic("bs_rec.enhance()")
        
        for fold_num in range(1, self.fold_nums + 1):
            print("\n")
            ic(fold_num)

            new_recommendations = []
        
            for user_id in self.user_train_ratings.keys():
                items_unrated = self.get_user_unrated_items(user_id, self.additions)

                for mm in items_unrated:
                    predicted_rating = self.predict_rating_user_based_nn_wtd(user_id, int(mm))
                    
                    if predicted_rating < 1.0:
                        predicted_rating = 1.0
                        
                    if predicted_rating > 5:
                        predicted_rating = 5.0
     
                    r = round(predicted_rating, 1)
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
        
                
 
            
                

        
        



            
