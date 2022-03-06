"""

"""


from icecream import ic
from recommender.GenericRecommender import GenericRecommender


class MeanRecommender(GenericRecommender): # TODO
    def __init__(self, dataset = None, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.which = kwargs["run_params"]["which"]
        
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        if self.which == "global":
            active_user_id = self.mean_train_rating
        
        elif self.which == "user":
            active_user_id = self.user_train_means[active_user_id]
        elif self.which == "item":
            active_user_id = self.item_train_means[candidate_item_id]
        else:
            raise KeyError("invalid mean type")
            
        if prediction < 1.0:
            prediction = 1.0
            
        if prediction > 5:
            prediction = 5.0
            
        return round(prediction, self.ROUNDING)


    def get_predictions(self):
        for i, test in enumerate(self.test_ratings):
            try:
                user_id = int(test['user_id'])
                item_id = int(test['item_id'])
                prediction = self.get_single_prediction(active_user_id = user_id, candidate_item_id = item_id)
                
                test["pred_rating"] = prediction
                self.add_prediction(test)
                
                if self.kwargs["testing_strategy"]["early_stop"]:
                    if i > 30:
                        break
                        
            except KeyboardInterrupt:
                break
            
        return test

   