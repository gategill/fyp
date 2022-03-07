"""

"""


from icecream import ic
from recommender.GenericRecommender import GenericRecommender


class MostPopRecommender(GenericRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.top_m = kwargs["run_params"]["top_m"]

    
        
    def get_single_prediction(self, candidate_item_id):
        # get k most popular items
        nns = self.get_m_most_popular_items(self.top_m)
        prediction = self.calculate_avg_rating(nns)
        
        if prediction:
            if prediction < 1.0:
                prediction = 1.0
                
            if prediction > 5:
                prediction = 5.0
                
            return round(prediction, self.ROUNDING)

        else:
            prediction = self.get_item_mean_rating(candidate_item_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_train_rating
    
    
    def predict(self):
        prediction = self.get_single_prediction(candidate_item_id = item_id)
        for i, test in enumerate(self.test_ratings):
            try:
                item_id = int(test['item_id'])
                
                    
                test["pred_rating"] = prediction
                self.add_prediction(test)
                
                if self.kwargs["experiment_config"]["early_stop"]:
                    if i > 30:
                        break
                        
            except KeyboardInterrupt:
                break
            
        return test

   