"""

"""


class Evaluation:
    def __init__(self):
        pass
        #self.training_data = training_data
        
    @staticmethod
    def mean_absolute_error(predictions):
        mae = 0

        for prediction in predictions:
            mae += abs(prediction["pred_rating"] - prediction["rating"])
            
        return mae/len(predictions)
        