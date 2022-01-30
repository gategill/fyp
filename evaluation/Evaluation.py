"""

"""


class Evaluation:
    def __init__(self):
        pass
        #self.training_data = training_data
        
    @staticmethod
    def mean_absolute_error(predictions: list) -> float:
        if len(predictions) == 0:
            raise TypeError("mean_absolute_error: you supplied an empty prediction list")
 
        mae = 0

        for prediction in predictions:
            mae += abs(prediction["pred_rating"] - prediction["rating"])
            
        return mae/len(predictions)
        