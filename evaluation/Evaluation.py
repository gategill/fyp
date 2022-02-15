"""

"""


from scipy.fftpack import ifft


class Evaluation:
    def __init__(self):
        pass

        
    @staticmethod
    def mean_absolute_error(predictions: list) -> float:
        if len(predictions) == 0:
            raise TypeError("mean_absolute_error: you supplied an empty prediction list")
 
        if "pred_rating" in predictions[0]:
            mae = 0
            for prediction in predictions:
                mae += abs(prediction["pred_rating"] - prediction["rating"])
                
            mae = round(mae/len(predictions), 5)
            
            return mae

        if ("user_pred_rating" in predictions[0]) and ("item_pred_rating" in predictions[0]):
            mae_user = 0
            mae_item = 0
            for prediction in predictions:
                mae_user += abs(prediction["user_pred_rating"] - prediction["rating"])
                mae_item += abs(prediction["item_pred_rating"] - prediction["rating"])
                
            mae_user = round(mae_user/len(predictions), 5)
            mae_item = round(mae_item/len(predictions), 5)
            
            return mae_user, mae_item
            
        raise KeyError("mean_absolute_error: you supplied a prediction list without predicted ratings")
            
        