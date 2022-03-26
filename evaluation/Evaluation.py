"""

"""


import numpy as np

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
                try:
                    mae += abs(prediction["pred_rating"] - prediction["rating"])
                
                except TypeError as e:
                    print("prediction is: {}".format(prediction))
                    #print("rating is: {}".format(prediction["rating"]))
                    raise TypeError("Error is: {}".format(e))
                    
                    
                
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
    
    
    @staticmethod
    def root_mean_square_error(predictions: list) -> float:
        if len(predictions) == 0:
            raise TypeError("root_mean_square_error: you supplied an empty prediction list")
 
        if "pred_rating" in predictions[0]:
            se = 0.0
            
            for prediction in predictions:
                se += (prediction["pred_rating"] - prediction["rating"])**2
                
            mse = se/len(predictions)
            rmse = np.sqrt(mse)
                
            return round(rmse, 5)
            
        if ("user_pred_rating" in predictions[0]) and ("item_pred_rating" in predictions[0]):
            se_user = 0.0
            se_item = 0.0
            
            for prediction in predictions:
                se_user += (prediction["user_pred_rating"] - prediction["rating"])**2
                se_item += (prediction["item_pred_rating"] - prediction["rating"])**2

            mse_user = se_user/len(predictions)
            mse_item = se_item/len(predictions)
                
            rmse_user = np.sqrt(mse_user)
            rmse_item = np.sqrt(mse_item)
            
            return round(rmse_user, 5), round(rmse_item, 5)

        raise KeyError("root_mean_square_error: you supplied a prediction list without predicted ratings")
            
        