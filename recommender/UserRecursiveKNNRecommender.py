from recommender.UserKNNRecommender import UserKNNRecommender
from icecream import ic

class UserRecursiveKNNRecommender(UserKNNRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        #ic("pp_rec.__init__()")

        super().__init__(dataset, **kwargs)
        self.weight_threshold = kwargs["run_params"]["weight_threshold"]
        self.recursion_threshold = kwargs["run_params"]["recursion_threshold"]
        self.phi = kwargs["run_params"]["phi"]
        self.k_prime = self.k 
        self.neighbour_selection = kwargs["run_params"]["neighbour_selection"]
        self.hashed_predictions = {}
    
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        return self.recursive_prediction(active_user_id, candidate_item_id)
      
    def recursive_prediction(self, active_user: int, candidate_item: int, recursion_level: int = 0) -> float:
        """"""
        #ic("pp_rec.recursive_prediction()")
        active_user = int(active_user)
        candidate_item = int(candidate_item)
        # starts at 0
        if recursion_level > self.recursion_threshold:
            hashkey_key = str(active_user) + "-"+ str(candidate_item)
            if hashkey_key in self.hashed_predictions:
                return self.hashed_predictions[hashkey_key]
            else:
                pr = self.baseline_predictor(active_user, candidate_item)
                self.hashed_predictions[hashkey_key] = pr
                return pr

        # no item id, doesn't limit to just rated
        nns = self.nearest_neighbour_seletion(active_user, candidate_item)
        
        alpha = 0.0
        beta = 0.0
        
        for neighbour in nns:
            neighbour_id = int(neighbour["user_id"])
            neighbour_item_rating = self.is_it_rated(user_ID = neighbour_id, item_ID = candidate_item)
            
            if neighbour_item_rating is not None:
                sim_x_y = self.get_user_similarity(self.similarity_function, active_user, neighbour_id)
                mean_rating_for_neighbour = self.get_user_mean_rating(neighbour_id)
                
                alpha += (neighbour_item_rating - mean_rating_for_neighbour) * sim_x_y
                beta += abs(sim_x_y)
                
            else:
                rec_pred = self.recursive_prediction(neighbour_id, candidate_item, recursion_level + 1)
                  
                hashkey_key = str(neighbour_id) + "-"+ str(candidate_item)
                self.hashed_predictions[hashkey_key] = rec_pred
                
                sim_x_y = self.get_user_similarity(self.similarity_function, active_user, neighbour_id)
                mean_rating_for_neighbour = self.get_user_mean_rating(neighbour_id)
                
                alpha += self.weight_threshold * (rec_pred - mean_rating_for_neighbour) * sim_x_y
                beta += self.weight_threshold * abs(sim_x_y)
        
        mean_rating_for_active_user = self.get_user_mean_rating(active_user)
        if beta == 0.0:
            
            return mean_rating_for_active_user
        else:
            prediction = mean_rating_for_active_user + (alpha/beta)
            
            if prediction < 1.0:
                prediction = 1.0
                
            if prediction > 5:
                prediction = 5.0
    
            return round(prediction, self.ROUNDING)
       
       
    def is_it_rated(self, user_ID, item_ID):  
        # Check if the rating exists in self.train, then in the intermediate calculations, else return None      
        if self.get_user_item_rating(user_id = user_ID, item_id = item_ID) is not None:
            return self.get_user_item_rating(user_id = user_ID, item_id = item_ID)
         
        else:
            hashkey_key = str(user_ID) + "-"+ str(item_ID)
            if hashkey_key in self.hashed_predictions:
                return self.hashed_predictions[hashkey_key]
            
            else:
                return None 
        
        
    def nearest_neighbour_seletion(self, active_user, candidate_item):
        if self.neighbour_selection == "bs":
            nns = self.get_k_nearest_users(self.similarity_function, self.k, active_user, candidate_item)
                        
        elif self.neighbour_selection == "bs+":
            nns = self.get_k_nearest_users_with_overlap(self.similarity_function, self.k, active_user, candidate_item, self.phi)

        elif self.neighbour_selection == "ss":
            nns = self.get_k_nearest_users(self.similarity_function, self.k_prime, active_user, None)         
                
        elif self.neighbour_selection == "cs":
            nns1 = self.get_k_nearest_users(self.similarity_function, self.k,active_user, candidate_item)
            nns2 = self.get_k_nearest_users(self.similarity_function, self.k_prime,  active_user, None)
            nns = nns1 + nns2
          
        elif self.neighbour_selection == "cs+":
            nns1 = self.get_k_nearest_users_with_overlap(self.similarity_function, self.k, active_user, candidate_item, self.phi)
            nns2 = self.get_k_nearest_users_with_overlap(self.similarity_function, self.k_prime, active_user, self.phi)
            nns = nns1 + nns2
            
        else:
            raise KeyError("Invalid neighbour_selection strategy")
        
        return nns
          
          
    def baseline_predictor(self, active_user, candidate_item):
        # baseline predictor = BS
        nns = self.get_k_nearest_users(self.similarity_function, self.k, active_user, candidate_item)
        prediction = self.calculate_wtd_avg_rating(nns)
        
        if prediction:  
            return prediction
        else:
            prediction = self.get_user_mean_rating(active_user)
            
            if prediction:
                return prediction
            else:
                return self.mean_train_rating      