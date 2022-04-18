from recommender.GenericRecommender import GenericRecommender
from icecream import ic
import types


class UserKNNRecommender(GenericRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        ic("user_rec.__init__()")
        super().__init__(dataset, **kwargs)


    def get_single_prediction(self, active_user_id, candidate_item_id):
        prediction =  self.predict_rating_user_based_nn_wtd(int(active_user_id), int(candidate_item_id))
    
        if prediction < 1.0:
            prediction = 1.0
    
        if prediction > 5:
            prediction = 5.0
            
        prediction = round(prediction, self.ROUNDING)
        return prediction
        
        
    def predict_rating_user_based_nn(self, active_user_id: int, candidate_item_id: int) -> float:
        #ic("user_rec.predict_rating_user_based_nn()")
        
        nns = self.get_k_nearest_users(self.similarity_function, self.k, active_user_id, candidate_item_id)
        prediction = self.calculate_avg_rating(nns)
        
        if prediction:    
            return prediction
        
        else:
            prediction = self.get_user_mean_rating(active_user_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_train_rating


    def predict_rating_user_based_nn_wtd(self, active_user_id: int, candidate_item_id: int) -> float:
        #ic("user_rec.predict_rating_user_based_nn_wtd()")

        nns = self.get_k_nearest_users(self.similarity_function, self.k, int(active_user_id), int(candidate_item_id))
        prediction = self.calculate_wtd_avg_rating(nns)
        
        if prediction:  
            return prediction
        else:
            prediction = self.get_user_mean_rating(int(active_user_id))
            
            if prediction:
                return prediction
            else:
                return self.mean_train_rating


    def get_k_nearest_users(self, similarity_function: types.FunctionType, k: int, active_user_id: int, candidate_item_id: int = None) -> list:
        #ic("user_rec.get_k_nearest_users()")
        
        """
        [{user_id: int, rating: float, sim: float}]

        Get the k nearest users to active_user_id.
        Optionally, if candidate_item_id is specifies, the set of neighbours (users) is confined to those who have 
        rated candidate_item_id.
        In this case, each neighbour's rating for candidate_item_id is part of the final result.
        
        THIS FOR PEARL PU!!!
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_nearest_users: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_nearest_users: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.user_train_ratings):
            raise ValueError("get_k_nearest_users: you supplied k = %i but this is too large" % k)
        if type(active_user_id) != int or active_user_id < 0:
            raise TypeError("get_k_nearest_users: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
        if active_user_id not in self.user_train_ratings:
            raise ValueError("get_k_nearest_users: you supplied active_user_id = %i but this user does not exist" % active_user_id)
        if candidate_item_id:
            if type(candidate_item_id) != int or candidate_item_id < 0:
                raise TypeError("get_k_nearest_users: you supplied candidate_item_id = '%s' but candidate_item_id must be a positive integer" % candidate_item_id)
            if candidate_item_id not in self.item_train_ratings:
                raise ValueError("get_k_nearest_users: you supplied candidate_item_id = %i but this item does not exist" % candidate_item_id)     
        
        nearest_neighbours = []
        
        for user_id in self.user_train_ratings:
            
            if active_user_id == user_id:
                continue
            if (not candidate_item_id is None) and (not candidate_item_id in self.user_train_ratings[user_id]):
                continue
            
            sim = self.get_user_similarity(similarity_function, active_user_id, user_id)
            
            candidate_neighbour = {'user_id': user_id, 'sim': sim}
            
            # if not None = if confined to users with that item ID
            if not candidate_item_id is None:
                # what's your ID? 
                candidate_neighbour['rating'] = self.user_train_ratings[user_id][candidate_item_id]
            
            # update_neighbours()
            nearest_neighbours.append(candidate_neighbour)
            
            # ensure there are at most k neighbours, else remove the most unsimilar
            if len(nearest_neighbours) > k:
                lowest_sim_index = -1
                lowest_sim = float('inf')
                index = 0
                
                for neighbour in nearest_neighbours:
                    
                    if neighbour['sim'] < lowest_sim:
                        lowest_sim_index = index
                        lowest_sim = neighbour['sim']
                        
                    index = index + 1
                nearest_neighbours.pop(lowest_sim_index)
                
        return nearest_neighbours
    
    
    
    
    def get_k_nearest_users_with_overlap(self, similarity_function: types.FunctionType, k: int, active_user_id: int, candidate_item_id: int = None, overlap: int = 0) -> list:
        #ic("user_rec.get_k_nearest_users_with_overlap()")
        
        """
        [{user_id: int, rating: float, sim: float}]

        Get the k nearest users to active_user_id.
        Optionally, if candidate_item_id is specified, the set of neighbours (users) is confined to those who have 
        rated candidate_item_id.
        In this case, each neighbour's rating for candidate_item_id is part of the final result.
        With sharing at least overlap items together 
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_nearest_users_with_overlap: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if (type(overlap) != int) or (overlap < 0):
            raise TypeError("get_k_nearest_users_with_overlap: you supplied overlap = '%s' but overlap must be a integer >= 0" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_nearest_users_with_overlap: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.user_train_ratings):
            raise ValueError("get_k_nearest_users_with_overlap: you supplied k = %i but this is too large" % k)
        if type(active_user_id) != int or active_user_id < 0:
            raise TypeError("get_k_nearest_users_with_overlap: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
        if active_user_id not in self.user_train_ratings:
            raise ValueError("get_k_nearest_users_with_overlap: you supplied active_user_id = %i but this user does not exist" % active_user_id)
        if candidate_item_id:
            if type(candidate_item_id) != int or candidate_item_id < 0:
                raise TypeError("get_k_nearest_users_with_overlap: you supplied candidate_item_id = '%s' but candidate_item_id must be a positive integer" % candidate_item_id)
            if candidate_item_id not in self.item_train_ratings:
                raise ValueError("get_k_nearest_users_with_overlap: you supplied candidate_item_id = %i but this item does not exist" % candidate_item_id)     
        
        nearest_neighbours = []
        
        for user_id in self.user_train_ratings:
            
            if active_user_id == user_id:
                continue
            if (not candidate_item_id is None) and (not candidate_item_id in self.user_train_ratings[user_id]):
                continue
            
            if self.get_num_shared_items(active_user_id, user_id) < overlap:
                continue
            
            sim = self.get_user_similarity(similarity_function, active_user_id, user_id)
            
            candidate_neighbour = {'user_id': user_id, 'sim': sim}
            
            # if not None = if confined to users with that item ID
            if not candidate_item_id is None:
                # what's your ID? 
                candidate_neighbour['rating'] = self.user_train_ratings[user_id][candidate_item_id]
            
            nearest_neighbours.append(candidate_neighbour)
            
            # ensure there are at most k neighbours, else remove the most unsimilar
            if len(nearest_neighbours) > k:
                lowest_sim_index = -1
                lowest_sim = float('inf')
                index = 0
                
                for neighbour in nearest_neighbours:
                    
                    if neighbour['sim'] < lowest_sim:
                        lowest_sim_index = index
                        lowest_sim = neighbour['sim']
                        
                    index = index + 1
                nearest_neighbours.pop(lowest_sim_index)
                
        return nearest_neighbours
    
    
    def get_num_shared_items(self, active_user_id, user_id):
        u1_items = set(self.user_train_ratings[active_user_id].keys())
        u2_items = set(self.user_train_ratings[user_id].keys())             
        shared_items = u1_items.intersection(u2_items)
        return len(shared_items)
        

    def get_thresholded_nearest_users(self, similarity_function: types.FunctionType, threshold: float, active_user_id: int, candidate_item_id: int = None) -> list:
        #ic("user_rec.get_thresholded_nearest_users()")
        
        """
        [{user_id: int, rating: float, sim: float}]

        Get the users who are more than a threshold similar to active_user_id
        Optionally, if item_id is specified, the set of neighbours (users) is confined to those who have rated candidate_item_id.
        In this case, each neighbour's rating for candidate_item_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_thresholded_nearest_users: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(threshold) != float:
            raise TypeError("get_thresholded_nearest_users: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)
        if type(active_user_id) != int or active_user_id < 0:
            raise TypeError("get_thresholded_nearest_users: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
        if active_user_id not in self.user_train_ratings:
            raise ValueError("get_thresholded_nearest_users: you supplied active_user_id = %i but this user does not exist" % active_user_id)
        if candidate_item_id:
            if type(candidate_item_id) != int or candidate_item_id < 0:
                raise TypeError("get_thresholded_nearest_users: you supplied candidate_item_id = '%s' but candidate_item_id must be a positive integer" % candidate_item_id)
            if candidate_item_id not in self.item_train_ratings:
                raise ValueError("get_thresholded_nearest_users: you supplied candidate_item_id = %i but this item does not exist" % candidate_item_id)     
        
        nearest_neighbours = []
        
        for user_id in self.user_train_ratings:
            
            if active_user_id == user_id:
                continue
            
            if (not candidate_item_id is None) and (not candidate_item_id in self.user_train_ratings[user_id]):
                continue
            
            sim = self.get_user_similarity(similarity_function, active_user_id, user_id)
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'user_id': user_id, 'sim': sim}
            
            if not candidate_item_id is None:
                candidate_neighbour['rating'] = self.user_train_ratings[user_id][candidate_item_id]
                
            nearest_neighbours.append(candidate_neighbour)
            
        return nearest_neighbours


    def get_k_thresholded_nearest_users(self, similarity_function: types.FunctionType, k:int, threshold: float, active_user_id: int, candidate_item_id: int = None) -> list:
        #ic("user_rec.get_k_thresholded_nearest_users()")
        
        """
        [{user_id: int, rating: float, sim: float}]

        Get the k nearest users to active_user_id provided their similarity to active_user_id exceeds the threshold.
        Optionally, if item_id is specified, the set of neighbours (users) is confined to those who have rated candidate_item_id.
        In this case, each neighbour's rating for candidate_item_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_thresholded_nearest_users: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_thresholded_nearest_users: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.user_train_ratings):
            raise ValueError("get_k_thresholded_nearest_users: you supplied k = %i but this is too large" % k)
        if type(threshold) != float:
            raise TypeError("get_k_thresholded_nearest_users: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)
        if type(active_user_id) != int or active_user_id < 0:
            raise TypeError("get_k_thresholded_nearest_users: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
        if active_user_id not in self.user_train_ratings:
            raise ValueError("get_k_thresholded_nearest_users: you supplied active_user_id = %i but this user does not exist" % active_user_id)
        if candidate_item_id:
            if type(candidate_item_id) != int or candidate_item_id < 0:
                raise TypeError("get_k_thresholded_nearest_users: you supplied candidate_item_id = '%s' but candidate_item_id must be a positive integer" % candidate_item_id)
            if candidate_item_id not in self.item_train_ratings:
                raise ValueError("get_k_thresholded_nearest_users: you supplied candidate_item_id = %i but this item does not exist" % candidate_item_id)     
        
        nearest_neighbours = []
        
        for user_id in self.user_train_ratings:
            
            if active_user_id == user_id:
                continue
            
            if (not candidate_item_id is None) and (not candidate_item_id in self.user_train_ratings[user_id]):
                continue
            
            sim = self.get_user_similarity(similarity_function, active_user_id, user_id)
            
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'user_id': user_id, 'sim': sim}
            
            if not candidate_item_id is None:
                candidate_neighbour['rating'] = self.user_train_ratings[user_id][candidate_item_id]
                
            nearest_neighbours.append(candidate_neighbour)
            
            if len(nearest_neighbours) > k:
                lowest_sim_index = -1
                lowest_sim = float('inf')
                index = 0
                
                for neighbour in nearest_neighbours:
                    
                    if neighbour['sim'] < lowest_sim:
                        lowest_sim_index = index
                        lowest_sim = neighbour['sim']
                        
                    index = index + 1
                    
                nearest_neighbours.pop(lowest_sim_index)
                
        return nearest_neighbours


    def get_user_item_rating(self, user_id: int, item_id: int) -> float:
        #ic("user_rec.get_user_item_rating()")
        
        """
        Gets user_id's rating for item_id from the train set or None if this user has no rating for this item in the
        train set.
        """
        
        if type(user_id) != int or user_id < 0:
            raise TypeError("get_user_item_rating: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_train_ratings:
            raise ValueError("get_user_item_rating: you supplied user_id = %i but this user does not exist" % user_id)
        if type(item_id) != int or item_id < 0:
            raise TypeError("get_user_item_rating: you supplied item_id = '%s' but item_id must be a positive integer" % item_id)
        if item_id not in self.item_train_ratings:
            raise ValueError("get_user_item_rating: you supplied item_id = %i but this item does not exist" % item_id)
        
        if user_id in self.user_train_ratings and item_id in self.user_train_ratings[user_id]:
            return self.user_train_ratings[user_id][item_id]
        else:
            return None
            
            
    def get_user_ratings(self, user_id: int) -> list:
        #ic("user_rec.get_user_ratings()")
         
        """
        [ratings: float]

        Gets all of user_id's ratings from the train set as a list. If this user has no ratings in the
        train set, an empty list is the result.
        """
        
        if type(user_id) != int or user_id < 0:
            raise TypeError("get_user_ratings: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_train_ratings:
            raise ValueError("get_user_ratings: you supplied user_id = %i but this user does not exist" % user_id)
        
        if user_id in self.user_train_ratings:
            return self.dataset.__d_to_dlist(self.user_train_ratings[user_id], 'item_id', 'rating')
        else:
            return []
            
            
    def get_user_mean_rating(self, user_id: int) -> float:
        #ic("user_rec.get_user_mean_rating()")
         
        """
        Gets the mean of user_id's ratings from the train set. If this user has no ratings in the
        train set, the mean is None.
        """
        
        if type(user_id) != int or user_id < 0:
            raise TypeError("get_user_mean_rating: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_train_ratings:
            raise ValueError("get_user_mean_rating: you supplied user_id = %i but this user does not exist" % user_id)
        
        if self.user_train_means[user_id] is None:
            return self.mean_train_rating
            
        else:
            return self.user_train_means[user_id]
    
           
    def get_demographic_ratings(self, age = None, gender = None, occupation = None, zipcode = None) -> list:
        #ic("user_rec.get_demographic_ratings()")
        
        """
        Gets all ratings from the train set as a list for users whose demographics matches the values in the arguments.
        """
        
        ratings = []
        
        for user_id, user_ratings in self.user_train_ratings.items():
            demographics = self.user_demographics[user_id]
            
            if (age is None or demographics['age'] == age) and (gender is None or demographics['gender'] == gender) and (occupation is None or demographics['occupation'] == occupation) and (zipcode is None or demographics['zipcode'] == zipcode):
                
                for item_id, rating in user_ratings.items():
                    ratings.append({'user_id': user_id, 'item_id': item_id, 'rating': rating})
                    
        return ratings               


    def get_user_demographics(self, user_id: int) -> dict:
        #ic("user_rec.get_user_demographics()")
        
        """
        Gets all of user_id's demographic data as a dictionary.
        """
        
        if type(user_id) != int or user_id < 0:
            raise TypeError("get_user_demographics: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_train_ratings:
            raise ValueError("get_user_demographics: you supplied user_id = %i but this user does not exist" % user_id)
        
        if user_id in self.user_demographics:
            return self.user_demographics[user_id]
        else:
            return {}
        
        
    def get_user_similarity(self, similarity_function: types.FunctionType, active_user_id: int, user_id: int) -> float:
        """"""
        #ic("user_rec.get_user_similarity()")
        
        sim = similarity_function(self.user_train_ratings[active_user_id], self.user_train_ratings[user_id])        
        
        return sim
