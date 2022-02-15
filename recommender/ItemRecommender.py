"""

"""


from icecream import ic
import types
from recommender.Similarities import Similarities
from recommender.GenericRecommender import GenericRecommender


class ItemRecommender(GenericRecommender):
    def __init__(self, k: int, dataset = None) -> None:
        ic("item_rec.__init__()")
        
        super().__init__(k, dataset)
    
    
    def predict_rating_item_based_nn(self, active_user_id: int, candidate_item_id: int) -> float:
        ic("item_rec.predict_rating_item_based_nn()")
        
        nns = self.get_k_thresholded_nearest_items(Similarities.sim_cosine, self.k, 0.0, candidate_item_id, active_user_id)
        prediction = self.calculate_avg_rating(nns)
        
        if prediction:
            return prediction
        else:
            prediction = self.get_item_mean_rating(candidate_item_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_training_rating


    def predict_rating_item_based_nn_wtd(self, active_user_id: int, candidate_item_id: int) -> float:
        ic("item_rec.predict_rating_item_based_nn_wtd()")
        
        #nns = self.get_k_thresholded_nearest_items(Similarities.sim_cosine, self.k, 0.0, candidate_item_id, active_user_id)
        nns = self.get_k_nearest_items(Similarities.sim_cosine, self.k, candidate_item_id, active_user_id)
        prediction = self.calculate_wtd_avg_rating(nns)
        
        if prediction:
            return prediction
        else:
            prediction = self.get_item_mean_rating(candidate_item_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_training_rating
            

    def get_k_nearest_items(self, similarity_function: types.FunctionType, k: int, candidate_item_id: int, active_user_id: int = None) -> list:
        ic("item_rec.get_k_nearest_items()")
        
        """
        [{item_id: int, rating: float, sim: float}]

        Get the k nearest items to candidate_item_id.
        Optionally, if active_user_id is specified, the set of neighbours (items) is confined to those rated by active_user_id.
        In this case, active_user_id's rating for candidate_item_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_nearest_items: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_nearest_items: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.item_training_ratings):
            raise ValueError("get_k_nearest_items: you supplied k = %i but this is too large" % k)
        if type(candidate_item_id) != int or candidate_item_id < 1:
            raise TypeError("get_k_nearest_items: you supplied candidate_item_id = '%s' but candidate_item_id must be a positive integer" % candidate_item_id)
        if candidate_item_id not in self.item_training_ratings:
            raise ValueError("get_k_nearest_items: you supplied candidate_item_id = %i but this item does not exist" % candidate_item_id)
        if active_user_id:
            if type(active_user_id) != int or active_user_id < 1:
                raise TypeError("get_k_nearest_items: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
            if active_user_id not in self.user_training_ratings:
                raise ValueError("get_k_nearest_items: you supplied active_user_id = %i but this user does not exist" % active_user_id)     
        
        nearest_neighbours = []
        
        for item_id in self.item_training_ratings: # what is happening here???
            if candidate_item_id == item_id:
                continue
            
            if (not active_user_id is None) and (not active_user_id in self.item_training_ratings[item_id]):
                continue
            
            sim = self.get_item_similarity(similarity_function, candidate_item_id, item_id)
            candidate_neighbour = {'item_id': item_id, 'sim': sim}
            
            if not active_user_id is None:
                candidate_neighbour['rating'] = self.item_training_ratings[item_id][active_user_id]
                
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


    def get_thresholded_nearest_items(self, similarity_function: types.FunctionType, threshold: float, candidate_item_id: int, active_user_id: int = None) -> list:
        ic("item_rec.get_thresholded_nearest_items()")
        
        """
        [{item_id: int, rating: float, sim: float}]
        
        Get the items which are more than a threshold similar to candidate_item_id
        Optionally, if active_user_id is specified, the set of neighbours (items) is confined to those rated by active_user_id.
        In this case, active_user_id's rating for candidate_item_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_thresholded_nearest_items: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(threshold) != float:
            raise TypeError("get_thresholded_nearest_items: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)
        if type(candidate_item_id) != int or candidate_item_id < 1:
            raise TypeError("get_thresholded_nearest_items: you supplied candidate_item_id = '%s' but candidate_item_id must be a positive integer" % candidate_item_id)
        if candidate_item_id not in self.item_training_ratings:
            raise ValueError("get_thresholded_nearest_items: you supplied candidate_item_id = %i but this item does not exist" % candidate_item_id)
        if active_user_id:
            if type(active_user_id) != int or active_user_id < 1:
                raise TypeError("get_thresholded_nearest_items: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
            if active_user_id not in self.user_training_ratings:
                raise ValueError("get_thresholded_nearest_items: you supplied active_user_id = %i but this user does not exist" % active_user_id)     
        
        nearest_neighbours = []
        
        for item_id in self.item_training_ratings:
            if candidate_item_id == item_id:
                continue
            
            if (not active_user_id is None) and (not active_user_id in self.item_training_ratings[item_id]):
                continue
            
            sim = self.get_item_similarity(similarity_function, candidate_item_id, item_id)
            
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'item_id': item_id, 'sim': sim}
            
            if not active_user_id is None:
                candidate_neighbour['rating'] = self.item_training_ratings[item_id][active_user_id]
                
            nearest_neighbours.append(candidate_neighbour)
            
        return nearest_neighbours 


    def get_k_thresholded_nearest_items(self, similarity_function: types.FunctionType, k: int, threshold: float, candidate_item_id: int, active_user_id: int = None) -> list:
        ic("item_rec.get_k_thresholded_nearest_items()")
        
        """
        [{item_id: int, rating: float, sim: float}]

        Get the k nearest items to candidate_item_id provided their similarity to candidate_item_id exceeds the threshold.
        Optionally, if active_user_id is specified, the set of neighbours (items) is confined to those rated by active_user_id.
        In this case, active_user_id's rating for candidate_item_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_thresholded_nearest_items: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_thresholded_nearest_items: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.item_training_ratings):
            raise ValueError("get_k_thresholded_nearest_items: you supplied k = %i but this is too large" % k)
        if type(threshold) != float:
            raise TypeError("get_k_thresholded_nearest_items: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)            
        if type(candidate_item_id) != int or candidate_item_id < 1:
            raise TypeError("get_k_thresholded_nearest_items: you supplied candidate_item_id = '%s' but candidate_item_id must be a positive integer" % candidate_item_id)
        if candidate_item_id not in self.item_training_ratings:
            raise ValueError("get_k_thresholded_nearest_items: you supplied candidate_item_id = %i but this item does not exist" % candidate_item_id)
        if active_user_id:
            if type(active_user_id) != int or active_user_id < 1:
                raise TypeError("get_k_thresholded_nearest_items: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
            if active_user_id not in self.user_training_ratings:
                raise ValueError("get_k_thresholded_nearest_items: you supplied active_user_id = %i but this user does not exist" % active_user_id)     
        
        nearest_neighbours = []
        
        for item_id in self.item_training_ratings:
            if candidate_item_id == item_id:
                continue
            
            if (not active_user_id is None) and (not active_user_id in self.item_training_ratings[item_id]):
                continue
            
            sim = self.get_item_similarity(similarity_function, candidate_item_id, item_id)
            
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'item_id': item_id, 'sim': sim}
            
            if not active_user_id is None:
                candidate_neighbour['rating'] = self.item_training_ratings[item_id][active_user_id]
                
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
    
    
    def get_item_ratings(self, item_id: int) -> list:
        ic("item_rec.get_item_ratings()")
        
        """
        [ratings: float]
        
        Gets all of item_id's ratings from the training set as a list. If this item has no ratings in the
        training set, an empty list is the result.
        """
        
        if type(item_id) != int or item_id < 1:
            raise TypeError("get_item_ratings: you supplied item_id = '%s' but item_id must be a positive integer" % item_id)
        if item_id not in self.item_training_ratings:
            raise ValueError("get_item_ratings: you supplied item_id = %i but this item does not exist" % item_id)
        
        if item_id in self.item_training_ratings:
            return self.dataset.__d_to_dlist(self.item_training_ratings[item_id], 'user_id', 'rating')
        else:
            return []
            
            
    def get_item_mean_rating(self, item_id: int) -> float:
        ic("item_rec.get_item_mean_rating()")
        
        """
        Gets the mean of item_id's ratings from the training set. If this item has no ratings in the
        training set, the mean is None.
        """
        
        if type(item_id) != int or item_id < 1:
            raise TypeError("get_item_mean_rating: you supplied item_id = '%s' but item_id must be a positive integer" % item_id)
        if item_id not in self.item_training_ratings:
            raise ValueError("get_item_mean_rating: you supplied item_id = %i but this item does not exist" % item_id)
        
        return self.item_training_means[item_id]
            
            
    def get_genre_ratings(self, genre: str) -> list:
        ic("item_rec.get_genre_ratings()")
        
        """
        [ratings: float]
        
        Gets all ratings from the training set as a list for items whose genre matches the value in the argument.
        """
        
        ratings = []
        
        for item_id, item_ratings in self.item_training_ratings.items():
            genres = self.item_descriptors[item_id]['genres']
            
            if genre in self.dataset.__genre_names and genres[self.dataset.__genre_names.index(genre)] == 1:
                
                for user_id, rating in item_ratings.items():
                    ratings.append({'user_id': user_id, 'item_id': item_id, 'rating': rating})
                    
        return ratings  


    def get_item_descriptors(self, item_id: int) -> dict:
        ic("item_rec.get_item_descriptors()")
        
        """
        Gets all of item_id's descriptors as a dictionary.
        """
        
        if type(item_id) != int or item_id < 1:
            raise TypeError("get_item_descriptors: you supplied item_id = '%s' but item_id must be a positive integer" % item_id)
        if item_id not in self.item_training_ratings:
            raise ValueError("get_item_descriptors: you supplied item_id = %i but this item does not exist" % item_id)
        
        if item_id in self.item_descriptors:
            return self.item_descriptors[item_id]
        else:
            return {} 
        
        
    def get_item_similarity(self, similarity_function: types.FunctionType, candidate_item_id: int, item_id: int) -> float:
        """"""
        #ic("item_rec.get_item_similarity()")
        
        return similarity_function(self.item_training_ratings[candidate_item_id], self.item_training_ratings[item_id])
        
