from GenericRecommender import GenericRecommender
from icecream import ic
import types
import Datasets as ds
import Similarities as sm
from GenericRecommender import GenericRecommender


class UserRecommender(GenericRecommender):
    def __init__(self):
        ic("user_rec.__init__()")
        
        super().__init__()


    def predict_rating_user_based_nn(self, active_user_id, candidate_movie_id, k):
        ic("user_rec.predict_rating_user_based_nn()")
        
        nns = self.get_k_nearest_users(sm.sim_pearson, k, active_user_id, candidate_movie_id)
        prediction = self.calculate_avg_rating(nns)
        
        if prediction:
            return prediction
        else:
            prediction = self.get_user_mean_rating(active_user_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_training_rating


    def predict_rating_user_based_nn_wtd(self, active_user_id, candidate_movie_id, k):
        ic("user_rec.predict_rating_user_based_nn_wtd()")
        
        nns = self.get_k_nearest_users(sm.sim_pearson, k, active_user_id, candidate_movie_id)
        prediction = self.calculate_wtd_avg_rating(nns)
        
        if prediction:
            return prediction 
        else: 
            prediction = self.get_user_mean_rating(active_user_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_training_rating


    def get_k_nearest_users(self, similarity_function, k, active_user_id, candidate_movie_id = None):
        ic("user_rec.get_k_nearest_users()")
        
        """
        Get the k nearest users to active_user_id.
        Optionally, if candidate_movie_id is not None, the set of neighbours is confined to those who have 
        rated candidate_movie_id.
        In this case, each neighbour's rating for candidate_movie_id is part of the final result.
        
        THIS FOR PEARL PU!!!
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_nearest_users: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_nearest_users: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.user_training_ratings):
            raise ValueError("get_k_nearest_users: you supplied k = %i but this is too large" % k)
        if type(active_user_id) != int or active_user_id < 1:
            raise TypeError("get_k_nearest_users: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
        if active_user_id not in self.user_training_ratings:
            raise ValueError("get_k_nearest_users: you supplied active_user_id = %i but this user does not exist" % active_user_id)
        if candidate_movie_id:
            if type(candidate_movie_id) != int or candidate_movie_id < 1:
                raise TypeError("get_k_nearest_users: you supplied candidate_movie_id = '%s' but candidate_movie_id must be a positive integer" % candidate_movie_id)
            if candidate_movie_id not in self.movie_training_ratings:
                raise ValueError("get_k_nearest_users: you supplied candidate_movie_id = %i but this movie does not exist" % candidate_movie_id)     
        
        nearest_neighbours = []
        
        for user_id in self.user_training_ratings:
            
            if active_user_id == user_id:
                continue
            if (not candidate_movie_id is None) and (not candidate_movie_id in self.user_training_ratings[user_id]):
                continue
            
            sim = similarity_function(self.user_training_ratings[active_user_id], self.user_training_ratings[user_id])
            candidate_neighbour = {'user_id': user_id, 'sim': sim}
            
            if not candidate_movie_id is None: # if not None = if confined to users with that movie ID
                candidate_neighbour['rating'] = self.user_training_ratings[user_id][candidate_movie_id] # what's your ID? 
            
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


    def get_thresholded_nearest_users(self, similarity_function, threshold, active_user_id, candidate_movie_id = None):
        ic("user_rec.get_thresholded_nearest_users()")
        
        """
        Get the users who are more than a threshold similar to active_user_id
        Optionally, if movie_id is not None, the set of neighbours is confined to those who have rated candidate_movie_id.
        In this case, each neighbour's rating for candidate_movie_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_thresholded_nearest_users: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(threshold) != float:
            raise TypeError("get_thresholded_nearest_users: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)
        if type(active_user_id) != int or active_user_id < 1:
            raise TypeError("get_thresholded_nearest_users: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
        if active_user_id not in self.user_training_ratings:
            raise ValueError("get_thresholded_nearest_users: you supplied active_user_id = %i but this user does not exist" % active_user_id)
        if candidate_movie_id:
            if type(candidate_movie_id) != int or candidate_movie_id < 1:
                raise TypeError("get_thresholded_nearest_users: you supplied candidate_movie_id = '%s' but candidate_movie_id must be a positive integer" % candidate_movie_id)
            if candidate_movie_id not in self.movie_training_ratings:
                raise ValueError("get_thresholded_nearest_users: you supplied candidate_movie_id = %i but this movie does not exist" % candidate_movie_id)     
        
        nearest_neighbours = []
        
        for user_id in self.user_training_ratings: # 
            
            if active_user_id == user_id:
                continue
            
            if (not candidate_movie_id is None) and (not candidate_movie_id in self.user_training_ratings[user_id]):
                continue
            
            sim = similarity_function(self.user_training_ratings[active_user_id], self.user_training_ratings[user_id])
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'user_id': user_id, 'sim': sim}
            
            if not candidate_movie_id is None:
                candidate_neighbour['rating'] = self.user_training_ratings[user_id][candidate_movie_id]
                
            nearest_neighbours.append(candidate_neighbour)
            
        return nearest_neighbours  


    def get_k_thresholded_nearest_users(self, similarity_function, k, threshold, active_user_id, candidate_movie_id = None):
        ic("user_rec.get_k_thresholded_nearest_users()")
        
        """
        Get the k nearest users to active_user_id provided their similarity to active_user_id exceeds the threshold.
        Optionally, if movie_id is not None, the set of neighbours is confined to those who have rated candidate_movie_id.
        In this case, each neighbour's rating for candidate_movie_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_thresholded_nearest_users: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_thresholded_nearest_users: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.user_training_ratings):
            raise ValueError("get_k_thresholded_nearest_users: you supplied k = %i but this is too large" % k)
        if type(threshold) != float:
            raise TypeError("get_k_thresholded_nearest_users: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)
        if type(active_user_id) != int or active_user_id < 1:
            raise TypeError("get_k_thresholded_nearest_users: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
        if active_user_id not in self.user_training_ratings:
            raise ValueError("get_k_thresholded_nearest_users: you supplied active_user_id = %i but this user does not exist" % active_user_id)
        if candidate_movie_id:
            if type(candidate_movie_id) != int or candidate_movie_id < 1:
                raise TypeError("get_k_thresholded_nearest_users: you supplied candidate_movie_id = '%s' but candidate_movie_id must be a positive integer" % candidate_movie_id)
            if candidate_movie_id not in self.movie_training_ratings:
                raise ValueError("get_k_thresholded_nearest_users: you supplied candidate_movie_id = %i but this movie does not exist" % candidate_movie_id)     
        
        nearest_neighbours = []
        
        for user_id in self.user_training_ratings:
            
            if active_user_id == user_id:
                continue
            
            if (not candidate_movie_id is None) and (not candidate_movie_id in self.user_training_ratings[user_id]):
                continue
            
            sim = similarity_function(self.user_training_ratings[active_user_id], self.user_training_ratings[user_id])
            
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'user_id': user_id, 'sim': sim}
            
            if not candidate_movie_id is None:
                candidate_neighbour['rating'] = self.user_training_ratings[user_id][candidate_movie_id]
                
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


    def get_user_movie_rating(self, user_id, movie_id):
        ic("user_rec.get_user_movie_rating()")
        
        """
        Gets user_id's rating for movie_id from the training set or None if this user has no rating for this movie in the
        training set.
        """
        
        if type(user_id) != int or user_id < 1:
            raise TypeError("get_user_movie_rating: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_training_ratings:
            raise ValueError("get_user_movie_rating: you supplied user_id = %i but this user does not exist" % user_id)
        if type(movie_id) != int or movie_id < 1:
            raise TypeError("get_user_movie_rating: you supplied movie_id = '%s' but movie_id must be a positive integer" % movie_id)
        if movie_id not in self.movie_training_ratings:
            raise ValueError("get_user_movie_rating: you supplied movie_id = %i but this movie does not exist" % movie_id)
        
        if user_id in self.user_training_ratings and movie_id in self.user_training_ratings[user_id]:
            return self.user_training_ratings[user_id][movie_id]
        else:
            return None
            
            
    def get_user_ratings(self, user_id):
        ic("user_rec.get_user_ratings()")
         
        """
        Gets all of user_id's ratings from the training set as a list. If this user has no ratings in the
        training set, an empty list is the result.
        """
        
        if type(user_id) != int or user_id < 1:
            raise TypeError("get_user_ratings: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_training_ratings:
            raise ValueError("get_user_ratings: you supplied user_id = %i but this user does not exist" % user_id)
        
        if user_id in self.user_training_ratings:
            return ds.__d_to_dlist(self.user_training_ratings[user_id], 'movie_id', 'rating')
        else:
            return []
            
            
    def get_user_mean_rating(self, user_id):
        ic("user_rec.get_user_mean_rating()")
         
        """
        Gets the mean of user_id's ratings from the training set. If this user has no ratings in the
        training set, the mean is None.
        """
        
        if type(user_id) != int or user_id < 1:
            raise TypeError("get_user_mean_rating: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_training_ratings:
            raise ValueError("get_user_mean_rating: you supplied user_id = %i but this user does not exist" % user_id)
        
        return self.user_training_means[user_id]
    
           
    def get_demographic_ratings(self, age = None, gender = None, occupation = None, zipcode = None):
        ic("user_rec.get_demographic_ratings()")
        
        """
        Gets all ratings from the training set as a list for users whose demographics matches the values in the arguments.
        """
        
        ratings = []
        
        for user_id, user_ratings in self.user_training_ratings.items():
            demographics = self.user_demographics[user_id]
            
            if (age is None or demographics['age'] == age) and (gender is None or demographics['gender'] == gender) and (occupation is None or demographics['occupation'] == occupation) and (zipcode is None or demographics['zipcode'] == zipcode):
                
                for movie_id, rating in user_ratings.items():
                    ratings.append({'user_id': user_id, 'movie_id': movie_id, 'rating': rating})
                    
        return ratings               


    def get_user_demographics(self, user_id):
        ic("user_rec.get_user_demographics()")
        
        """
        Gets all of user_id's demographic data as a dictionary.
        """
        
        if type(user_id) != int or user_id < 1:
            raise TypeError("get_user_demographics: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_training_ratings:
            raise ValueError("get_user_demographics: you supplied user_id = %i but this user does not exist" % user_id)
        
        if user_id in self.user_demographics:
            return self.user_demographics[user_id]
        else:
            return {}