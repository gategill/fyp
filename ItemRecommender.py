from icecream import ic
import types
import Datasets as ds
import Similarities as sm

class ItemRecommender:
    def __init__(self):
        ic("item_rec.__init__()")
        
        dataset = ds.Datasets()
        self.user_training_ratings = dataset.get_user_training_ratings()
        self.user_training_means = dataset.get_user_training_means()
        self.movie_training_ratings = dataset.get_movie_training_ratings()
        self.movie_training_means = dataset.get_movie_training_means()
        self.test_ratings = dataset.get_test_ratings()
        
        
    def calculate_avg_rating(self, neighbours):
        ic("item_rec.calculate_avg_rating()")
        
        if len(neighbours) == 0:
            return None
        numerator = 0.0
        denominator = len(neighbours)
        
        for u_s_r in neighbours:
            rating = u_s_r['rating']
            numerator = numerator + rating
            
        if denominator <= 0.0:
            return None
        
        return numerator / denominator


    def calculate_wtd_avg_rating(self, neighbours): # weighted, introduces similarity
        ic("item_rec.calculate_wtd_avg_rating()")
        
        if len(neighbours) == 0:
            return None
        numerator = 0.0
        denominator = 0.0
        
        for u_s_r in neighbours:
            rating = u_s_r['rating']
            sim = u_s_r['sim']
            numerator = numerator + sim * rating
            denominator = denominator + sim
            
        if denominator <= 0.0:
            return None
        
        return numerator / denominator
    
    
    def predict_rating_item_based_nn(self, active_user_id, candidate_movie_id, k):
        ic("item_rec.predict_rating_item_based_nn()")
        
        nns = self.get_k_thresholded_nearest_movies(sm.sim_cosine, k, 0.0, candidate_movie_id, active_user_id)
        prediction = self.calculate_avg_rating(nns)
        
        if prediction:
            return prediction 
        else: 
            prediction = self.get_movie_mean_rating(candidate_movie_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_training_rating


    def predict_rating_item_based_nn_wtd(self, active_user_id, candidate_movie_id, k):
        ic("item_rec.predict_rating_item_based_nn_wtd()")
        
        nns = self.get_k_thresholded_nearest_movies(sm.sim_cosine, k, 0.0, candidate_movie_id, active_user_id)
        prediction = self.calculate_wtd_avg_rating(nns)
        
        if prediction:
            return prediction 
        else: 
            prediction = self.get_movie_mean_rating(candidate_movie_id)
            
            if prediction:
                return prediction
            else:
                return self.mean_training_rating
            

    def get_k_nearest_movies(self, similarity_function, k, candidate_movie_id, active_user_id = None):
        ic("item_rec.get_k_nearest_movies()")
        
        """
        Get the k nearest movies to candidate_movie_id.
        Optionally, if active_user_id is not None, the set of neighbours is confined to those rated by active_user_id.
        In this case, active_user_id's rating for candidate_movie_id is part of the final result.
        """
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_nearest_movies: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_nearest_movies: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.movie_training_ratings):
            raise ValueError("get_k_nearest_movies: you supplied k = %i but this is too large" % k)
        if type(candidate_movie_id) != int or candidate_movie_id < 1:
            raise TypeError("get_k_nearest_movies: you supplied candidate_movie_id = '%s' but candidate_movie_id must be a positive integer" % candidate_movie_id)
        if candidate_movie_id not in self.movie_training_ratings:
            raise ValueError("get_k_nearest_movies: you supplied candidate_movie_id = %i but this movie does not exist" % candidate_movie_id)
        if active_user_id:
            if type(active_user_id) != int or active_user_id < 1:
                raise TypeError("get_k_nearest_movies: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
            if active_user_id not in self.user_training_ratings:
                raise ValueError("get_k_nearest_movies: you supplied active_user_id = %i but this user does not exist" % active_user_id)     
        
        nearest_neighbours = []
        
        for movie_id in self.movie_training_ratings:
            if candidate_movie_id == movie_id:
                continue
            
            if (not active_user_id is None) and (not active_user_id in self.movie_training_ratings[movie_id]):
                continue
            
            sim = similarity_function(self.movie_training_ratings[candidate_movie_id], self.movie_training_ratings[movie_id])
            candidate_neighbour = {'movie_id': movie_id, 'sim': sim}
            
            if not active_user_id is None:
                candidate_neighbour['rating'] = self.movie_training_ratings[movie_id][active_user_id]
                
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


    def get_thresholded_nearest_movies(self, similarity_function, threshold, candidate_movie_id, active_user_id = None):
        ic("item_rec.get_thresholded_nearest_movies()")
        
        """
        Get the movies which are more than a threshold similar to candidate_movie_id
        Optionally, if active_user_id is not None, the set of neighbours is confined to those rated by active_user_id.
        In this case, active_user_id's rating for candidate_movie_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_thresholded_nearest_movies: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(threshold) != float:
            raise TypeError("get_thresholded_nearest_movies: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)
        if type(candidate_movie_id) != int or candidate_movie_id < 1:
            raise TypeError("get_thresholded_nearest_movies: you supplied candidate_movie_id = '%s' but candidate_movie_id must be a positive integer" % candidate_movie_id)
        if candidate_movie_id not in self.movie_training_ratings:
            raise ValueError("get_thresholded_nearest_movies: you supplied candidate_movie_id = %i but this movie does not exist" % candidate_movie_id)
        if active_user_id:
            if type(active_user_id) != int or active_user_id < 1:
                raise TypeError("get_thresholded_nearest_movies: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
            if active_user_id not in self.user_training_ratings:
                raise ValueError("get_thresholded_nearest_movies: you supplied active_user_id = %i but this user does not exist" % active_user_id)     
        
        nearest_neighbours = []
        
        for movie_id in self.movie_training_ratings:
            if candidate_movie_id == movie_id:
                continue
            
            if (not active_user_id is None) and (not active_user_id in self.movie_training_ratings[movie_id]):
                continue
            
            sim = similarity_function(self.movie_training_ratings[candidate_movie_id], self.movie_training_ratings[movie_id])
            
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'movie_id': movie_id, 'sim': sim}
            
            if not active_user_id is None:
                candidate_neighbour['rating'] = self.movie_training_ratings[movie_id][active_user_id]
                
            nearest_neighbours.append(candidate_neighbour)
            
        return nearest_neighbours   


    def get_k_thresholded_nearest_movies(self, similarity_function, k, threshold, candidate_movie_id, active_user_id = None):
        ic("item_rec.get_k_thresholded_nearest_movies()")
        
        """
        Get the k nearest movies to candidate_movie_id provided their similarity to candidate_movie_id exceeds the threshold.
        Optionally, if active_user_id is not None, the set of neighbours is confined to those rated by active_user_id.
        In this case, active_user_id's rating for candidate_movie_id is part of the final result.
        """
        
        if type(similarity_function) != types.FunctionType:
            raise TypeError("get_k_thresholded_nearest_movies: you supplied similarity_function = '%s' but similarity_function must be a function" % similarity_function)
        if type(k) != int or k < 1:
            raise TypeError("get_k_thresholded_nearest_movies: you supplied k = '%s' but k must be a positive integer" % k)
        if k > len(self.movie_training_ratings):
            raise ValueError("get_k_thresholded_nearest_movies: you supplied k = %i but this is too large" % k)
        if type(threshold) != float:
            raise TypeError("get_k_thresholded_nearest_movies: you supplied threshold = '%s' but threshold must be a floating point number" % threshold)            
        if type(candidate_movie_id) != int or candidate_movie_id < 1:
            raise TypeError("get_k_thresholded_nearest_movies: you supplied candidate_movie_id = '%s' but candidate_movie_id must be a positive integer" % candidate_movie_id)
        if candidate_movie_id not in self.movie_training_ratings:
            raise ValueError("get_k_thresholded_nearest_movies: you supplied candidate_movie_id = %i but this movie does not exist" % candidate_movie_id)
        if active_user_id:
            if type(active_user_id) != int or active_user_id < 1:
                raise TypeError("get_k_thresholded_nearest_movies: you supplied active_user_id = '%s' but active_user_id must be a positive integer" % active_user_id)
            if active_user_id not in self.user_training_ratings:
                raise ValueError("get_k_thresholded_nearest_movies: you supplied active_user_id = %i but this user does not exist" % active_user_id)     
        
        nearest_neighbours = []
        
        for movie_id in self.movie_training_ratings:
            if candidate_movie_id == movie_id:
                continue
            
            if (not active_user_id is None) and (not active_user_id in self.movie_training_ratings[movie_id]):
                continue
            
            sim = similarity_function(self.movie_training_ratings[candidate_movie_id], self.movie_training_ratings[movie_id])
            
            if sim <= threshold:
                continue
            
            candidate_neighbour = {'movie_id': movie_id, 'sim': sim}
            
            if not active_user_id is None:
                candidate_neighbour['rating'] = self.movie_training_ratings[movie_id][active_user_id]
                
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
    
    
    def get_movie_ratings(self, movie_id):
        ic("item_rec.get_movie_ratings()")
        
        """
        Gets all of movie_id's ratings from the training set as a list. If this movie has no ratings in the
        training set, an empty list is the result.
        """
        
        if type(movie_id) != int or movie_id < 1:
            raise TypeError("get_movie_ratings: you supplied movie_id = '%s' but movie_id must be a positive integer" % movie_id)
        if movie_id not in self.movie_training_ratings:
            raise ValueError("get_movie_ratings: you supplied movie_id = %i but this movie does not exist" % movie_id)
        
        if movie_id in self.movie_training_ratings:
            return ds.__d_to_dlist(self.movie_training_ratings[movie_id], 'user_id', 'rating')
        else:
            return []
            
            
    def get_movie_mean_rating(self, movie_id):
        ic("item_rec.get_movie_mean_rating()")
        
        """
        Gets the mean of movie_id's ratings from the training set. If this movie has no ratings in the
        training set, the mean is None.
        """
        
        if type(movie_id) != int or movie_id < 1:
            raise TypeError("get_movie_mean_rating: you supplied movie_id = '%s' but movie_id must be a positive integer" % movie_id)
        if movie_id not in self.movie_training_ratings:
            raise ValueError("get_movie_mean_rating: you supplied movie_id = %i but this movie does not exist" % movie_id)
        
        return self.movie_training_means[movie_id]
            
            
    def get_genre_ratings(self, genre):
        ic("item_rec.get_genre_ratings()")
        
        """
        Gets all ratings from the training set as a list for movies whose genre matches the value in the argument.
        """
        
        ratings = []
        
        for movie_id, movie_ratings in self.movie_training_ratings.items():
            genres = self.movie_descriptors[movie_id]['genres']
            
            if genre in ds.__genre_names and genres[ds.__genre_names.index(genre)] == 1:
                
                for user_id, rating in movie_ratings.items():
                    ratings.append({'user_id': user_id, 'movie_id': movie_id, 'rating': rating})
                    
        return ratings  


    def get_movie_descriptors(self, movie_id):
        ic("item_rec.get_movie_descriptors()")
        
        """
        Gets all of movie_id's descriptors as a dictionary.
        """
        
        if type(movie_id) != int or movie_id < 1:
            raise TypeError("get_movie_descriptors: you supplied movie_id = '%s' but movie_id must be a positive integer" % movie_id)
        if movie_id not in self.movie_training_ratings:
            raise ValueError("get_movie_descriptors: you supplied movie_id = %i but this movie does not exist" % movie_id)
        
        if movie_id in self.movie_descriptors:
            return self.movie_descriptors[movie_id]
        else:
            return {} 