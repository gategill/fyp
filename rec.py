import random
from tkinter.messagebox import NO
import types
import math

def sim_pearson(ratings1, ratings2):
    # Get co-rated
    co_rated = []
    for id in ratings1: 
        if id in ratings2: 
            co_rated.append(id)
    n = len(co_rated)
    # If no co-rated, return 0
    if n == 0: 
        return 0.0
    # Means over co-rated
    mean1 = sum([ratings1[id] for id in co_rated]) * 1.0 / n
    mean2 = sum([ratings2[id] for id in co_rated]) * 1.0 / n
    # Sums of products of differences and sums of squared differences
    sum_prods = 0.0
    sum_squares1 = 0.0
    sum_squares2 = 0.0
    for id in co_rated:
        difference1 = ratings1[id] - mean1
        difference2 = ratings2[id] - mean2
        sum_prods = sum_prods + difference1 * difference2
        sum_squares1 = sum_squares1 + difference1 * difference1
        sum_squares2 = sum_squares2 + difference2 * difference2
    # Handle zero variances
    if sum_squares1 == 0 or sum_squares2 == 0:
        return 0.0
    # Calculate Pearson correlation
    return sum_prods / (math.sqrt(sum_squares1) * math.sqrt(sum_squares2))

def sim_cosine(ratings1, ratings2):
    if len(ratings1) == 0 or len(ratings2) == 0:
        return 0.0
    return dot_product(ratings1, ratings2) / (magnitude(ratings1) * magnitude(ratings2))

def dot_product(ratings1, ratings2):
    dot_product = 0.0
    for id in ratings1:
        if id in ratings2:
            dot_product = dot_product + ratings1[id] * ratings2[id]
    return dot_product
    
def magnitude(ratings):
    sum_squares = 0.0
    for rating in ratings.values():
        sum_squares = sum_squares + rating * rating
    return math.sqrt(sum_squares)
    
class Recommender:

    def __init__(self):
        self.movie_descriptors = {}
        for line in open('movies.txt'):
            substrings = line.strip().split('|')
            movie_id, title, release_date, video_release_date, url = substrings[:5]
            genres = [int(genre) for genre in substrings[-19:]]
            self.movie_descriptors[int(movie_id)] = {'title': title, 'release_date': release_date, 'video_release_date': video_release_date, 'url': url, 'genres': genres}
        self.user_demographics = {}
        for line in open('users.txt'):
            user_id, age, gender, occupation, zipcode = line.strip().split('|')
            self.user_demographics[int(user_id)] = {'age': int(age), 'gender': gender, 'occupation': occupation, 'zipcode': zipcode}
        self.__reset()
                    
    def load_ratings(self, filename, test_percentage = 30, seed = None):
        """
        Reads in the data from a ratings file.
        It partitions the data randomly so that approximately test_percentage of the ratings are treated as a test set and 
        the remaining ratings are treated as the training set.
        """
        if type(filename) != str:
            raise TypeError("load_ratings: you supplied filename = '%s' but filename must be a string" % filename)
        if type(test_percentage) != int:
            raise TypeError("load_ratings: you supplied test_percentage = '%s' but test_percentage must be an integer" % test_percentage)
        if test_percentage < 0 or test_percentage > 100:
            raise ValueError("load_ratings: you supplied test_percentage = '%i' but test_percentage must be between 0 and 100 inclusive" % test_percentage)
        if not seed is None and type(seed) != int:
            raise TypeError("load_ratings: you supplied seed = '%s' but seed, if supplied at all, must be an integer" % seed)
        random.seed(seed)
        self.__reset()
        test_proportion = test_percentage / 100.0
        self.mean_training_rating = 0.0
        num_ratings = 0
        for line in open(filename):
            substrings = line.strip().split('\t')
            user_id = int(substrings[0])
            movie_id = int(substrings[1])
            rating = float(substrings[2])
            self.user_training_ratings.setdefault(user_id, {})
            self.movie_training_ratings.setdefault(movie_id, {})
            if random.random() >= test_proportion:
                self.user_training_ratings[user_id][movie_id] = rating
                self.movie_training_ratings[movie_id][user_id] = rating
                self.mean_training_rating= self.mean_training_rating + rating
                num_ratings = num_ratings + 1
            else:
                self.test_ratings.append({'user_id': user_id, 'movie_id': movie_id, 'rating': rating})
        self.mean_training_rating = self.mean_training_rating / num_ratings
        for user_id, ratings in self.user_training_ratings.items():
            if len(ratings) > 0:
                self.user_training_means[user_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:
                self.user_training_means[user_id] = None
        for movie_id, ratings in self.movie_training_ratings.items():
            if len(ratings) > 0:
                self.movie_training_means[movie_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:
                self.movie_training_means[movie_id] = None

    def get_test_ratings(self):
        """
        Gets all the test ratings.
        """
        return self.test_ratings

    def calculate_avg_rating(self, neighbours):
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

    def calculate_wtd_avg_rating(self, neighbours):
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

    def predict_rating_user_based_nn(self, active_user_id, candidate_movie_id, k):
        nns = self.get_k_nearest_users(sim_pearson, k, active_user_id, candidate_movie_id)
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
        nns = self.get_k_nearest_users(sim_pearson, k, active_user_id, candidate_movie_id)
        prediction = self.calculate_wtd_avg_rating(nns)
        if prediction:
            return prediction 
        else: 
            prediction = self.get_user_mean_rating(active_user_id)
            if prediction:
                return prediction
            else:
                return self.mean_training_rating

    def predict_rating_item_based_nn(self, active_user_id, candidate_movie_id, k):
        nns = self.get_k_thresholded_nearest_movies(sim_cosine, k, 0.0, candidate_movie_id, active_user_id)
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
        nns = self.get_k_thresholded_nearest_movies(sim_cosine, k, 0.0, candidate_movie_id, active_user_id)
        prediction = self.calculate_wtd_avg_rating(nns)
        if prediction:
            return prediction 
        else: 
            prediction = self.get_movie_mean_rating(candidate_movie_id)
            if prediction:
                return prediction
            else:
                return self.mean_training_rating

    def get_k_nearest_users(self, similarity_function, k, active_user_id, candidate_movie_id = None):
        """
        Get the k nearest users to active_user_id.
        Optionally, if candidate_movie_id is not None, the set of neighbours is confined to those who have 
        rated candidate_movie_id.
        In this case, each neighbour's rating for candidate_movie_id is part of the final result.
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

    def get_thresholded_nearest_users(self, similarity_function, threshold, active_user_id, candidate_movie_id = None):
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
        return nearest_neighbours  

    def get_k_thresholded_nearest_users(self, similarity_function, k, threshold, active_user_id, candidate_movie_id = None):
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

    def get_k_nearest_movies(self, similarity_function, k, candidate_movie_id, active_user_id = None):
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

    def get_user_movie_rating(self, user_id, movie_id):
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
        """
        Gets all of user_id's ratings from the training set as a list. If this user has no ratings in the
        training set, an empty list is the result.
        """
        if type(user_id) != int or user_id < 1:
            raise TypeError("get_user_ratings: you supplied user_id = '%s' but user_id must be a positive integer" % user_id)
        if user_id not in self.user_training_ratings:
            raise ValueError("get_user_ratings: you supplied user_id = %i but this user does not exist" % user_id)
        if user_id in self.user_training_ratings:
            return Recommender.__d_to_dlist(self.user_training_ratings[user_id], 'movie_id', 'rating')
        else:
            return []
            
    def get_user_mean_rating(self, user_id):
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
            
    def get_movie_ratings(self, movie_id):
        """
        Gets all of movie_id's ratings from the training set as a list. If this movie has no ratings in the
        training set, an empty list is the result.
        """
        if type(movie_id) != int or movie_id < 1:
            raise TypeError("get_movie_ratings: you supplied movie_id = '%s' but movie_id must be a positive integer" % movie_id)
        if movie_id not in self.movie_training_ratings:
            raise ValueError("get_movie_ratings: you supplied movie_id = %i but this movie does not exist" % movie_id)
        if movie_id in self.movie_training_ratings:
            return Recommender.__d_to_dlist(self.movie_training_ratings[movie_id], 'user_id', 'rating')
        else:
            return []
            
    def get_movie_mean_rating(self, movie_id):
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
        """
        Gets all ratings from the training set as a list for movies whose genre matches the value in the argument.
        """
        ratings = []
        for movie_id, movie_ratings in self.movie_training_ratings.items():
            genres = self.movie_descriptors[movie_id]['genres']
            if genre in Recommender.__genre_names and genres[Recommender.__genre_names.index(genre)] == 1:
                for user_id, rating in movie_ratings.items():
                    ratings.append({'user_id': user_id, 'movie_id': movie_id, 'rating': rating})
        return ratings  

    def get_movie_descriptors(self, movie_id):
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

    def __reset(self):
        self.user_training_ratings = {}
        self.user_training_means = {}
        self.movie_training_ratings = {}
        self.movie_training_means = {}
        self.test_ratings = []

    @staticmethod
    def __d_to_dlist(dict, keykey, valkey):
        list = []
        for key, val in dict.items():
            list.append({keykey: key, valkey: val})
        return list
        
    __genre_names = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
