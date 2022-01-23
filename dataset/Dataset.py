"""

"""


import random
from icecream import ic
import os
import numpy as np


class Dataset:
    def __init__(self):
        ic("ds.__init__()")
        self.DATA_PATH = "C:/Users/Asus/Desktop/UCC 4/CS4701 -Final Year Project/FYP_Code/data/derek/"
        self.__reset()
        self.load_items()
        self.load_users()
        self.load_ratings()
       
        
    def load_items(self, filename: str = "movies.txt") -> None:
        ic("ds.load_items()")
        
        START = 5
        FINISH = -19
        self.movie_descriptors = {}
        movie_ids = []
        
        for line in open(self.DATA_PATH + filename):
            substrings = line.strip().split('|')
            movie_id, title, release_date, video_release_date, url = substrings[:START]
            genres = [int(genre) for genre in substrings[FINISH:]]
            self.movie_descriptors[int(movie_id)] = {'title': title, 'release_date': release_date, 'video_release_date': video_release_date, 'url': url, 'genres': genres}
            movie_ids.append(int(movie_id))
            
        self.movie_ids = np.unique(movie_ids)
        
        
    def load_users(self, filename: str = "users.txt") -> None:
        ic("ds.load_users()")

        user_ids = []
        self.user_demographics = {}
        
        for line in open(self.DATA_PATH + filename):
            user_id, age, gender, occupation, zipcode = line.strip().split('|')
            self.user_demographics[int(user_id)] = {'age': int(age), 'gender': gender, 'occupation': occupation, 'zipcode': zipcode}
            user_ids.append(int(user_id))
        
        self.user_ids = np.unique(user_ids)
        
        
    def load_ratings(self, filename: str = "ratings.txt", test_percentage: int = 20, seed: int = 2) -> None:
        ic("ds.load_ratings()")

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
        
        #self.__reset()
        test_proportion = test_percentage / 100.0
        self.mean_training_rating = 0.0
        num_ratings = 0
        
        for line in open(self.DATA_PATH + filename):
            substrings = line.strip().split('\t')
            user_id = int(substrings[0])
            movie_id = int(substrings[1])
            rating = float(substrings[2])
            self.user_training_ratings.setdefault(user_id, {})
            self.movie_training_ratings.setdefault(movie_id, {})
            
            if random.random() >= test_proportion: # goes to training
                self.user_training_ratings[user_id][movie_id] = rating
                self.movie_training_ratings[movie_id][user_id] = rating
                self.mean_training_rating= self.mean_training_rating + rating
                num_ratings = num_ratings + 1
                
                #ic(self.user_training_ratings)
                #ic(self.movie_training_ratings)
 
            else: # goes to testing
                self.test_ratings.append({'user_id': user_id, 'movie_id': movie_id, 'rating': rating})
                #ic(self.test_ratings)
                
            #sleep(1)
                
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
                
        for val in self.test_ratings:
            user_id = val["user_id"]
            movie_id = val["movie_id"]
            rating = val["rating"]
            
            self.user_test_ratings.setdefault(user_id, {})
            self.movie_test_ratings.setdefault(movie_id, {})
            
            self.user_test_ratings[user_id][movie_id] = rating
            self.movie_test_ratings[movie_id][user_id] = rating
                
                
    def get_user_ids(self) -> list:
        # [user_ids]
        ic("ds.get_user_ids()")
        
        return self.user_ids
        
        
    def get_movie_ids(self) -> list:
        # [movie_ids]
        ic("ds.get_movie_ids()")
        
        return self.movie_ids


    def get_user_training_ratings(self) -> dict:
        # {user_id: int : {moive_id: int : rating: float}
        ic("ds.get_user_training_ratings()")
    
        return self.user_training_ratings
    
    
    def get_user_training_means(self) -> dict:
        # {user_id: int, rating: float}
        ic("ds.get_user_training_means()")
    
        return self.user_training_means
    
    
    def get_movie_training_ratings(self) -> dict:
        # {movie_id: int : {user_id: int, rating: float}
        ic("ds.get_movie_training_ratings()")
    
        return self.movie_training_ratings
    
    
    def get_movie_training_means(self) -> dict:
        # {movie_id: int : rating: float}
        ic("ds.get_movie_training_means()")

        return self.movie_training_means
        

    def get_test_ratings(self) -> list:
        # [{movie_id: int, rating: float, user_id: int}]
        ic("ds.get_test_ratings()")

        return self.test_ratings
    
    
    def get_user_test_ratings(self) -> dict:
        # {user_id : {movie_id, rating}
        ic("ds.get_user_test_ratings()")

        return self.user_test_ratings


    def get_movie_test_ratings(self) -> dict:
        # {movie_d: int : {user_id: int, rating: float}
        ic("ds.get_movie_test_ratingss()")
        
        return self.movie_test_ratings
    
    
    def __reset(self) -> None:
        ic("ds.__reset()")
        
        self.user_ids = []
        self.movie_ids = []
        
        self.user_training_ratings = {}
        self.user_training_means = {}
        self.movie_training_ratings = {}
        self.movie_training_means = {}
        
        self.user_test_ratings = {}
        self.movie_test_ratings = {}
        self.test_ratings = []


    @staticmethod
    def __d_to_dlist(dict: dict, keykey: int, valkey: int) -> list:
        ic("ds.__d_to_dlist()")
        ic(keykey)
        ic(valkey)
        
        list = []
        for key, val in dict.items():
            
            list.append({keykey: key, valkey: val})
            
        return list
        
        
    __genre_names = ["unknown", "Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
