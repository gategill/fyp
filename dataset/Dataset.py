"""

"""


import random
from icecream import ic
import os
import numpy as np
import pandas as pd

class Dataset:
    def __init__(self):
        #ic("ds.__init__()")
        self.DATA_PATH = "./data/derek/"
        self.__reset()
        self.load_items()
        self.load_users()
        self.read_in_ratings()
        self.load_ratings()    
        
        print("There are {} ratings in the trainset".format(self.num_ratings))  
        print("Sparsity of the trainset is: {}%".format(round(self.sparsity, 5)))  
    
        
    def load_items(self, filename: str = "movies.txt") -> None:
        #ic("ds.load_items()")
        
        START = 5
        FINISH = -19
        self.item_descriptors = {}
        item_ids = []
        
        for line in open(self.DATA_PATH + filename):
            substrings = line.strip().split('|')
            item_id, title, release_date, video_release_date, url = substrings[:START]
            genres = [int(genre) for genre in substrings[FINISH:]]
            self.item_descriptors[int(item_id)] = {'title': title, 'release_date': release_date, 'video_release_date': video_release_date, 'url': url, 'genres': genres}
            item_ids.append(int(item_id))
            
        self.item_ids = np.unique(item_ids)
        
        
    def load_users(self, filename: str = "users.txt") -> None:
        #ic("ds.load_users()")

        user_ids = []
        self.user_demographics = {}
        
        for line in open(self.DATA_PATH + filename):
            user_id, age, gender, occupation, zipcode = line.strip().split('|')
            self.user_demographics[int(user_id)] = {'age': int(age), 'gender': gender, 'occupation': occupation, 'zipcode': zipcode}
            user_ids.append(int(user_id))
        
        self.user_ids = np.unique(user_ids)
        # shuffle, k fod
        
    def read_in_ratings(self,  filename: str = "ratings.txt"):
        """ Reads in the data from a ratings file."""
        
        #ic("ds.read_in_ratings()")
        
        if type(filename) != str:
            raise TypeError("load_ratings: you supplied filename = '%s' but filename must be a string" % filename)
        
        all_ratings = []
        for line in open(self.DATA_PATH + filename):
            substrings = line.strip().split('\t')
            user_id = int(substrings[0])
            item_id = int(substrings[1])
            rating = float(substrings[2])
            
            if rating < 1.0:
                rating = 1.0
                
            if rating > 5:
                rating = 5.0
                                
            all_ratings.append({'user_id': user_id, 'item_id': item_id, 'rating': rating})
        
        random.shuffle(all_ratings) # inplace, returns None
        
        self.all_ratings = all_ratings
        #self.all_ratings = random.shuffle(all_ratings)
        
        
    def get_ratings_as_df(self):
        return pd.DataFrame(self.all_ratings)
    
        
    def load_ratings(self, test_percentage: int = 20, seed: int = 42) -> None:
        #ic("ds.load_ratings()")

        """
        It partitions the data randomly so that approximately test_percentage of the ratings are treated as a test set and 
        the remaining ratings are treated as the train set.
        """
     
        if type(test_percentage) != int:
            raise TypeError("load_ratings: you supplied test_percentage = '%s' but test_percentage must be an integer" % test_percentage)
        if test_percentage < 0 or test_percentage > 100:
            raise ValueError("load_ratings: you supplied test_percentage = '%i' but test_percentage must be between 0 and 100 inclusive" % test_percentage)
        if not seed is None and type(seed) != int:
            raise TypeError("load_ratings: you supplied seed = '%s' but seed, if supplied at all, must be an integer" % seed)
        
        random.seed(seed) # removed seed
        
        #self.__reset()
        test_proportion = test_percentage / 100.0
        num_ratings = 0
        
        for entry in self.all_ratings:
            #substrings = line.strip().split('\t')
            user_id = entry["user_id"]
            item_id = entry["item_id"]
            rating = entry["rating"]
            
            self.user_train_ratings.setdefault(user_id, {})
            self.item_train_ratings.setdefault(item_id, {})
            
            if random.random() >= test_proportion: # goes to train
                self.user_train_ratings[user_id][item_id] = rating
                self.item_train_ratings[item_id][user_id] = rating
                self.mean_train_rating = self.mean_train_rating + rating
                self.train_ratings.append(entry)
                num_ratings = num_ratings + 1
                
                #ic(self.user_train_ratings)
                #ic(self.item_train_ratings)
 
            else: # goes to testing
                self.test_ratings.append(entry)
                #ic(self.test_ratings)
                
            #sleep(1)
                
        self.mean_train_rating = self.mean_train_rating / num_ratings
        
        self.num_ratings = num_ratings
        self.sparsity = 1 - self.num_ratings / (len(self.user_ids) * len(self.item_ids))
        
        
        for user_id, ratings in self.user_train_ratings.items():
            
            if len(ratings) > 0:
                self.user_train_means[user_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:
                self.user_train_means[user_id] = None
                
        for item_id, ratings in self.item_train_ratings.items():
            
            if len(ratings) > 0:
                self.item_train_means[item_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:
                self.item_train_means[item_id] = None
                
        for val in self.test_ratings:
            user_id = val["user_id"]
            item_id = val["item_id"]
            rating = val["rating"]
            
            self.user_test_ratings.setdefault(user_id, {})
            self.item_test_ratings.setdefault(item_id, {})
            
            self.user_test_ratings[user_id][item_id] = rating
            self.item_test_ratings[item_id][user_id] = rating


    def add_new_recommendations_to_dataset(self, new_recommendations):
        # 
        #ic("ds.add_new_recommendations_to_dataset()")
        
        self.update_user_train_ratings(new_recommendations)
        self.update_item_train_ratings(new_recommendations)
        self.update_train_ratings(new_recommendations)
        self.update_user_train_means()
        self.update_item_train_means()
        self.update_mean_train_rating()
        self.update_num_ratings(new_recommendations)
        
        # will have to update these !!!
        print("There are {} ratings in the trainset".format(self.num_ratings))
        #print("Sparsity of the trainset is: {}%".format(self.sparsity))

                
    def get_user_ids(self) -> list:
        # [user_ids]
        #ic("ds.get_user_ids()")
        
        return self.user_ids
        
        
    def get_item_ids(self) -> list:
        # [item_ids]
        #ic("ds.get_item_ids()")
        
        return self.item_ids


    def get_user_train_ratings(self) -> dict:
        # {user_id: int : {moive_id: int : rating: float}
        #ic("ds.get_user_train_ratings()")
    
        return self.user_train_ratings


    def update_user_train_ratings(self, new_recommendations: list) -> None:
        # 
        #ic("ds.update_user_train_ratings()")
        
        #self.num_ratings += len(new_recommendations)
        
        for recommendation in new_recommendations:
            user_id = recommendation["user_id"]
            item_id = recommendation["item_id"]
            rating = recommendation["rating"]
            
            self.user_train_ratings[user_id][item_id] = rating
                
        
    def update_user_train_means(self) -> None:
        # 
        #ic("ds.update_user_train_means()")
        
        for user_id, ratings in self.user_train_ratings.items():
            if len(ratings) > 0:
                self.user_train_means[user_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:
                self.user_train_means[user_id] = None
                
                
    def update_item_train_ratings(self, new_recommendations: list) -> None:
        # 
        #ic("ds.update_item_train_ratings()")
        
   
        for recommendation in new_recommendations:
            user_id = recommendation["user_id"]
            item_id = recommendation["item_id"]
            rating = recommendation["rating"]
            
            self.item_train_ratings[item_id][user_id] = rating
            
            
    def update_train_ratings(self, new_recommendations: list) -> None:
        # 
        #ic("ds.update_train_ratings()")
        
   
        for recommendation in new_recommendations:            
            self.train_ratings.append(recommendation)
                                
        
    def update_item_train_means(self) -> None:
        # 
        #ic("ds.update_item_train_means()")
        
        for item_id, ratings in self.item_train_ratings.items():
            if len(ratings) > 0:
                self.item_train_means[item_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:
                self.item_train_means[item_id] = None
                
                    
    def update_mean_train_rating(self) -> None:
        # 
        #ic("ds.update_mean_train_rating()")
        
        new_mean_train_rating = np.sum(list(self.item_train_means.values())) / len(self.item_train_means)
        
        self.mean_train_rating = new_mean_train_rating

                
    def get_mean_train_rating(self) -> float:
        # 
        #ic("ds.get_mean_train_rating()")

        return self.mean_train_rating


    def get_user_train_means(self) -> dict:
        # {user_id: int, rating: float}
        #ic("ds.get_user_train_means()")
    
        return self.user_train_means
    
    
    def get_item_train_ratings(self) -> dict:
        # {item_id: int : {user_id: int, rating: float}
        #ic("ds.get_item_train_ratings()")
    
        return self.item_train_ratings
    
    
    def get_item_train_means(self) -> dict:
        # {item_id: int : rating: float}
        #ic("ds.get_item_train_means()")

        return self.item_train_means
    

    def get_train_ratings(self) -> dict:
        # {item_id: int : rating: float}
        #ic("ds.get_train_ratings()")

        return self.train_ratings
        

    def get_test_ratings(self) -> list:
        # [{item_id: int, rating: float, user_id: int}]
        #ic("ds.get_test_ratings()")

        return self.test_ratings
    
    
    def get_user_test_ratings(self) -> dict:
        # {user_id : {item_id, rating}
        #ic("ds.get_user_test_ratings()")

        return self.user_test_ratings


    def get_item_test_ratings(self) -> dict:
        # {item_d: int : {user_id: int, rating: float}
        #ic("ds.get_item_test_ratingss()")
        
        return self.item_test_ratings
        
        
    def update_num_ratings(self, new_recommendations: list) -> None:
        """"""
        # TODO should I replace this???
        #ic("ds.update_num_ratings()")
        
        self.num_ratings += len(new_recommendations)
        
        #print("is self.num_ratings == len(self.train_ratings) ?")
        #print(self.num_ratings == len(self.train_ratings))
        
        
    def get_user_popularity(self, user_id: int) -> int:
        """"""
        ##ic("ds.get_user_popularity()")

        return len(self.user_train_ratings[user_id])

        
    def get_item_popularity(self, item_id: int) -> int:
        """"""
        ##ic("ds.get_item_popularity()")

        return len(self.item_train_ratings[item_id])

    
    def __reset(self) -> None:
        #ic("ds.__reset()")
        
        self.user_ids = []
        self.item_ids = []
        
        self.user_train_ratings = {}
        self.user_train_means = {}
        self.item_train_ratings = {}
        self.item_train_means = {}
        self.train_ratings = []
        
        self.user_test_ratings = {}
        self.item_test_ratings = {}
        self.test_ratings = []
        
        self.mean_train_rating = 0.0
        
        self.transactions = 0
        self.sparsity = 0.0
        

    @staticmethod
    def __d_to_dlist(dict: dict, keykey: int, valkey: int) -> list:
        #ic("ds.__d_to_dlist()")
        ic(keykey)
        ic(valkey)
        
        list = []
        for key, val in dict.items():
            
            list.append({keykey: key, valkey: val})
            
        return list
        
        
    __genre_names = ["unknown", "Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
