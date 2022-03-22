"""

"""


import random
from icecream import ic
import numpy as np
import pandas as pd
import math

class Dataset:
    def __init__(self, **kwargs):
        #ic("ds.__init__()")
        
        self.kwargs = kwargs
        self.DATA_PATH = self.kwargs["dataset_path"]
        self.__reset()
        self.load_items()
        self.load_users()
        self.read_in_ratings()
        self.prefilter_ratings()
    
        
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
        
        
    def read_in_ratings(self,  filename: str = "ratings_part.txt"):
        """ Reads in the data from a ratings file."""
        #ic("ds.read_in_ratings()")
        
        if self.kwargs["kfolds"] == 1:
            filename = "ratings.txt"
            # full test
        else:
            filename = "ratings_part.txt"
            # validation k folds
            

        
        if type(filename) != str:
            raise TypeError("load_ratings: you supplied filename = '%s' but filename must be a string" % filename)
        
        all_ratings = []
        for line in open(self.DATA_PATH + filename):
            substrings = line.strip().split('\t')
            if substrings[0] == "user_id":
                continue
            user_id = int(float(substrings[0]))
            item_id = int(float(substrings[1]))
            rating = float(substrings[2])
            
            if rating < 1.0:
                rating = 1.0
                
            if rating > 5:
                rating = 5.0
                                
            all_ratings.append({'user_id': user_id, 'item_id': item_id, 'rating': rating})
        
        # inplace, returns None
        random.shuffle(all_ratings)
        
        self.all_ratings = all_ratings
        
        
    def get_ratings_as_df(self):
        return pd.DataFrame(self.all_ratings)
    
    
    def prefilter_ratings(self):
        # cold start < 20 ratings

        # ic("ds.prefilter_rating()"       
        prefilterings = self.kwargs["prefiltering"]
        
        df = self.get_ratings_as_df()
        
        print("\n")
        for strategy, threshold in prefilterings.items():
            if type(strategy) != str:
                raise TypeError("Invalid prefiltering strategy. Valid strategies must be a string")
        
            if strategy not in ["user_k_core", "item_k_core", "cold_users"]:
                raise KeyError("Invalid prefiltering strategy. Valid strategies include: user_k_core, item_k_core, cold_users")
        
            if type(threshold) != int:
                raise TypeError("Invalid prefiltering threshold parameter. Ensure it's an integer")
        
            if threshold <= 0:
                raise ValueError("Invalid prefiltering threshold parameter. Ensure it's greater than 0")
        

            if strategy == "user_k_core":
                data = df.copy()
                
                print(f"\nPrefiltering with user {threshold}-core")
                print("df.shape BEFORE user_k_core: " + str(df.shape))

                print(f"The transactions before filtering are {len(data)}")
                print(f"The users before filtering are {data['user_id'].nunique()}")
                
                user_groups = data.groupby(['user_id'])
                data = user_groups.filter(lambda x: len(x) >= threshold)
                
                print(f"The transactions after filtering are {len(data)}")
                print(f"The users after filtering are {data['user_id'].nunique()}")
                
                data["user_id"] = data["user_id"].astype(int)
                data["item_id"] = data["item_id"].astype(int)
                
                df = data
                self.user_ids = list(map(int, np.unique(df["user_id"].to_list())))
                self.item_ids = list(map(int, np.unique(df["item_id"].to_list())))
                
                print("df.shape AFTER user_k_core: " + str(df.shape))

                
            if strategy == "item_k_core":
                data = df.copy()
                
                print(f"\nPrefiltering with item {threshold}-core")
                print("df.shape BEFORE item_k_core: " + str(df.shape))
                print(f"The transactions before filtering are {len(data)}")
                print(f"The items before filtering are {data['item_id'].nunique()}")
                
                item_groups = data.groupby(['item_id'])
                data = item_groups.filter(lambda x: len(x) >= threshold)
                
                print(f"The transactions after filtering are {len(data)}")
                print(f"The items after filtering are {data['item_id'].nunique()}")
                
                data["user_id"] = data["user_id"].astype(int)
                data["item_id"] = data["item_id"].astype(int)
                
                df = data
                self.user_ids = list(map(int, np.unique(df["user_id"].to_list())))
                self.item_ids = list(map(int, np.unique(df["item_id"].to_list())))
                
                print("df.shape AFTER item_k_core: " + str(df.shape))
                
            if strategy == "cold_users":
                data = df.copy()

                print(f"\nPrefiltering retaining cold users with {threshold} or less ratings")
                print("df.shape BEFORE cold_users: " + str(df.shape))
                print(f"The transactions before filtering are {len(data)}")
                print(f"The users before filtering are {data['user_id'].nunique()}")
                
                user_groups = data.groupby(['user_id'])
                data = user_groups.filter(lambda x: len(x) <= threshold)
                
                print(f"The transactions after filtering are {len(data)}")
                print(f"The users after filtering are {data['user_id'].nunique()}")
                data["user_id"] = data["user_id"].astype(int)
                data["item_id"] = data["item_id"].astype(int)
                
                df = data
                # should these be all or training? TODO
                self.user_ids = list(map(int, np.unique(df["user_id"].to_list())))
                self.item_ids = list(map(int, np.unique(df["item_id"].to_list())))
                
                print("df.shape AFTER cold_users: " + str(df.shape))
                
        print("\n")
        reduced_all_ratings = list(df.T.to_dict().values())
        self.all_ratings = reduced_all_ratings

          
    def load_ratings(self, fold_num = None) -> None:
        #ic("ds.load_ratings()")

        """
        It partitions the data randomly so that approximately test_splitting_ratio of the ratings are treated as a test set and 
        the remaining ratings are treated as the train set.
        """
     
        num_ratings = 0
        n_split = self.kwargs["kfolds"]
        
        df = self.get_ratings_as_df()
        nrow = df.shape[0]
        
        if n_split == 1:
            #df_test = df.sample(frac = 0.2)
            #df_train = pd.read_csv("./data/given/ratings_part.txt", sep = "\t")
            df_test = pd.read_csv("./data/given/ratings_part_test.txt", sep = "\t")
            #print(df_test.head())
            
        else:
            a,b = ((fold_num * nrow) // n_split), ((1 + fold_num) * nrow // n_split)
            df_test = df.loc[np.r_[a:b], :]
        
        df_train = df[~df.isin(df_test)].dropna()
        
        
        #df_train.to_csv("./ratings_part.txt", header=True, index=None, sep="\t", mode='a')
        #df_test.to_csv("./ratings_part_test.txt", header=True, index=None, sep="\t", mode='a')

        
        self.train_ratings = list(df_train.T.to_dict().values())
        self.test_ratings = list(df_test.T.to_dict().values())
        
        # train_ratings
        for entry in self.train_ratings:
            entry["user_id"] = int(entry["user_id"])
            entry["item_id"] = int(entry["item_id"])
            entry["rating"] = float(entry["rating"])

            user_id = entry["user_id"]
            item_id = entry["item_id"]
            rating = entry["rating"]
            
            self.user_train_ratings.setdefault(user_id, {})
            self.item_train_ratings.setdefault(item_id, {})
        
            self.user_train_ratings[user_id][item_id] = rating
            self.item_train_ratings[item_id][user_id] = rating
            self.mean_train_rating = self.mean_train_rating + rating
            num_ratings += 1            
                
        self.mean_train_rating = self.mean_train_rating / num_ratings
        
        self.num_ratings = num_ratings
        
        # should this be user/items in trainset? TODO
        self.sparsity = 1 - (self.num_ratings / (len(self.user_ids) * len(self.item_ids)))
        
        
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

        for entry in self.test_ratings:
            user_id = int(entry["user_id"])
            item_id = int(entry["item_id"])
            rating = entry["rating"]
            
            self.user_test_ratings.setdefault(user_id, {})
            self.item_test_ratings.setdefault(item_id, {})
            
            self.user_test_ratings[user_id][item_id] = rating
            self.item_test_ratings[item_id][user_id] = rating
            
        for entry in self.validation_ratings:
            user_id = int(entry["user_id"])
            item_id = int(entry["item_id"])
            rating = entry["rating"]
            
            self.user_validation_ratings.setdefault(user_id, {})
            self.item_validation_ratings.setdefault(item_id, {})
            
            self.user_validation_ratings[user_id][item_id] = rating
            self.item_validation_ratings[item_id][user_id] = rating
            
        #all_ratings_df = pd.DataFrame(self.all_ratings)
        #train_ratings_df = pd.DataFrame(self.train_ratings)
        #user_train_ratings_df = pd.DataFrame(self.user_train_ratings)
        #item_train_ratings_df = pd.DataFrame(self.item_train_ratings)
        
        #all_ratings_df.to_csv(self.DATA_PATH + "all_ratings.csv")
        #train_ratings_df.to_csv(self.DATA_PATH + "train_ratings.csv")
        #user_train_ratings_df.to_csv(self.DATA_PATH + "user_train_ratings.csv")
        #item_train_ratings_df.to_csv(self.DATA_PATH + "item_train_ratings.csv")
        
        
            
        print("there are {} ratings in the trainset".format(self.num_ratings))  
        print("sparsity of the trainset is: {}%".format(100 * round(self.sparsity, 4)))  


    def add_new_recommendations_to_trainset(self, new_recommendations):
        ic("ds.add_new_recommendations_to_trainset()")
        
        self.append_new_user_train_ratings(new_recommendations)
        self.append_new_item_train_ratings(new_recommendations)
        self.append_new_train_ratings(new_recommendations)
        self.update_user_train_means()
        self.update_item_train_means()
        self.update_num_ratings(new_recommendations)
        
        print("added {} new recommendations to the trainset".format(len(new_recommendations)))
        print("there are now {} ratings in the trainset".format(self.num_ratings))
        
        
    def get_user_validation_ratings(self):
        return self.user_validation_ratings 
    
    
    def get_item_validation_ratings(self):
        return self.item_validation_ratings 
    
    
    def get_validation_ratings(self):
        return self.validation_ratings 
       
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


    def append_new_user_train_ratings(self, new_recommendations: list) -> None:
        ic("ds.append_new_user_train_ratings()")
                
        for recommendation in new_recommendations:
            user_id = int(recommendation["user_id"])
            item_id = int(recommendation["item_id"])
            rating = recommendation["rating"]

            if user_id not in self.user_train_ratings:
                self.user_train_ratings[user_id] = {}    
                        
            self.user_train_ratings[user_id][item_id] = rating
                
        
    def update_user_train_means(self) -> None:
        ic("ds.update_user_train_means()")
        
        for user_id, ratings in self.user_train_ratings.items():
            if len(ratings) > 0:
                self.user_train_means[user_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:
                self.user_train_means[user_id] = None
                
                
    def append_new_item_train_ratings(self, new_recommendations: list) -> None:
        ic("ds.append_new_item_train_ratings()")
        
        for recommendation in new_recommendations:
            user_id = int(recommendation["user_id"])
            item_id = int(recommendation["item_id"])    
            rating = recommendation["rating"]  
            
            if rating is None:
                raise TypeError
              
            if item_id not in self.item_train_ratings:
                self.item_train_ratings[item_id] = {}

            self.item_train_ratings[item_id][user_id] = rating
            
            
    def append_new_train_ratings(self, new_recommendations: list) -> None:
        ic("ds.append_new_train_ratings()")
   
        for recommendation in new_recommendations:  
            if recommendation["rating"] is None:
                raise TypeError("'rating' is missing in recommendaion in ds.append_new_train_ratings()")
            self.train_ratings.append(recommendation)
                                
        
    def update_item_train_means(self) -> None:
        ic("ds.update_item_train_means()")
        
        for item_id, ratings in self.item_train_ratings.items():
            if len(ratings) > 0:
                self.item_train_means[item_id] = sum(ratings.values()) * 1.0 / len(ratings)
            else:

                self.item_train_means[item_id] = None

                
    def get_mean_train_rating(self) -> float:
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
        #ic("ds.update_num_ratings()")
        try:
            self.num_ratings += len(new_recommendations)
            assert(self.num_ratings == len(self.train_ratings)), "length of self.num_ratings != length of self.train_ratings"
            
        except AssertionError as msg:
            print(msg)


    def get_user_popularity(self, user_id: int) -> int:
        """"""
        ##ic("ds.get_user_popularity()")

        return len(self.user_train_ratings[user_id])

        
    def get_item_popularity(self, item_id: int) -> int:
        """"""
        ##ic("ds.get_item_popularity()")

        return len(self.item_train_ratings[item_id])
    
    
    def get_m_most_popular_items(self, m): # TODO
        k_most_popular_items = sorted(self.item_train_ratings, key = lambda k: len(self.item_train_ratings[k]))[-m:]
        
        print(k_most_popular_items)
        return k_most_popular_items

    
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
        
        self.user_validation_ratings = {}
        self.item_validation_ratings = {}
        self.validation_ratings = []
        
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
