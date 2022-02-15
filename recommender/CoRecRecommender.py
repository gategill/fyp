"""

"""


#from time import sleep
from icecream import ic
from dataset.Dataset import Dataset
from recommender.GenericRecommender import GenericRecommender
from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender
from recommender.Similarities import Similarities
import random
import copy
import pandas as pd


class CoRecRecommender(GenericRecommender):
    def __init__(self, k: int, additions: int, top_m: int, dataset) -> None:
        ic("cr_rec.__init__()")
        
        super().__init__(k, dataset)
        
        self.additions = additions
        self.top_m = top_m
        
        
    def train_co_rec(self):
        
        # step 1: datasets
        null_entries = []
        for user_id in self.user_train_ratings.keys():
             for unseen_item in self.get_user_unrated_items(user_id, self.additions):
                null_entries.append({user_id : unseen_item})
        
        # step 2: co-training
        self.user_rec = UserRecommender(self.k, copy.deepcopy(self.dataset))    
        self.item_rec = ItemRecommender(self.k, copy.deepcopy(self.dataset))        
        
        train_unlabelled_users = copy.deepcopy(null_entries)
        train_unlabelled_items = copy.deepcopy(null_entries)
        
        while (not train_unlabelled_users) and (not train_unlabelled_items):      
            predicted_user_ratings = []
            predicted_item_ratings = []
            
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_users):
                user_id = entry['user_id']
                item_id = entry['item_id']
                
                print("\nRunning USER Recommender for CoRec\n")          
                predicted_rating = self.user_rec.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
                
                
                if predicted_rating < 1.0:
                    predicted_rating = 1.0
                    
                if predicted_rating > 5:
                    predicted_rating = 5.0
                
                entry["pred_rating"] = predicted_rating
                confidence = self.get_confidence_measure("user", user_id, item_id, predicted_rating)
                entry["confidence"] = confidence
                predicted_user_ratings.append(entry)
                
                if i > 100:
                    break
                
            top_m_user_predictions = sorted(predicted_user_ratings, key = lambda d: d["confidence"])[-self.top_m:] # get top m confident predictions
            print(top_m_user_predictions)
        
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_items):
                user_id = entry['user_id']
                item_id = entry['item_id']
                
                print("\nRunning ITEM Recommender for CoRec\n")          
                predicted_rating = self.item_rec.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
                
                if predicted_rating < 1.0:
                    predicted_rating = 1.0
                    
                if predicted_rating > 5:
                    predicted_rating = 5.0
                
                entry["pred_rating"] = predicted_rating
                confidence = self.get_confidence_measure("item", user_id, item_id, predicted_rating)
                entry["confidence"] = confidence
                predicted_item_ratings.append(entry)
                
                if i > 100:
                    break
                
                
            top_m_item_predictions = sorted(predicted_item_ratings, key = lambda d: d["confidence"])[-self.top_m:] # get top m confident predictions
            print(top_m_item_predictions)
            
            
            # get union of top_m_confident_user_predictions and top_m_confident_item_predictions
            # TODO what if duplicate???
            top_m_predictions = top_m_user_predictions + top_m_item_predictions
            
            # update unlabelled datasets train_unlabelled_users and train_unlabelled_items
            # TODO fix
            top_m_predictions_df = pd.DataFrame(top_m_predictions)
            top_m_predictions_df.drop(columns=["confidence", "pred_rating"], inplace = True)
            partial_top_m_predictions = top_m_predictions_df.T.to_dict().values() # does this work ???
                
            print(len(train_unlabelled_users))
            print(len(partial_top_m_predictions))
            train_unlabelled_users = list(set(train_unlabelled_users) - set(partial_top_m_predictions))
            train_unlabelled_items = list(set(train_unlabelled_items) - set(partial_top_m_predictions))
            print(len(train_unlabelled_users))

            # update labelled datasets to include the most confidents results of the other trainset
            self.user_rec.add_new_recommendations(top_m_item_predictions)
            self.item_rec.add_new_recommendations(top_m_user_predictions)
        

    def predict_co_rec_for_users(self, user_id, item_id):
        """step 3: Recommendation Task for Users"""
        
        predicted_rating = self.user_rec.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
        return predicted_rating
    
    
    def predict_co_rec_for_items(self, user_id, item_id):
        """step 3: Recommendation Task for Items"""
        
        predicted_rating = self.item_rec.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
        return predicted_rating
    

    def get_user_unrated_items(self, user_id: int,  number: int) -> list:
        """"""
        ic("bs_rec.get_user_unrated_items()")
        
        value = self.user_train_ratings[user_id]
        items_rated_in_train = set(list(value.keys()))
        items_rated_in_test = set()
        
        if user_id in self.user_test_ratings:
            items_rated_in_test = set(list(self.user_test_ratings[user_id].keys()))
        
        items_rated = items_rated_in_train.intersection(items_rated_in_test)
        null_entries = list(set(self.item_ids).difference(items_rated))
        null_entries = random.sample(null_entries, k = number)
        
        return null_entries
        
                
    def get_confidence_measure(self, algorithm, user_id, item_id, prediction): 
        trustworthiness = self.get_measure_of_trustworthiness(algorithm, user_id, item_id, prediction)
        Nu = self.get_Nu(algorithm, user_id)
        Ni = self.get_Ni(algorithm, item_id)
        
        return trustworthiness*Nu*Ni
        
        
    def get_measure_of_trustworthiness(self, algorithm, user_id, item_id, prediction): 
        baseline_estimate =  self.get_baseline_estimate(user_id, item_id)
        
        return abs(1/(baseline_estimate - prediction))
        
        
    def get_baseline_estimate(self, user_id, item_id):
        return self.mean_train_rating # TODO change to proper
    
    
    def get_Nu(self, algorithm, user_id):
        if algorithm == "user":
            return self.user_rec.dataset.get_user_popularity(user_id)
        
        if algorithm == "item":
            return self.item_rec.dataset.get_user_popularity(user_id)
    
    
    def get_Ni(self, algorithm, item_id):
        if algorithm == "user":
            return self.user_rec.dataset.get_item_popularity(item_id)
        
        if algorithm == "item":
            return self.item_rec.dataset.get_item_popularity(item_id)
