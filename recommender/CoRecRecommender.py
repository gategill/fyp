"""

"""


#from time import sleep
from icecream import ic
#from dataset.Dataset import Dataset
from recommender.GenericRecommender import GenericRecommender
from recommender.UserRecommender import UserRecommender
from recommender.ItemRecommender import ItemRecommender
from recommender.Similarities import Similarities
from evaluation.Evaluation import Evaluation
import random
import copy
import pandas as pd


class CoRecRecommender(GenericRecommender):
    def __init__(self, k: int, additions: int, top_m: int, dataset = None) -> None:
        ic("cr_rec.__init__()")
        
        super().__init__(k, dataset)
        
        self.additions = additions
        self.top_m = top_m
        
        
    def train_co_rec(self):
        ic("cr_rec.train_co_rec()")

        # step 1: datasets
        null_entries = []
        
        for user_id in self.user_train_ratings.keys():
            for unseen_item in self.get_user_unrated_items(user_id, self.additions):
                null_entries.append({"user_id": user_id, "unseen_item_id": unseen_item})            
        
        #ic(null_entries)
        # step 2: co-training
        self.user_rec = UserRecommender(self.k, copy.deepcopy(self.dataset))    
        self.item_rec = ItemRecommender(self.k, copy.deepcopy(self.dataset))        
        
        train_unlabelled_users = copy.deepcopy(null_entries)
        train_unlabelled_items = copy.deepcopy(null_entries)
        
        while (train_unlabelled_users) and (train_unlabelled_items): 
            ic("here") 
            predicted_user_ratings = []
            predicted_item_ratings = []
            
            print("\nRunning USER Recommender for CoRec\n")          
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_users):
                user_id = int(entry["user_id"])
                item_id = int(entry["unseen_item_id"])

                
                predicted_rating = self.user_rec.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id, similarity_function = Similarities.sim_sim)
                
                if predicted_rating < 1.0:
                    predicted_rating = 1.0
                    
                if predicted_rating > 5:
                    predicted_rating = 5.0
                
                entry["pred_rating"] = round(predicted_rating, 2)
                confidence = self.get_confidence_measure("user", user_id, item_id, predicted_rating)
                entry["confidence"] = confidence
                predicted_user_ratings.append(entry)
                
                if i > 100:
                    break
                
            top_m_user_predictions = sorted(predicted_user_ratings, key = lambda d: d["confidence"])[-self.top_m:] # get top m confident predictions
            #ic(top_m_user_predictions)
        
            print("\nRunning ITEM Recommender for CoRec\n")          
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_items):
                user_id = int(entry["user_id"])
                item_id = int(entry["unseen_item_id"])
                
                predicted_rating = self.item_rec.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id, similarity_function = Similarities.sim_sim)
                
                if predicted_rating < 1.0:
                    predicted_rating = 1.0
                    
                if predicted_rating > 5:
                    predicted_rating = 5.0
                
                entry["pred_rating"] = round(predicted_rating,2)
                confidence = self.get_confidence_measure("item", user_id, item_id, predicted_rating)
                entry["confidence"] = confidence
                predicted_item_ratings.append(entry)
                
                if i > 100:
                    break
                
                
            top_m_item_predictions = sorted(predicted_item_ratings, key = lambda d: d["confidence"])[-self.top_m:] # get top m confident predictions
            
            
            # get union of top_m_confident_user_predictions and top_m_confident_item_predictions
            # TODO what if duplicate???
            top_m_predictions = top_m_user_predictions + top_m_item_predictions
            #ic(top_m_predictions)

            # update unlabelled datasets train_unlabelled_users and train_unlabelled_items
            # TODO fix
            top_m_predictions_df = pd.DataFrame(top_m_predictions)
            ic(top_m_predictions_df.shape)

            train_unlabelled_users_df = pd.DataFrame(train_unlabelled_users)
            ic(train_unlabelled_users_df.shape)

            # fiddling with dataframes
            inner_df = pd.merge(train_unlabelled_users_df, top_m_predictions_df, on=["user_id", "unseen_item_id"], how = "left")
            inner_df.drop(columns = {"pred_rating_y", "confidence_y"}, inplace = True)
            inner_df.rename(columns= {"pred_rating_x" : "pred_rating", "confidence_x" : "confidence"}, inplace = True)
            #inner_df.dropna(inplace = True)
            
            #inner_df = train_unlabelled_users_df[~train_unlabelled_users_df.isin(top_m_predictions_df)].dropna()

            ic(inner_df.head())
            ic(inner_df.shape)
            #top_m_predictions_df.drop(columns=["confidence", "pred_rating"], inplace = True)
            #ic(top_m_predictions_df.head())
            train_unlabelled_users = inner_df.T.to_dict().values() # does this work ???
            train_unlabelled_items = inner_df.T.to_dict().values() # does this work ???
            #ic(partial_top_m_predictions)
            
            #ic(len(train_unlabelled_users))
            #ic(len(partial_top_m_predictions))
            #train_unlabelled_users = list(set(train_unlabelled_users) - set(partial_top_m_predictions))
            #train_unlabelled_items = list(set(train_unlabelled_items) - set(partial_top_m_predictions))
            print(len(train_unlabelled_users))

            # update labelled datasets to include the most confidents results of the other trainset
            self.user_rec.add_new_recommendations(top_m_item_predictions)
            self.item_rec.add_new_recommendations(top_m_user_predictions)
        

    def predict_co_rec_for_users(self, user_id, item_id) -> float:
        #ic("cr_rec.predict_co_rec_for_users()")

        """step 3: Recommendation Task for Users"""
        
        predicted_rating = self.user_rec.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
        
        if predicted_rating < 1.0:
            predicted_rating = 1.0
                
        if predicted_rating > 5.0:
            predicted_rating = 5.0
            
        return round(predicted_rating, 2)
    
    
    def predict_co_rec_for_items(self, user_id, item_id):
        #ic("cr_rec.predict_co_rec_for_items()")

        """step 3: Recommendation Task for Items"""
        
        predicted_rating = self.item_rec.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
        
        if predicted_rating < 1.0:
            predicted_rating = 1.0
                
        if predicted_rating > 5.0:
            predicted_rating = 5.0
            
        return round(predicted_rating, 2)
    

    def get_user_unrated_items(self, user_id: int,  number: int) -> list:
        """"""
        #ic("cr_rec.get_user_unrated_items()")
        
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
        #ic("cr_rec.get_confidence_measure()")

        trustworthiness = self.get_measure_of_trustworthiness(algorithm, user_id, item_id, prediction)
        Nu = self.get_Nu(algorithm, user_id)
        Ni = self.get_Ni(algorithm, item_id)
        
        return trustworthiness*Nu*Ni
        
        
    def get_measure_of_trustworthiness(self, algorithm, user_id, item_id, prediction):
        #ic("cr_rec.get_measure_of_trustworthiness()")
        epsilon = 0.01
        baseline_estimate =  self.get_baseline_estimate(user_id, item_id)
   
        return abs(1/(baseline_estimate - prediction + epsilon))
        

    def get_baseline_estimate(self, user_id, item_id):
        #ic("cr_rec.get_baseline_estimate()")
        mu = self.mean_train_rating
        bu = self.user_rec.get_user_mean_rating(user_id)
        bi = self.item_rec.get_item_mean_rating(item_id)
        
        #ic(bu)
        #ic(bi)
        
        if bu is None:
            bu = 0.0
            
        if bi is None:
            bi = 0.0
        
        #ic( mu + bu + bi)
        
        return mu # + bu + bi
    
    
    def get_Nu(self, algorithm, user_id):
        #ic("cr_rec.get_Nu()")

        if algorithm == "user":
            return self.user_rec.dataset.get_user_popularity(user_id)
        
        if algorithm == "item":
            return self.item_rec.dataset.get_user_popularity(user_id)
    
    
    def get_Ni(self, algorithm, item_id):
        #ic("cr_rec.get_Ni()")

        if algorithm == "user":
            return self.user_rec.dataset.get_item_popularity(item_id)
        
        if algorithm == "item":
            return self.item_rec.dataset.get_item_popularity(item_id)


