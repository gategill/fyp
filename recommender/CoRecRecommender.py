"""

"""


from icecream import ic
from recommender.GenericRecommender import GenericRecommender
from recommender.UserKNNRecommender import UserKNNRecommender
from recommender.ItemKNNRecommender import ItemKNNRecommender
from evaluation.Evaluation import Evaluation
import random
import copy
import pandas as pd
import math


class CoRecRecommender(GenericRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        ic("cr_rec.__init__()")
        
        super().__init__(dataset, **kwargs)
        
        self.additions = kwargs["run_params"]["additions"]
        self.top_m = kwargs["run_params"]["top_m"]
        

    def train(self):
        self.load_dataset(**self.kwargs["dataset_config"])
        self.co_training()
        
        
    def get_single_prediction(self, active_user_id, candidate_item_id):
        #return super().get_single_prediction(active_user_id, candidate_item_id)
        user_predicted_rating = self.predict_co_rec_for_users(active_user_id, candidate_item_id)
        item_predicted_rating = self.predict_co_rec_for_items(active_user_id, candidate_item_id)
        return user_predicted_rating, item_predicted_rating
    
    
    def get_predictions(self):
        for i, test in enumerate(self.test_ratings):
            try:
                user_id = int(test['user_id'])
                item_id = int(test['item_id'])
                
                user_predicted_rating, item_predicted_rating = self.get_single_prediction(active_user_id = user_id, candidate_item_id = item_id)
                    
                test["user_pred_rating"] = user_predicted_rating
                test["item_pred_rating"] = item_predicted_rating
                self.add_prediction(test)

                if self.kwargs["experiment_config"]["early_stop"]:
                    if i > 30:
                        break
                        
            except KeyboardInterrupt:
                break
        
        return test
    
    
    def evaluate_predictions(self, method = "MAE"):
        # ic("gen_rec.evaluate_predictions()")
        
        mae_user, mae_item = Evaluation.mean_absolute_error(self.predictions)

        mae_user = round(mae_user, 5)
        mae_item = round(mae_item, 5)
        
        return mae_user, mae_item
    
    
    def co_training(self):
        ic("cr_rec.train_co_rec()")

        # step 1: datasets
        unlabelled_entries = []
        
        for user_id in self.user_train_ratings.keys():
            for unseen_item in self.get_user_unrated_items(user_id, self.additions):
                unlabelled_entries.append({"user_id": user_id, "item_id": unseen_item})    
        
        # step 2: co-training
        self.user_rec = UserKNNRecommender(copy.deepcopy(self.dataset), **self.kwargs)
        self.item_rec = ItemKNNRecommender(copy.deepcopy(self.dataset), **self.kwargs) 
        
        self.user_rec.train()  
        self.item_rec.train()  
                
        train_unlabelled_users = copy.deepcopy(unlabelled_entries)
        train_unlabelled_items = copy.deepcopy(unlabelled_entries)
        
        old_num_users = len(train_unlabelled_users)
        old_num_items = len(train_unlabelled_items)
        
        while (train_unlabelled_users) and (train_unlabelled_items): 
            print("There are {} ratings in the USER dataset".format(self.user_rec.dataset.num_ratings))
            print("There are {} ratings in the ITEM dataset".format(self.item_rec.dataset.num_ratings))
                        
            predicted_user_ratings = []
            predicted_item_ratings = []
                         
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_users):
                try:
                    user_id = int(entry["user_id"])
                    item_id = int(entry["item_id"])

                    predicted_rating = self.user_rec.get_single_prediction(active_user_id = user_id, candidate_item_id = item_id)
                    
                    entry["rating"] = predicted_rating
                    confidence = self.get_confidence_measure("user", user_id, item_id, predicted_rating)
                    entry["confidence"] = round(confidence, 3)
                    predicted_user_ratings.append(entry)
                                        
                    if self.kwargs["experiment_config"]["early_stop"]:
                        if i > 100:
                            break
                
                except KeyboardInterrupt:
                    break
            
            # get top m confident predictions                    
            top_m_user_predictions = sorted(predicted_user_ratings, key = lambda d: d["confidence"])[-self.top_m:]
            
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_items):
                try:
                    user_id = int(entry["user_id"])
                    item_id = int(entry["item_id"])
                    
                    predicted_rating = self.item_rec.get_single_prediction(active_user_id = user_id, candidate_item_id = item_id)
                    
                    entry["rating"] = predicted_rating
                    confidence = self.get_confidence_measure("item", user_id, item_id, predicted_rating)
                    entry["confidence"] = round(confidence, 3)
                    predicted_item_ratings.append(entry)
                    
                    if self.kwargs["experiment_config"]["early_stop"]:
                        if i > 100:
                            break
                    
                except KeyboardInterrupt:
                    break
                
            # get top m confident predictions
            top_m_item_predictions = sorted(predicted_item_ratings, key = lambda d: d["confidence"])[-self.top_m:]
            
            # get union of top_m_confident_user_predictions and top_m_confident_item_predictions
            top_m_predictions = top_m_user_predictions + top_m_item_predictions
             
            try:         
                for ite in top_m_user_predictions:
                    if ite in train_unlabelled_users:
                        train_unlabelled_users.remove(ite)

                for ite in top_m_item_predictions:
                    if ite in train_unlabelled_items:
                        train_unlabelled_items.remove(ite)
        
                assert(old_num_users > len(train_unlabelled_users)), "ERROR: the number of unlabelled users is NOT decreasing"
                assert(old_num_items > len(train_unlabelled_items)), "ERROR: the number of unlabelled items is NOT decreasing"
                
                old_num_users -= len(train_unlabelled_users)
                old_num_items -= len(train_unlabelled_items)
            
            except AssertionError as msg:
                print(msg)
                break
     
            # update unlabelled datasets train_unlabelled_users and train_unlabelled_items
            # remove "confidence" and "rating" keys from list of dicts 
            print(str(len(train_unlabelled_users)) + " ratings left in train_unlabelled_users")
            if len(train_unlabelled_users) > 0:
                train_unlabelled_users_df = pd.DataFrame(train_unlabelled_users)
                if ("confidence" and "rating") in train_unlabelled_users_df.columns:
                    train_unlabelled_users_df.drop(columns=["confidence", "rating"], inplace = True)
                train_unlabelled_users = list(train_unlabelled_users_df.T.to_dict().values()) 
            else:
                print("length of train_unlabelled_users is 0")  
                         
            # remove "confidence" and "rating" keys from list of dicts 
            print(str(len(train_unlabelled_items)) + " ratings left in train_unlabelled_items\n")
            if len(train_unlabelled_items) > 0:
                train_unlabelled_items_df = pd.DataFrame(train_unlabelled_items)
                if ("confidence" and "rating") in train_unlabelled_items_df.columns:
                    train_unlabelled_items_df.drop(columns=["confidence", "rating"], inplace = True)
                train_unlabelled_items = list(train_unlabelled_items_df.T.to_dict().values()) 
            else:
                ic("length of train_unlabelled_items is 0")
                
            # remove confidence key from list of dicts
            top_m_user_predictions_df = pd.DataFrame(top_m_user_predictions)
            if "confidence" in top_m_user_predictions_df.columns:
                top_m_user_predictions_df.drop(columns=["confidence"], inplace = True)
            top_m_user_predictions_df.drop_duplicates(inplace = True)
            top_m_user_predictions = list(top_m_user_predictions_df.T.to_dict().values()) 
            
            # remove confidence key from list of dicts
            top_m_item_predictions_df = pd.DataFrame(top_m_item_predictions)
            if "confidence" in top_m_item_predictions_df.columns:
                top_m_item_predictions_df.drop(columns=["confidence"], inplace = True)
            top_m_item_predictions_df.drop_duplicates(inplace = True)
            top_m_item_predictions = list(top_m_item_predictions_df.T.to_dict().values()) 
            
            self.user_rec.add_new_recommendations(top_m_item_predictions) # add item to user
            self.item_rec.add_new_recommendations(top_m_user_predictions) # add user to item
            
            if (self.kwargs["experiment_config"]["early_stop"]) and (old_num_items < int(math.floor(0.9 * train_unlabelled_users))):
                break


    def predict_co_rec_for_users(self, user_id, item_id) -> float:
        #ic("cr_rec.predict_co_rec_for_users()")

        """step 3: Recommendation Task for Users"""
        predicted_rating = self.user_rec.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
        
        if predicted_rating < 1.0:
            predicted_rating = 1.0
                
        if predicted_rating > 5.0:
            predicted_rating = 5.0
            
        return round(predicted_rating, self.ROUNDING)
    
    
    def predict_co_rec_for_items(self, user_id, item_id):
        #ic("cr_rec.predict_co_rec_for_items()")

        """step 3: Recommendation Task for Items"""
        
        predicted_rating = self.item_rec.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
        
        if predicted_rating < 1.0:
            predicted_rating = 1.0
                
        if predicted_rating > 5.0:
            predicted_rating = 5.0
            
        return round(predicted_rating, self.ROUNDING)
    

    def get_user_unrated_items(self, user_id: int,  number: int) -> list:
        """"""
        #ic("cr_rec.get_user_unrated_items()")
        
        value = self.user_train_ratings[user_id]
        items_rated_in_train = set(list(value.keys()))
        items_rated_in_test = set()
        
        if user_id in self.user_test_ratings:
            items_rated_in_test = set(list(self.user_test_ratings[user_id].keys()))
        
        items_rated = items_rated_in_train.intersection(items_rated_in_test)
        unlabelled_entries = list(set(self.item_ids).difference(items_rated))
        unlabelled_entries = random.sample(unlabelled_entries, number)
        
        return unlabelled_entries
        
                
    def get_confidence_measure(self, algorithm, user_id, item_id, prediction): 
        #ic("cr_rec.get_confidence_measure()")

        trustworthiness = self.get_measure_of_trustworthiness(user_id, item_id, prediction)
        Nu = self.get_Nu(algorithm, user_id)
        Ni = self.get_Ni(algorithm, item_id)
        
        return trustworthiness*Nu*Ni
        
        
    def get_measure_of_trustworthiness(self, user_id, item_id, prediction):
        #ic("cr_rec.get_measure_of_trustworthiness()")
        
        epsilon = 0.001
        baseline_estimate =  self.get_baseline_estimate(user_id, item_id)
        
        if baseline_estimate - prediction == 0.0:
            return abs(1 / epsilon)

        else:
            return abs(1 / (baseline_estimate - prediction))
        

    def get_baseline_estimate(self, user_id, item_id):
        #ic("cr_rec.get_baseline_estimate()")
        
        mu = self.mean_train_rating
        bu = self.user_rec.get_user_mean_rating(user_id)
        bi = self.item_rec.get_item_mean_rating(item_id)
        
        if bu is None:
            bu = 0.0
            
        if bi is None:
            bi = 0.0

        return mu
    
    
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


