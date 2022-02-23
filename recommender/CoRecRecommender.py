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

ic.disable()

class CoRecRecommender(GenericRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        ic("cr_rec.__init__()")
        
        super().__init__(dataset, **kwargs)
        
        self.additions = kwargs["run_params"]["additions"]
        self.top_m = kwargs["run_params"]["top_m"]
        
        
    def train_co_rec(self):
        ic("cr_rec.train_co_rec()")

        # step 1: datasets
        null_entries = []
        
        for user_id in self.user_train_ratings.keys():
            for unseen_item in self.get_user_unrated_items(user_id, self.additions):
                null_entries.append({"user_id": user_id, "item_id": unseen_item})            
        
        #ic(null_entries)
        # step 2: co-training
        self.user_rec = UserRecommender(copy.deepcopy(self.dataset), **self.kwargs)    
        self.item_rec = ItemRecommender(copy.deepcopy(self.dataset), **self.kwargs)        
        
        train_unlabelled_users = copy.deepcopy(null_entries)
        train_unlabelled_items = copy.deepcopy(null_entries)
        
        while (train_unlabelled_users) and (train_unlabelled_items): 
            #ic("here") 
            print("There are {} ratings in the USER dataset".format(self.user_rec.dataset.num_ratings))
            print("There are {} ratings in the ITEM dataset".format(self.item_rec.dataset.num_ratings))
                        
            predicted_user_ratings = []
            predicted_item_ratings = []
            
            print("Running USER Recommender for CoRec")          
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_users):
                try:
                    user_id = int(entry["user_id"])
                    item_id = int(entry["item_id"])

                    
                    predicted_rating = self.user_rec.predict_rating_user_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
                    
                    if predicted_rating < 1.0:
                        predicted_rating = 1.0
                        
                    if predicted_rating > 5:
                        predicted_rating = 5.0
                    
                    entry["rating"] = round(predicted_rating, 2)
                    confidence = self.get_confidence_measure("user", user_id, item_id, predicted_rating)
                    entry["confidence"] = confidence
                    predicted_user_ratings.append(entry)
                    
                    if self.kwargs["exp_config"]["early_stop"]:
                        if i > 100:
                            break
                
                except KeyboardInterrupt:
                    ic("\nStopping\n")
                    ic(i)
                    break
            
            top_m_user_predictions = sorted(predicted_user_ratings, key = lambda d: d["confidence"])[-self.top_m:] # get top m confident predictions
            #ic(top_m_user_predictions)
        
            print("Running ITEM Recommender for CoRec")          
            # train item rec, predict and get confident results
            for i, entry in enumerate(train_unlabelled_items):
                try:
                    user_id = int(entry["user_id"])
                    item_id = int(entry["item_id"])
                    
                    predicted_rating = self.item_rec.predict_rating_item_based_nn_wtd(active_user_id = user_id, candidate_item_id = item_id)
                    
                    if predicted_rating < 1.0:
                        predicted_rating = 1.0
                        
                    if predicted_rating > 5:
                        predicted_rating = 5.0
                    
                    entry["rating"] = round(predicted_rating,2)
                    confidence = self.get_confidence_measure("item", user_id, item_id, predicted_rating)
                    entry["confidence"] = confidence
                    predicted_item_ratings.append(entry)
                    
                    if self.kwargs["exp_config"]["early_stop"]:
                        if i > 100:
                            break
                    
                except KeyboardInterrupt:
                    ic("\nStopping\n")
                    ic(i)
                    break
                
            top_m_item_predictions = sorted(predicted_item_ratings, key = lambda d: d["confidence"])[-self.top_m:] # get top m confident predictions
            
            # get union of top_m_confident_user_predictions and top_m_confident_item_predictions
            top_m_predictions = top_m_user_predictions + top_m_item_predictions
                      
            for i in top_m_predictions:
                if i in train_unlabelled_users:
                    train_unlabelled_users.remove(i)
                    
                    
            for i in top_m_predictions:
                if i in train_unlabelled_items:
                    train_unlabelled_items.remove(i)
                    
            #ic(top_m_predictions)

            # update unlabelled datasets train_unlabelled_users and train_unlabelled_items
            print("There are " + str(len(train_unlabelled_users)) + " ratings left in train_unlabelled_users")
            if len(train_unlabelled_users):
                train_unlabelled_users_df = pd.DataFrame(train_unlabelled_users)
                train_unlabelled_users_df.drop(columns=["confidence", "rating"], inplace = True)
                train_unlabelled_users = list(train_unlabelled_users_df.T.to_dict().values()) # does this work ???
                #ic(train_unlabelled_users[:10])
            else:
                ic("Length of train_unlabelled_users is 0")  
                          
            print("There are " + str(len(train_unlabelled_items)) + " ratings left in train_unlabelled_items")
            if len(train_unlabelled_items) > 0:
                train_unlabelled_items_df = pd.DataFrame(train_unlabelled_items)
                train_unlabelled_items_df.drop(columns=["confidence", "rating"], inplace = True)
                train_unlabelled_items = list(train_unlabelled_items_df.T.to_dict().values()) # does this work ???
            else:
                ic("Length of train_unlabelled_items is 0")
                
            top_m_user_predictions_df = pd.DataFrame(top_m_user_predictions)
            max_user_confidence = top_m_user_predictions_df["confidence"].max()
            top_m_user_predictions_df.drop(columns=["confidence"], inplace = True)
            top_m_user_predictions_df.drop_duplicates(inplace = True)
            top_m_user_predictions = list(top_m_user_predictions_df.T.to_dict().values()) # does this work ???
            
            top_m_item_predictions_df = pd.DataFrame(top_m_item_predictions)
            max_item_confidence = top_m_item_predictions_df["confidence"].max()

            top_m_item_predictions_df.drop(columns=["confidence"], inplace = True)
            top_m_item_predictions_df.drop_duplicates(inplace = True)
            top_m_item_predictions = list(top_m_item_predictions_df.T.to_dict().values()) # does this work ???

            self.max_user_confidence = max_user_confidence
            self.max_item_confidence = max_item_confidence
            
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
        
        if baseline_estimate - prediction == 0.0:
            #print("DIVIDING BY 0")
            return abs(1/epsilon)

        else:
            return abs(1/(baseline_estimate - prediction))
        

    def get_baseline_estimate(self, user_id, item_id):
        #ic("cr_rec.get_baseline_estimate()")
        mu = self.mean_train_rating
        bu = self.user_rec.get_user_mean_rating(user_id)
        bi = self.item_rec.get_item_mean_rating(item_id)
        
        if bu is None:
            bu = 0.0
            
        if bi is None:
            bi = 0.0

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


