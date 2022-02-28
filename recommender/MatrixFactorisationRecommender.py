   
"""

"""


from icecream import ic
from recommender.GenericRecommender import GenericRecommender
import numpy as np


class MatrixFactorisationRecommender(GenericRecommender):
    def __init__(self, dataset = None, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        '''
        R: rating matrix
        P: |U| * K (User features matrix)
        Q: |D| * K (Item features matrix)
        K: latent features
        steps: iterations
        alpha: learning rate
        beta: regularization parameter'''
        
        
        self.R = kwargs["run_params"]["R"]
        self.P = kwargs["run_params"]["P"]
        self.Q = kwargs["run_params"]["Q"]
        self.K = kwargs["run_params"]["K"]
        self.steps = kwargs["run_params"]["steps"]
        self.alpha = kwargs["run_params"]["alpha"]
        self.beta = kwargs["run_params"]["beta"]

    
    def train(self, **kwargs):
        self.load_dataset(**self.kwargs["dataset_config"])
        
        self.Q = self.Q.T

        for step in range(self.steps):
            for i in range(len(self.R)):
                for j in range(len(self.R[i])):
                    if R[i][j] > 0:
                        # calculate error
                        eij = self.R[i][j] - np.dot(self.P[i,:],Q[:,j])

                        for k in range(K):
                            # calculate gradient with a and beta parameter
                            self.P[i][k] = self.P[i][k] + self.alpha * (2 * eij * self.Q[k][j] - self.beta * self.P[i][k])
                            self.Q[k][j] = self.Q[k][j] + self.alpha * (2 * eij * self.P[i][k] - self.beta * self.Q[k][j])

            eR = np.dot(self.P,self.Q)

            e = 0

            for i in range(len(self.R)):

                for j in range(len(self.R[i])):

                    if self.R[i][j] > 0:

                        e = e + pow(self.R[i][j] - np.dot(self.P[i,:],self.Q[:,j]), 2)

                        for k in range(self.K):

                            e = e + (self.beta/2) * (pow(self.P[i][k],2) + pow(self.Q[k][j],2))
            # 0.001: local minimum
            if e < 0.001:

                break

        return self.P, self.Q.T
    
    def get_single_prediction(self, nP, nQ):
        nR = np.dot(nP, nQ.T)

        return nR

