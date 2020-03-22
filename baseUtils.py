import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator
from scipy import sparse

X     = sparse.load_npz("data-cleaned/recipes.npz")

def recipe_score_int(i, js):
    score = np.zeros_like(js)
    #iterate through all recommendations
    for n, j in enumerate(js):
        #if didn't make enough recommendations, counts as 0
        if j == -1:
            continue
        temp = X[i, :-1] + X[j, :-1]
        score[n] = np.count_nonzero(temp.data==2) + int(X[i,-1]==X[j,-1])
    return score

def recipe_score_com(i, js):
    score = np.zeros_like(js)
    #iterate through all recommendations
    for n, j in enumerate(js):
        #if didn't make enough recommendations, counts as 0
        if j == -1:
            continue
        temp = X[i, :-1] + X[j, :-1]
        score[n] = np.count_nonzero(temp.data==1) + int(X[i,-1]!=X[j,-1])
    return -score

def recommend_scoring(y_true, y_pred, sc):
    max_scores = np.full(y_pred.shape, -np.inf)
    #iterate through all data
    for i, (yi, y_predi) in enumerate(zip(y_true, y_pred)):
        #iterate through each liked recipe to find closest
        for l in yi:
            #if they didn't like enough, skip the blank spots
            if l == -1:
                continue
            if sc == 'int':
                max_scores[i] = np.maximum(max_scores[i], recipe_score_int(l, y_predi))
            elif sc == 'com':
                max_scores[i] = np.maximum(max_scores[i], recipe_score_com(l, y_predi))

    return max_scores.mean(axis=1)

class Recommender(BaseEstimator):
    def predict(self, X, y=None):
        pass

    def score(self, X, y):
        return recommend_scoring(y, self.predict(X), self.sc).mean()
