import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator
from scipy import sparse

X     = sparse.load_npz("data-cleaned/recipes.npz")


def recipe_score(i, js):
    score = np.zeros_like(js)
    #iterate through all recommendations
    for n, j in enumerate(js):
        #if didn't make enough recommendations, counts as 0
        if j == -1:
            continue
        temp = X[i, :-1] + X[j, :-1]
        score[n] = np.count_nonzero(temp.data==2) + int(X[i,-1]==X[j,-1])
    return score

def recommend_scoring(y_true, y_pred):
    max_scores = np.zeros_like(y_pred)
    #iterate through all data
    for i, (yi, y_predi) in enumerate(zip(y_true, y_pred)):
        #iterate through each liked recipe to find closest
        for l in yi:
            if l == -1:
                continue
            max_scores[i] = np.maximum(max_scores[i], recipe_score(l, y_predi))
            
    return max_scores.mean(axis=1)

class Recommender(BaseEstimator):
    def predict(self, X, y=None):
        pass

    def score(self, X, y):
        return recommend_scoring(y, self.predict(X)).mean()