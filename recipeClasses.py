from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from baseUtils import *
import heapq
from datetime import datetime

R = sparse.load_npz("data-cleaned/recipes.npz")
col_sum = np.load("data-cleaned/recipe_col_sum.npy")
num_items = 8576
num_rec = 178265
REC = 5

def user_to_recipe(users, rating, tfidf=False):
    row_users = []
    column_item = []
    counts = []
    row_sum = np.zeros(users.shape[0])
    for i, user in enumerate(users):
        #get highly rated recipes
        idx = user.nonzero()
        ratings = np.array(user[idx]).flatten()
        recipes = idx[1][ratings>=rating]

        #compile those into ingredients
        ingredients = list(set(R[recipes].nonzero()[1]))
        row_sum[i] += len(ingredients)

        #add into lists
        row_users += [i]*len(ingredients)
        column_item += ingredients
        counts += [1]*len(ingredients)

    if tfidf:
        #do the scaling
        for i, (row, col) in enumerate(zip(row_users, column_item)):
            #don't rescale the calories/minutes yet
            if col == num_items or col == num_items-1:
                continue
            counts[i] = (counts[i] / row_sum[row])*np.log(num_rec/col_sum[col])

    return sparse.csr_matrix((counts, [row_users, column_item]), shape=(users.shape[0], num_items), dtype=np.float)

class recipeKNN(Recommender):
    def __init__(self, metric='minkowski', algorithm='brute', log="knn"):
        self.metric      = metric
        self.algorithm   = algorithm
        self.estimator   = NearestNeighbors(n_neighbors=REC, metric=metric, algorithm = algorithm)
        self.log = log
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. recipeKNN started. metric={self.metric}.\n") 

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        #find recipes each user has "rated", along with their rating
        return self.estimator.kneighbors(X)[1]

    def score(self, X, y):
        score = super().score(X, y)
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. recipeKNN finished. metric={self.metric}, score={score}.\n") 
        return score