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
    def __init__(self, metric='minkowski', algorithm='brute', log="recknn", sc='int'):
        self.sc = sc
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


class recipeMultipleKNN(Recommender):
    def __init__(self, metric='minkowski', rating=5, R=None, algorithm='brute', log="recmultknn", sc='int', n_neighbors=5):
        self.sc = sc
        self.R = R
        self.metric      = metric
        self.algorithm   = algorithm
        self.n_neighbors = n_neighbors
        self.rating = rating
        self.estimator   = NearestNeighbors(metric=metric, algorithm=algorithm, n_neighbors=n_neighbors)
        self.log = log
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. recipeKNN started. metric={self.metric}.\n") 

    def fit(self, X, y=None):
        self.estimator.fit(self.R)
        return self

    def predict(self, X):
        recommendations = np.zeros((X.shape[0], REC))
        for i, u in enumerate(X):
            #find recipes each user has "rated"
            idx = u.nonzero()
            ratings = np.array(u[idx]).flatten()
            recipes = idx[1][ratings>=self.rating]
        
            #find their closest neighbors, making sure not to include self
            #if there isn't any, randomly recommend something
            if len(recipes) == 0:
                closest = np.random.choice(self.R.shape[0], size=5)
            else:
                closest = self.estimator.kneighbors(self.R[recipes])[1][:,1:].flatten()

            #find which ones were most frequent
            recipe_count = dict()
            for recipe in closest:
                recipe_count[recipe] = recipe_count.get(recipe, 0) + 1
            lst = [(count, recipe) for recipe, count in recipe_count.items()]

            tog = heapq.nlargest(REC, lst)
            if len(tog) < REC:
                recommendations[i] = np.array([k[1] for k in tog] + list(np.random.choice(self.R.shape[0], size=REC-len(tog))))
            else:
                recommendations[i] =  np.array([k[1] for k in tog[:REC]])

        return recommendations

    def score(self, X, y):
        score = super().score(X, y)
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. recipeKNN finished. metric={self.metric}, score={score}.\n") 
        return score
