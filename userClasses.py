from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from baseUtils import *
import heapq

U     = sparse.load_npz("data-cleaned/user_train.npz")
REC = 5

def recommend_freq_rating(user_idx):
    #get recipes and ratings
    idx = U[user_idx].nonzero()
    recipes = idx[1]
    ratings = np.array(U[user_idx][idx]).flatten()
    
    #iterate through counting and summing ratings
    recipe_count = dict()
    rating_count = dict()
    for recipe, rating in zip(recipes, ratings):
        recipe_count[recipe] = recipe_count.get(recipe, 0) + 1
        rating_count[recipe] = rating_count.get(recipe, 0) + rating

    #put all together into list (which will determine sorting)
    lst = [(count, rating / count, recipe) for (recipe, count), (_, rating) in zip(recipe_count.items(), rating_count.items())]
    lst = heapq.nlargest(REC*10, lst)
    tog = sorted(lst, reverse=True)

    #return
    if len(tog) < REC:
        return np.array([k[2] for k in tog] + [-1]*(REC-len(tog)))
    else:
        return np.array([k[2] for k in tog[:REC]])

def recommend_rating_freq(user_idx):
    #get recipes and ratings
    idx = U[user_idx].nonzero()
    recipes = idx[1]
    ratings = np.array(U[user_idx][idx]).flatten()
    
    #iterate through counting and summing ratings
    recipe_count = dict()
    rating_count = dict()
    for recipe, rating in zip(recipes, ratings):
        recipe_count[recipe] = recipe_count.get(recipe, 0) + 1
        rating_count[recipe] = rating_count.get(recipe, 0) + rating

    #put all together into list (which will determine sorting)
    lst = [(rating / count, count, recipe) for (recipe, count), (_, rating) in zip(recipe_count.items(), rating_count.items())]
    lst = heapq.nlargest(REC*10, lst)
    tog = sorted(lst, reverse=True)

    #return
    if len(tog) < REC:
        return np.array([k[2] for k in tog] + [-1]*(REC-len(tog)))
    else:
        return np.array([k[2] for k in tog[:REC]])

def recommend_sum(user_idx):
    #get recipes and ratings
    idx = U[user_idx].nonzero()
    recipes = idx[1]
    ratings = np.array(U[user_idx][idx]).flatten()
    
    #iterate through counting and summing ratings
    rating_count = dict()
    for recipe, rating in zip(recipes, ratings):
        rating_count[recipe] = rating_count.get(recipe, 0) + rating

    #put all together into list (which will determine sorting)
    lst = [(rating, recipe) for recipe, rating in rating_count.items()]
    lst = heapq.nlargest(REC*10, lst)
    tog = sorted(lst, reverse=True)

    #return
    if len(tog) < REC:
        return np.array([k[1] for k in tog] + [-1]*(REC-len(tog)))
    else:
        return np.array([k[1] for k in tog[:REC]]) 

class userKNN(Recommender):
    def __init__(self, n_neighbors=5, metric='minkowski', algorithm='brute', recommend=recommend_sum):
        self.n_neighbors = n_neighbors #add one to exclude ourself
        self.metric      = metric
        self.algorithm   = algorithm
        self.estimator   = NearestNeighbors(n_neighbors=self.n_neighbors+1, metric=metric, algorithm = algorithm)
        self.recommend = recommend

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        #find closest users
        idxs = self.estimator.kneighbors(X)[1][:,1:]

        #get recommendations based on those users
        return np.array([self.recommend(idx) for i, idx in enumerate(idxs)])

class userNNBall(Recommender):
    def __init__(self, radius=1, metric='minkowski', algorithm='brute', recommend=recommend_sum):
        self.radius    = radius
        self.metric    = metric
        self.algorithm = algorithm
        self.estimator = NearestNeighbors(radius=self.radius, metric=metric, algorithm = algorithm)
        self.recommend = recommend

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        #find closest users
        distances, idxs = self.estimator.radius_neighbors(X)

        #get recommendations based on those users
        rec_recipes = np.array([self.recommend(idx[distance!=0]) for i, (distance, idx) in enumerate(zip(distances, idxs))])

        return rec_recipes

class userCluster(Recommender):
    def __init__(self, algorithm='kmeans', recommend=recommend_sum, n_clusters=10):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        if algorithm == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters)
        elif algorithm == 'gmm':
            self.clusterer = GaussianMixture(n_components=n_clusters)
        elif algorithm == 'mincut':
            self.clusterer = SpectralClustering(n_clusters=n_clusters)
        self.recommend = recommend

    def fit(self, X, y=None):
        self.labels = self.clusterer.fit_predict(X)
        return self

    def predict(self, X):
        #find clusters of users
        labels = self.clusterer.predict(X)

        #get recommendations based on those users (all users in the same cluster)
        return np.array([self.recommend( np.argwhere(self.labels==label).flatten() ) for i, label in enumerate(labels)])

class userClusterKNN(Recommender):
    def __init__(self, algorithm='kmeans', n_neighbors=5, metric="minkowski", recommend=recommend_sum, n_clusters=10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        if algorithm == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters)
        elif algorithm == 'gmm':
            self.clusterer = GaussianMixture(n_components=n_clusters)
        elif algorithm == 'mincut':
            self.clusterer = SpectralClustering(n_clusters=n_clusters)
        self.knn         = NearestNeighbors(n_neighbors=self.n_neighbors+1, metric=self.metric)
        self.recommend = recommend

    def fit(self, X, y=None):
        self.X = X
        self.labels = self.clusterer.fit_predict(X)
        return self

    def predict(self, X):
        #find closest users
        labels = self.clusterer.predict(X)

        rec_recipes = np.zeros((X.shape[0], REC), dtype='int')
        #iterate through each user's cluster
        for i, (label, x) in enumerate(zip(labels, X)):
            #find all recipes in cluster
            cluster = np.argwhere(self.labels==label).flatten()

            #find k nearest neighbors in the cluster (unless there isn't enough of them)
            if len(cluster) > self.n_neighbors+1:
                idx = self.knn.fit(self.X[cluster]).kneighbors(x.reshape(1,-1))[1][0,1:]
            else:
                idx = cluster
                
            #get recommended recipes
            rec_recipes[i]   = self.recommend(idx)

        return rec_recipes