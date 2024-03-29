from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from baseUtils import *
import heapq
from datetime import datetime

U     = sparse.load_npz("data-cleaned/user_train.npz")
R     = sparse.load_npz("data-cleaned/recipes.npz")
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
        return np.array([k[2] for k in tog] + list(np.random.choice(R.shape[0], size=REC-len(tog))))
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
        return np.array([k[2] for k in tog] + list(np.random.choice(R.shape[0], size=REC-len(tog))))
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
        # return np.array([k[1] for k in tog] + [-1]*(REC-len(tog)))
        return np.array([k[1] for k in tog] + list(np.random.choice(R.shape[0], size=REC-len(tog))))
    else:
        return np.array([k[1] for k in tog[:REC]]) 

class userKNN(Recommender):
    def __init__(self, n_neighbors=5, metric='minkowski', algorithm='brute', recommend=recommend_sum, log="knn", sc='int'):
        self.sc = sc
        self.n_neighbors = n_neighbors #add one to exclude ourself
        self.metric      = metric
        self.algorithm   = algorithm
        self.estimator   = NearestNeighbors(n_neighbors=self.n_neighbors+1, metric=metric, algorithm = algorithm)
        self.recommend = recommend
        self.log = log
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. kNN started. n_neighbors={self.n_neighbors}, metric={self.metric}.\n") 

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        #find closest users
        idxs = self.estimator.kneighbors(X)[1][:,1:]

        #get recommendations based on those users
        return np.array([self.recommend(idx) for i, idx in enumerate(idxs)])

    def score(self, X, y):
        score = super().score(X, y)
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. kNN finished. n_neighbors={self.n_neighbors}, metric={self.metric}, score={score}.\n") 
        return score

class userNNBall(Recommender):
    def __init__(self, radius=1, metric='minkowski', algorithm='brute', recommend=recommend_sum, log='NNBall', sc='int'):
        self.sc = sc
        self.radius    = radius
        self.metric    = metric
        self.algorithm = algorithm
        self.estimator = NearestNeighbors(radius=self.radius, metric=metric, algorithm = algorithm)
        self.recommend = recommend
        self.log = log
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. NNBall started. radius={self.radius}, metric={self.metric}.\n") 

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        #find closest users
        distances, idxs = self.estimator.radius_neighbors(X)

        #get recommendations based on those users
        return np.array([self.recommend(idx[distance!=0]) for i, (distance, idx) in enumerate(zip(distances, idxs))])

    def score(self, X, y):
        score = super().score(X, y)
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. NNBall finished. radius={self.radius}, metric={self.metric}, score={score}.\n") 
        return score

class userCluster(Recommender):
    def __init__(self, algorithm='kmeans', recommend=recommend_sum, n_clusters=10, log='cluster', sc='int', eps=0.5, min_samples=5):
        self.sc = sc
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        #used for DBSCAN
        self.eps = eps
        self.min_samples = min_samples
        if algorithm == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters)
        elif algorithm == 'gmm':
            self.clusterer = GaussianMixture(n_components=n_clusters)
        elif algorithm == 'dbscan':
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == 'mincut':
            self.clusterer = SpectralClustering(n_clusters=n_clusters, eigen_solver='amg')
        self.recommend = recommend
        self.log = log
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. Cluster started. algo={self.algorithm}, n_clusters={self.n_clusters}.\n") 

    def fit(self, X, y=None):
        self.X = X
        self.labels = self.clusterer.fit_predict(X)
        return self

    def predict(self, X):
        #find clusters of users
        idx = []
        j = 0
        for i in range(self.X.shape[0]):
            if np.allclose(self.X[i], X[j]):
                idx.append(i)
                j += 1
            if j >= X.shape[0]:
                break
        idxs = np.array(idx, dtype='int')
        labels = self.labels[idxs]

        #get recommendations based on those users (all users in the same cluster)
        #make sure we don't include ourselves by using np.setdiff1d
        return np.array([self.recommend( np.setdiff1d( np.argwhere(self.labels==label).flatten(),  idx) ) for i, (idx, label) in enumerate(zip(idxs, labels))])

    def score(self, X, y):
        score = super().score(X, y)
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. Cluster started. algo={self.algorithm}, n_clusters={self.n_clusters}, score={score}.\n") 
        return score

class userClusterKNN(Recommender):
    def __init__(self, algorithm='kmeans', n_neighbors=5, metric="minkowski", recommend=recommend_sum, n_clusters=10, log='clusternn', sc='int',eps=0.5, min_samples=5):
        self.sc = sc
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        #used for DBSCAN
        self.eps = eps
        self.min_samples = min_samples
        if algorithm == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters)
        elif algorithm == 'gmm':
            self.clusterer = GaussianMixture(n_components=n_clusters)
        elif algorithm == 'dbscan':
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == 'mincut':
            self.clusterer = SpectralClustering(n_clusters=n_clusters, eigen_solver='amg')
        self.knn         = NearestNeighbors(n_neighbors=self.n_neighbors+1, metric=self.metric)
        self.recommend = recommend
        self.log = log
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. ClusterNN started. algo={self.algorithm}, n_clusters={self.n_clusters}, n_neighbors={self.n_neighbors}, metric={self.metric}.\n") 

    def fit(self, X, y=None):
        self.X = X
        self.labels = self.clusterer.fit_predict(X)
        return self

    def predict(self, X):
        #find closest users
         #find clusters of users
        idx = []
        j = 0
        for i in range(self.X.shape[0]):
            if np.allclose(self.X[i], X[j]):
                idx.append(i)
                j += 1
            if j >= X.shape[0]:
                break
        idxs = np.array(idx, dtype=np.int8)
        labels = self.labels[idxs]

        rec_recipes = np.zeros((X.shape[0], REC), dtype='int')
        #iterate through each user's cluster
        for i, (idx, label, x) in enumerate(zip(idx, labels, X)):
            #find all recipes in cluster
            cluster = np.argwhere(self.labels==label).flatten()

            #find k nearest neighbors in the cluster (unless there isn't enough of them)
            #in both cases make sure we don't include ourselves
            if len(cluster) > self.n_neighbors+1:
                rec = self.knn.fit(self.X[cluster]).kneighbors(x.reshape(1,-1))[1][0,1:]
            else:
                rec = np.setdiff1d(cluster, idx)
                
            #get recommended recipes
            rec_recipes[i]   = self.recommend(rec)

        return rec_recipes

    def score(self, X, y):
        score = super().score(X, y)
        with open(self.log+".log", 'a+') as f:
            f.write(f"{datetime.now()}. ClusterNN started. algo={self.algorithm}, n_clusters={self.n_clusters}, n_neighbors={self.n_neighbors}, metric={self.metric}, score={score}.\n") 
        return score

class MyNMF(NMF):
    def fit_transform(self, X, y=None, W=None, H=None):
        return self.fit(X).transform(X)
    def fit(self, X, y=None, **params):
        super().fit_transform(X, **params)
        return self