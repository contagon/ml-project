from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from baseUtils import *

U     = sparse.load_npz("data-cleaned/user_train.npz")

class userKNN(Recommender):
    def __init__(self, n_neighbors=5, metric='minkowski', algorithm='brute'):
        self.n_neighbors = n_neighbors #add one to exclude ourself
        self.metric      = metric
        self.algorithm   = algorithm
        self.estimator   = NearestNeighbors(n_neighbors=self.n_neighbors+1, metric=metric, algorithm = algorithm)

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        #find closest users
        idxs = self.estimator.kneighbors(X)[1][:,1:]

        rec_recipes = np.zeros((X.shape[0], 5), dtype='int')
        #iterate through each user's closest users
        for i, idx in enumerate(idxs):
            #find their liked recipes
            close_recipes = U[idx].nonzero()[1]
            #and find 5 most common ones to recommend
            rec_recipe   = Counter(close_recipes).most_common(5)
            #sometimes there's not 5 recommendations - make due
            if len(rec_recipe) < 5:
                rec_recipes[i] = np.array([k[0] for k in rec_recipe] + [-1]*(5-len(rec_recipe)))
            else:
                rec_recipes[i]   = np.array([k[0] for k in rec_recipe][:5])

        return rec_recipes

class userNNBall(Recommender):
    def __init__(self, radius=1, metric='minkowski', algorithm='brute'):
        self.radius    = radius
        self.metric    = metric
        self.algorithm = algorithm
        self.estimator = NearestNeighbors(radius=self.radius, metric=metric, algorithm = algorithm)

    def fit(self, X, y=None):
        self.estimator.fit(X)
        return self

    def predict(self, X):
        #find closest users
        distances, idxs = self.estimator.radius_neighbors(X)

        rec_recipes = np.zeros((X.shape[0], 5), dtype='int')
        #iterate through each user's closest users
        for i, (distance, idx) in enumerate(zip(distances, idxs)):
            #find their liked recipes (making sure to exclude ourself)
            close_recipes = U[idx[distance!=0]].nonzero()[1]
            #and find 5 most common ones to recommend
            rec_recipe   = Counter(close_recipes).most_common(5)
            #sometimes there's not 5 recommendations - make due
            if len(rec_recipe) < 5:
                rec_recipes[i] = np.array([k[0] for k in rec_recipe] + [-1]*(5-len(rec_recipe)))
            else:
                rec_recipes[i]   = np.array([k[0] for k in rec_recipe][:5])

        return rec_recipes

class userCluster(Recommender):
    def __init__(self, clusterer='kmeans', **kwargs):
        if clusterer == 'kmeans':
            self.clusterer = KMeans(**kwargs)
        elif clusterer == 'gmm':
            self.clusterer = GaussianMixture(**kwargs)
        elif clusterer == 'mincut':
            self.clusterer = SpectralClustering(**kwargs)

    def fit(self, X, y=None):
        self.labels = self.clusterer.fit_predict(X)
        return self

    def predict(self, X):
        #find closest users
        labels = self.clusterer.predict(X)

        rec_recipes = np.zeros((X.shape[0], 5), dtype='int')
        #iterate through each user's cluster
        for i, label in enumerate(labels):
            #find all recipes in cluster
            idx = np.argwhere(self.labels==label).flatten()
            #find their liked recipes
            close_recipes = U[idx].nonzero()[1]
            #and find 5 most common ones to recommend
            rec_recipe   = Counter(close_recipes).most_common(5)
            #sometimes there's not 5 recommendations - make due
            if len(rec_recipe) < 5:
                rec_recipes[i] = np.array([k[0] for k in rec_recipe] + [-1]*(5-len(rec_recipe)))
            else:
                rec_recipes[i]   = np.array([k[0] for k in rec_recipe][:5])

        return rec_recipes

class userClusterKNN(Recommender):
    def __init__(self, clusterer='kmeans', n_neighbors=5, **kwargs):
        self.n_neighbors = n_neighbors
        if clusterer == 'kmeans':
            self.clusterer = KMeans(**kwargs)
        elif clusterer == 'gmm':
            self.clusterer = GaussianMixture(**kwargs)
        elif clusterer == 'mincut':
            self.clusterer = SpectralClustering(**kwargs)
        self.knn         = NearestNeighbors(n_neighbors=self.n_neighbors+1)

    def fit(self, X, y=None):
        self.X = X
        self.labels = self.clusterer.fit_predict(X)
        return self

    def predict(self, X):
        #find closest users
        labels = self.clusterer.predict(X)

        rec_recipes = np.zeros((X.shape[0], 5), dtype='int')
        #iterate through each user's cluster
        for i, (label, x) in enumerate(zip(labels, X)):
            #find all recipes in cluster
            cluster = np.argwhere(self.labels==label).flatten()

            #find k nearest neighbors in the cluster (unless there isn't enough of them)
            if len(cluster) > self.n_neighbors+1:
                idx = self.knn.fit(self.X[cluster]).kneighbors(x.reshape(1,-1))[1][0,1:]
            else:
                idx = cluster
                
            #find their liked recipes
            close_recipes = U[idx].nonzero()[1]
            #and find 5 most common ones to recommend
            rec_recipe   = Counter(close_recipes).most_common(5)
            #sometimes there's not 5 recommendations - make due
            if len(rec_recipe) < 5:
                rec_recipes[i] = np.array([k[0] for k in rec_recipe] + [-1]*(5-len(rec_recipe)))
            else:
                rec_recipes[i]   = np.array([k[0] for k in rec_recipe][:5])

        return rec_recipes

