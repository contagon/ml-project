#load all data
from scipy import sparse
R     = sparse.load_npz("data-cleaned/recipes.npz")
Rhat  = sparse.load_npz("data-cleaned/recipes_tfidf.npz")
U     = sparse.load_npz("data-cleaned/user_train.npz")
Uhat  = sparse.load_npz("data-cleaned/user_train_tfidf.npz")
Utest = sparse.load_npz("data-cleaned/user_test.npz")

#for multi-processing stuff
#from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
from joblib import Parallel, parallel_backend, register_parallel_backend

from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend

#misc imports
import pandas as pd
import argparse

#import all estimators
from userClasses import *
from sklearn.decomposition import TruncatedSVD, NMF, PCA, \
            KernelPCA, LatentDirichletAllocation, FactorAnalysis

#import grid search utilities
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
"""
All possible combinations:
1.  Dimension Reduction:
    * PCA (TruncatedSVD)
    * KPCA (Test if works)
    * NMF
    * SparsePCA
    * LDA
    * PPCA
    * UMAP ???
    * None
2.  Recommendation Algorithmms
    * kNN
    * NNBall
    * Cluster
        * Kmeans
        * GMM
        * MinCut
    * ClusterNN

I could try different choosing algorithmms as well - currently just using most common
"""
def main(rdr, data, sc, rec, n_jobs, profile):
    with open(profile+".log", 'a+') as f:
            f.write(f"{rdr}, {data}, {sc} \n")
    #prepare testing data and splitting data
    user_test = Utest[:,-1].toarray().flatten().astype('int')
    y = np.zeros((user_test.shape[0], 2), dtype='int')-1
    for i in range(len(y)):
        recipes = Utest[i].nonzero()[1][:-1]
        if len(recipes) == 1:
            y[i,0] = recipes
        elif len(recipes) == 2:
            y[i,:] = recipes
        else:
            raise ValueError("Someone reviewed 3 recipes!")
    y_blank = np.zeros((U.shape[0], 2), dtype='int')
    y_tog = np.concatenate([y, y_blank])

    test_fold = np.concatenate([
        # The training data.
        np.full(user_test.shape[0], 0, dtype='int'),
        # The development data.
        np.full(U.shape[0], -1, dtype=np.int8)
    ])
    cv = PredefinedSplit(test_fold)
    
    ###### GET ALL BACKUP PROCESSES READY AND RUNNING
    c = Client(profile=profile)
    print(c.ids)
    bview = c.load_balanced_view()

    register_parallel_backend('ipyparallel', lambda : IPythonParallelBackend(view=bview))

    ###### CHANGE EVERYTHING THAT MIGHT NEED TO BE TWEAKED HERE ########
    filename = f"results/user_{data}_{sc}.pkl"
    if data == "U":
        data = U
    elif data == "Uhat":
        data = Uhat

    if rec == "sum":
        recommend = recommend_sum
    elif rec == "fr":
        recommend = recommend_freq_rating
    elif rec == "rf":
        recommend = recommend_rating_freq

    if rdr == "kNN":
        rdr_class = userKNN(recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__metric": ['minkowski', 'cosine'],
                        "rdr__n_neighbors": [2, 10, 50, 100]}
    elif rdr == "NNBall":
        rdr_class = userNNBall(recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__radius": [0.5, 1, 3, 5]}
    elif rdr == "KMeans":
        rdr_class = userCluster(algorithm='kmeans', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__n_clusters": [10, 30, 50, 100]}
    elif rdr == "MinCut":
        rdr_class = userCluster(algorithm='mincut', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__n_clusters": [10, 30, 50, 100]}
    elif rdr == "GMM":
        rdr_class = userCluster(algorithm='gmm', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__n_clusters": [10, 30, 50, 100]}
    elif rdr == "DBSCAN":
        rdr_class = userCluster(algorithm='dbscan', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__eps": [.25, .5, .75, 1],
                    "rdr__min_samples": [2, 5, 10, 20]}
    elif rdr == "KMeansNN":
        rdr_class = userClusterKNN(algorithm='kmeans', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__n_clusters": [10, 30, 50, 100],
                        "rdr__metric": ['minkowski', 'cosine'],
                        "rdr__n_neighbors": [2, 10, 50, 100]}
    elif rdr == "MinCutNN":
        rdr_class = userClusterKNN(algorithm='mincut', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__n_clusters": [10, 30, 50, 100],
                        "rdr__metric": ['minkowski', 'cosine'],
                        "rdr__n_neighbors": [2, 10, 50, 100]}
    elif rdr == "GMMNN":
        rdr_class = userClusterKNN(algorithm='gmm', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__n_clusters": [10, 30, 50, 100],
                        "rdr__metric": ['minkowski', 'cosine'],
                        "rdr__n_neighbors": [2, 10, 50, 100]}
    elif rdr == "DBSCANNN":
        rdr_class = userClusterKNN(algorithm='dbscan', recommend=recommend, log=profile, sc=sc)
        rdr_params = {"rdr__n_clusters": [10, 30, 50, 100],
                    "rdr__eps": [.25, .5, .75, 1],
                    "rdr__min_samples": [2, 5, 10, 20]}

    ###### Actual Grid search is right here, we'll essentially do everything one at a time #############
    U_tog = sparse.vstack([data[user_test], data])

    dr_options = [TruncatedSVD(), MyNMF(solver='mu'), LatentDirichletAllocation(learning_method='online'), KernelPCA(eigen_solver="arpack")]
    dr_names = ["PCA", "NMF", "LDA", "KPCA"]
    dr_options = [LatentDirichletAllocation(learning_method='online')]
    dr_names = ["LDA"]

    #iterate through all dimension reducers as we go!
    for dr, dr_class in zip(dr_names, dr_options):
        with open(profile+".log", 'a+') as f:
            f.write(f"######### {dr} ############# \n")
        
        pipe = Pipeline([("dr", dr_class),
                        ("rdr", rdr_class)])

        params = {"dr__n_components": [20, 40, 60, 80, 100],
                    **rdr_params
                    }
        if dr == "KPCA":
            params = {"dr__kernel": ["linear", "poly", "rbf"], 
                    "dr__n_components": [20, 60, 100],
                    **rdr_params}

        gs = GridSearchCV(pipe, params, cv=cv, verbose=51, refit=False, n_jobs=n_jobs, pre_dispatch='n_jobs')
        with parallel_backend('ipyparallel'):
            gs.fit(U_tog, y_tog)
        results = pd.DataFrame(gs.cv_results_)

        #open all data files
        scores = pd.read_pickle(filename)

        #save all data
        temp = results[results["params"]==gs.best_params_]
        scores[rdr][dr] = (gs.best_score_, temp["mean_fit_time"].iloc[0], 
                        temp["mean_score_time"].iloc[0], gs.best_params_)
        scores.to_pickle(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender System Grid Search")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--rec", type=str, default='sum')
    parser.add_argument("--rdr", type=str, required=True)
    parser.add_argument("--profile", type=str, default="ipy_profile")
    parser.add_argument("--sc", type=str, default="int")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
