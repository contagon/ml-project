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
from recipeClasses import *
from userClasses import *
from sklearn.decomposition import TruncatedSVD, NMF, PCA, \
            KernelPCA, LatentDirichletAllocation, FactorAnalysis

#import grid search utilities
from sklearn.model_selection import PredefinedSplit, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline

def main(rdr, data, rating, sc, n_jobs, profile):
    with open(profile+".log", 'a+') as f:
            f.write(f"{data}, {rating}, {sc} \n")
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

    dr = NMF(n_components=10)
    data_red = dr.fit_transform(U)
    print("PCA performed")
    rdr = userCluster()
    print("Fitting")
    rdr.fit(data_red)
    print("Predicting")
    p = rdr.predict(data_red[user_test])
    print("Scoring")
    score = rdr.score(data_red[user_test], y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender System Grid Search")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--rating", type=int, default=5)
    parser.add_argument("--data", type=str, default="R")
    parser.add_argument("--rdr", type=str, default='MkNN')
    parser.add_argument("--profile", type=str, default="ipy_profile")
    parser.add_argument("--sc", type=str, default="int")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)