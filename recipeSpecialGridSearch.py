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
from sklearn.decomposition import TruncatedSVD, NMF, PCA, \
            KernelPCA, LatentDirichletAllocation, FactorAnalysis

#import grid search utilities
from sklearn.model_selection import PredefinedSplit, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
"""
All possible combinations:
1.  Dimension Reduction:
    * PCA (TruncatedSVD)
    * KPCA (Test if works)
    * NMF
    * LDA
    * UMAP ???
    * None
2.  Recommendation Algorithmms
    * kNN

I could try different choosing algorithms as well - currently just using most common
"""
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
    y_blank = np.zeros((U.shape[0], 2))
    y_tog = np.concatenate([y, y_blank])

    test_fold = np.concatenate([
        # The training data.
        np.full(user_test.shape[0], 0, dtype=np.int8),
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
    filename = f"results/recipe_{data}_{sc}.pkl"
    Urecipe = user_to_recipe(U[user_test], rating, tfidf=(data=="Rhat"))
    if data == "R":
        data = R
    elif data == "Rhat":
        data = Rhat

    if rdr == "MkNN":
        rdr_class = recipeMultipleKNN(log=profile, sc=sc, rating=rating)
        rdr_params = {"metric": ['minkowski', 'cosine'],
                    "n_neighbors": [2, 10, 50, 100]}

    ###### Actual Grid search is right here, we'll essentially do everything one at a time #############
    U_tog = sparse.vstack([U[user_test], U])

    dr_options = [TruncatedSVD(), NMF(solver='mu'), LatentDirichletAllocation(learning_method='online'), KernelPCA(eigen_solver="arpack")]
    dr_names = ["PCA", "NMF", "LDA", "KPCA"]
    dr_options = [KernelPCA(eigen_solver="arpack")]
    dr_names = ["KPCA"]

    #iterate through all dimension reducers as we go!
    for dr, dr_class in zip(dr_names, dr_options):
        with open(profile+".log", 'a+') as f:
            f.write(f"######### {dr} ############# \n")

        dr_params = {"n_components": [20, 40, 60, 80, 100],
                    }
        if dr == "KPCA":
            dr_params = {"kernel": ["linear", "poly", "rbf"], 
                    "n_components": [20, 60, 100]}

        
        best_score_ = -np.inf
        #iterate by hand through parameter grid
        for dr_param in ParameterGrid(dr_params):
            #transform recipe data
            dr_class.set_params(**dr_param)
            data_red = dr_class.fit_transform(data)

            #saved it to our class
            rdr_class.R = data_red

            #perform grid search on estimator
            gs = GridSearchCV(rdr_class, rdr_params, cv=cv, verbose=51, refit=False, n_jobs=n_jobs, pre_dispatch='n_jobs')
            with parallel_backend('ipyparallel'):
                gs.fit(U_tog, y_tog)

            #if it improved for this iteration of dimension reducing, keep it
            if gs.best_score_ > best_score_:
                best_score_ = gs.best_score_
                best_params_ = gs.best_params_
                best_dr_params = dr_param
                results = pd.DataFrame(gs.cv_results_)
        #open all data files
        scores = pd.read_pickle(filename)

        #save all data
        temp = results[results["params"]==best_params_]
        column = "MkNN_" + str(rating)
        scores[column][dr] = (best_score_, temp["mean_fit_time"].iloc[0], 
                        temp["mean_score_time"].iloc[0], {**best_params_, **best_dr_params})
        scores.to_pickle(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender System Grid Search")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--rating", type=int, default=5)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--rdr", type=str, default='MkNN')
    parser.add_argument("--profile", type=str, default="ipy_profile")
    parser.add_argument("--sc", type=str, default="int")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
