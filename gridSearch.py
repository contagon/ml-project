#load all data
from scipy import sparse
X     = sparse.load_npz("data-cleaned/recipes.npz")
Xhat  = sparse.load_npz("data-cleaned/recipes_tfidf.npz")
U     = sparse.load_npz("data-cleaned/user_train.npz")
Uhat  = sparse.load_npz("data-cleaned/user_train_tfidf.npz")
Utest = sparse.load_npz("data-cleaned/user_test.npz")

#misc imports
import pandas as pd

#import all estimators
from userClasses import *
from sklearn.decomposition import TruncatedSVD, NMF, PCA, \
            KernelPCA, LatentDirichletAllocation, FactorAnalysis

#import grid search utilities
from searchgrid import set_grid, make_grid_search, make_pipeline
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline

#prepare testing data and splitting data
user_test = Utest[:,-1].toarray().flatten().astype('int')
y = np.zeros((Utest.shape[0], 2), dtype='int')-1
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
    np.full(U[user_test].shape[0], 0, dtype=np.int8),
    # The development data.
    np.full(U.shape[0], -1, dtype=np.int8)
])
cv = PredefinedSplit(test_fold)

###### CHANGE U HERE TO Uhat to data ########
U_tog = sparse.vstack([U[user_test], U])


###### Actual Grid search is right here, we'll essentially do everything one at a time
# n_clusters = [10, 50, 100, 150]
rdr = "kNN"
dr_options = [TruncatedSVD(), KernelPCA(), NMF(), LatentDirichletAllocation()]
dr_names = ["PCA", "KPCA", "NMF", "LDA"]

#iterate through all dimension reducers as we go!
for dr, dr_class in zip(dr_names, dr_options):
    print(f"######### {dr} #############")
    
    pipe = Pipeline([("dr", dr_class),
                    ("rdr", userKNN())])

    params = {"dr__n_components": [20, 40, 60, 80],
                "rdr__n_neighbors": [2, 10, 50, 100],
                }

    gs = GridSearchCV(pipe, params, cv=cv, verbose=3)
    gs.fit(U_tog, y_tog)
    results = pd.DataFrame(gs.cv_results_)

    #open all data files
    scores = pd.read_pickle("results/user_U.pkl")

    #save all data
    temp = results[results["params"]==gs.best_params_]
    scores[rdr][dr] = (gs.best_score_, temp["mean_fit_time"].iloc[0], 
                    temp["mean_score_time"].iloc[0], gs.best_params_)
    scores.to_pickle("results/user_U.pkl")

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
2.  Recommendation Algorithms
    * kNN
    * NNBall
    * Cluster
        * Kmeans
        * GMM
        * MinCut
    * ClusterNN

I could try different choosing algorithms as well - currently just using most common
"""