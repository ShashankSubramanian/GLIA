import matplotlib as mpl
# mpl.use('Agg')
import pandas as pd
import ntpath
import argparse
import shutil
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter
import scipy
from os import listdir
import numpy as np
import os
from tabulate import tabulate
from time import time
from pprint import pprint

import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC
# from sklearn import grid_search
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import manifold

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost import XGBClassifier


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def getSurvivalClass(x):
    m = x/30.
    if m < 10:
        return 0;# np.array([1,0,0])
    elif m < 15:
        return 1;# np.array([0,1,0])
    else:
        return 2;# np.array([0,0,1])


###
### ------------------------------------------------------------------------ ###
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def read_data(dir, level, file):
    FILTER = ['Brats18_CBICA_AUR_1', 'Brats18_CBICA_ANI_1']
    FAILED_TO_ADD = []
    max_l2c1error=0.8  # filter out 'failed cases'

    brats_data = pd.DataFrame();
    # read survival data
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True);
    survival_mean = np.mean(survival_data["Survival"])
    survival_var  = np.var(survival_data["Survival"])
    survival_std  = np.std(survival_data["Survival"])
    # survival_data.hist() #(column="Survival")
    # plt.show()
    print("Survival Data Statistics: mean %1.3f, variance: %1.3f, std: %1.3f" % (survival_mean, survival_var, survival_std));


    # read brats data
    fltr = 0
    if dir is not None:
        PATIENTS = os.listdir(dir);
        for P in PATIENTS:
            if not P.startswith('Brats'):
                continue;
            BID = str(P)
            if BID in FILTER:
                print(bcolors.FAIL + "   --> filtering ", BID , " due to errors in simulation." + bcolors.ENDC);
                fltr += 1;
                continue;
            print(bcolors.OKBLUE + "   ### processing", P , "###" + bcolors.ENDC)
            patient_path = os.path.join(dir, P);

            # read features
            try:
                patient_data = pd.read_csv(os.path.join(patient_path,"features_"+str(level)+".csv"), index_col=0, header = None, error_bad_lines=True, sep=';', skipinitialspace=True).transpose();
            except:
                print("failed to add {}".format(BID))
                FAILED_TO_ADD.append(BID)
            patient_data['BID'] = BID;
            survival_row = survival_data.loc[survival_data['BraTS18ID'] == BID]
            if not survival_row.empty:
                patient_data['age']              = float(survival_row.iloc[0]['Age']);                         # age
                patient_data['survival_class']   = getSurvivalClass(float(survival_row.iloc[0]['Survival']));  # survival class
                patient_data['survival(days)']   = float(survival_row.iloc[0]['Survival']);                    # survival
                patient_data['resection_status'] = str(survival_row.iloc[0]['ResectionStatus']) if (str(survival_row.iloc[0]['ResectionStatus']) != 'nan' and str(survival_row.iloc[0]['ResectionStatus']) != "NA") else "no";
            else:
                patient_data['age']              = -1;
                patient_data['survival_class']   = -1; #np.array([-1,-1,-1]);
                patient_data['survival(days)']   = -1;
                patient_data['resection_status'] = 'NA'
            if brats_data.empty:
                brats_data = patient_data;
            else:
                brats_data = brats_data.append(patient_data, ignore_index=True, sort=True);

        brats_data.to_csv(os.path.join(dir, "features_"+str(level)+".csv"))
        print("\nCould not add the following brains:\n", FAILED_TO_ADD);

    # read brats data
    if dir is None and file is not None:
        brats_data = pd.read_csv(os.path.join(basedir,file), header = 0, error_bad_lines=True, skipinitialspace=True)
        print("read brats simulation data of length %d" % len(brats_data))

    nbshort = len(brats_data.loc[brats_data['survival_class'] ==  0])
    nbmid   = len(brats_data.loc[brats_data['survival_class'] ==  1])
    nblong  = len(brats_data.loc[brats_data['survival_class'] ==  2])
    sum     = nbshort + nbmid + nblong
    weights = {}
    # weights[1] = 1. / nbshort;
    # weights[2] = 1. / nbmid;
    # weights[3] = 1. / nblong;
    for i in range(1,4):
        weights[i] = 1.
    print("work dataset of length %d/%d. short: %d, mid: %d, long: %d" % (len(brats_data),len(survival_data),nbshort,nbmid,nblong))

    return brats_data;

###
### ------------------------------------------------------------------------ ###
def clean_data(brats_data, max_l2c1error = 0.8, filter_GTR=True):

    # add vol(EN)/vol(TC)
    brats_data['vol(EN)/vol(TC)'] = brats_data['vol(EN)_a'].astype('float') / brats_data['vol(TC)_a'].astype('float');

    # 1. add rho-over-k
    brats_data['rho-inv'] = brats_data['rho-inv'].astype('float')
    brats_data['k-inv']   = brats_data['k-inv'].astype('float')
    dat_out  = brats_data.loc[brats_data['k-inv'] <= 0]
    brats_data  = brats_data.loc[brats_data['k-inv'] >  0]
    dat_out["filter-reason"] = "k zero"
    dat_filtered_out = dat_out;
    brats_data["rho-over-k"] = brats_data["rho-inv"]/brats_data["k-inv"]

    # 2. filter data with too large misfit
    brats_data['l2[Oc(1)-TC]'] = brats_data['l2[Oc(1)-TC]'].astype('float')
    dat_out = brats_data.loc[brats_data['l2[Oc(1)-TC]'] >= max_l2c1error]
    brats_data = brats_data.loc[brats_data['l2[Oc(1)-TC]'] <  max_l2c1error]
    dat_out["filter-reason"] = "l2(Oc1-d)err > "+str(max_l2c1error)
    dat_filtered_out = dat_out;
    # dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)

    brats_survival = brats_data.copy();

    # 3. filter survival data
    brats_survival['age'] = brats_survival['age'].astype('float')
    dat_out = brats_survival.loc[brats_survival['age'] <= 0]
    brats_survival = brats_survival.loc[brats_survival['age'] >  0]
    dat_out["filter-reason"] = "no survival data"
    # dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)

    brats_clustering = brats_survival.copy();

    # 4. filter GTR resection status
    if filter_GTR:
        dat_out = brats_survival.loc[brats_survival['resection_status'] != 'GTR']
        brats_survival = brats_survival.loc[brats_survival['resection_status'] ==  'GTR']
        dat_out["filter-reason"] = "no GTR"
        dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)
    else:
        brats_survival['resection_status[0]'] = brats_survival['resection_status'].apply(lambda x: np.array([0]) if pd.isna(x) else np.array([0]) if x == 'STR' else np.array([1]) );
        brats_survival['resection_status[1]'] = brats_survival['resection_status'].apply(lambda x: np.array([0]) if pd.isna(x) else np.array([1]) if x == 'STR' else np.array([0]) );


    print("\n\n### BraTS simulation data [cleaned] ### ")
    print(tabulate(brats_survival[["BID", "survival(days)", "age", "resection_status"]], headers='keys', tablefmt='psql'))
    print("\n\n### BraTS simulation data [discarded] ### ")
    print(tabulate(dat_filtered_out[["BID", "filter-reason"]], headers='keys', tablefmt='psql'))
    print()
    print("remaining data set for clustering consists of {} patients".format(len(brats_clustering)))
    print("remaining data set for survival prediction consists of {} patients".format(len(brats_survival)))
    return brats_clustering, brats_survival;



###
### ------------------------------------------------------------------------ ###
def plot_feature_subset(brats_data, cols):
    X = brats_data[cols].values.astype('float');
    Y = np.ravel(brats_data[['survival(days)']].values).astype('float');

    brats_data = brats_data.sort_values(by=['survival(days)']);
    # figure
    mpl.style.use('seaborn')
    sns.set(style="ticks")
    # f = plt.figure(figsize=(8,9))
    # grid1 = plt.GridSpec(4, 1, wspace=0.2, hspace=0.4, height_ratios=[4,4,1,1])

    # brats_data.plot.kde(y=cols, x='survival(days)', subplots=True);

    cols.append('survival_class')
    g = sns.PairGrid(brats_data[cols], hue='survival_class')
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.add_legend();

    # import SeabornFig2Grid as sfg
    # import matplotlib.gridspec as gridspec
    # fig = plt.figure(figsize=(20,20));
    # gs = gridspec.GridSpec(4, 4);
    # i = 0;
    # j = 0;
    # k = 0;
    # for c in cols:
    #     g = sns.jointplot(brats_data['survival(days)'], brats_data[c], kind="hex", color="#4CB391");
    #     mg0 = sfg.SeabornFig2Grid(g, fig, gs[i,j])
    #     k += 1;
    #     i  = i  + 1 if k % 4 == 0 and k > 0 else i
    #     j  = j  + 1 if k % 4 != 0 else 0
    # # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.4);
    # # fig.tight_layout();
    # gs.tight_layout(fig)
    # # fig.savefig(os.path.join('feature_plots', "feature_hexbinplot.pdf"), format='pdf', dpi=1200);



    plt.show();





###
### ------------------------------------------------------------------------ ###
def get_feature_subset(brats_data, type, purpose, estimator='classifier'):

    cols = []
    cols_p = []
    # features used for brain clustering
    if 'image_based' in type and purpose == 'clustering':
        cols.append('vol(TC)_r');                  # ib.01 TC rel. volume
        # cols.append('area(TC)_r');               # ib.02 TC rel. surface (rel. to sphere with volume of TC)
        cols.append('vol(ED)_r');                  # ib.03 ED rel. volume
        # cols.append('area(ED)_r');               # ib.04 ED rel. surface (rel. to sphere with volume of ED)
        cols.append('vol(NEC)_r');                 # ib.05 NE rel. volume
        # cols.append('area(NEC)_r');              # ib.06 NE rel. surface (rel. to sphere with volume of NE)
        cols.append('age');                        # ib.07 patient age
        cols.append('survival(days)');
        cols.append('n_comps');                    # number of components with rel. mass larger 1E-3
    # image based features used for survival prediciton
    if  'image_based' in type and purpose == 'prediction':
        cols.append('vol(TC)_r');                   # ib.01 TC rel. volume
        # cols.append('area(TC)_r');                  # ib.02 TC rel. surface (rel. to sphere with volume of TC)
        cols.append('vol(ED)_r');                   # ib.03 ED rel. volume
        # cols.append('area(ED)_r');                  # ib.04 ED rel. surface (rel. to sphere with volume of ED)
        cols.append('vol(NEC)_r');                  # ib.05 NE rel. volume
        # cols.append('area(NEC)_r');                 # ib.06 NE rel. surface (rel. to sphere with volume of NE)
        cols.append('age');                         # ib.07 patient age
        # cols.append('int{c(0)}/int{c(1)}');
        cols.append('vol(EN)/vol(TC)');
        # cols.append('resection_status[0]');
        # cols.append('resection_status[1]');
        # cols.append('resection_status');          # ib.08 resection status TODO
        # cols_p.append('cm(NEC|_#c) (#c=0,apsace)')  # ib.09 center of mass of NE of largest TC component, in a-space
        # cols.append('vol(TC|_#c)_r(#c=0)')          # ib.10 vol(TC) in comp #0 rel. to toatal vol(TC)
        # cols.append('vol(TC|_#c)_r(#c=1)')          # ib.10 vol(TC) in comp #1 rel. to toatal vol(TC)
        # cols.append('vol(TC|_#c)_r(#c=2)')          # ib.10 vol(TC) in comp #2 rel. to toatal vol(TC)
        # cols.append('vol(NEC|_#c)_r(#c=0)')         # ib.10 vol(NE) in comp #0 rel. to toatal vol(NE)
        # cols.append('vol(NEC|_#c)_r(#c=1)')         # ib.10 vol(NE) in comp #1 rel. to toatal vol(NE)
        # cols.append('vol(NEC|_#c)_r(#c=2)')         # ib.10 vol(NE) in comp #2 rel. to toatal vol(NE)
        cols.append('n_comps');                     # number of components with rel. mass larger 1E-3
    # physics based features used for survival prediciton
    if  'physics_based' in type and purpose == 'prediction':
        # cols.append('l2[c(0)|_#c]_r(#c=0)');        # ph.01 l2norm of c(0) in comp #0 rel. to total l2norm of c(0)
        # cols.append('l2[c(0)|_#c]_r(#c=1)');        # ph.01 l2norm of c(0) in comp #1 rel. to total l2norm of c(0)
        # cols.append('l2[c(0)|_#c]_r(#c=2)');        # ph.01 l2norm of c(0) in comp #2 rel. to total l2norm of c(0)
        cols_p.append('cm(c(0)|_#c) (#c=0,aspace)') # ph.02 center of mass of c(0) in comp #0 (in a-space)
        # cols_p.append('cm(c(0)|_#c) (#c=1,aspace)') # ph.02 center of mass of c(0) in comp #1 (in a-space)
        # cols_p.append('cm(c(0)|_#c) (#c=2,aspace)') # ph.02 center of mass of c(0) in comp #2 (in a-space)
        # add distance c(0) to VE as feature
        # cols.append('rho-inv');                     # ph.03 inversion variables prolifaration, migration
        cols.append('k-inv');                       # ph.03 inversion variables prolifaration, migration
        cols.append('rho-over-k');                  # ph.03 inversion variables prolifaration, migration
        # cols.append('l2[c(1)|_TC-TC,scaled]_r')     # ph.04 rescaled misfit, i.e., recon. 'Dice' of TC
        pass;

    # brats_data['vol(TC|_#c)_r(#c=0)'] = brats_data['vol(TC|_#c)_r(#c=0)'].apply(lambda x: -1 if pd.isna(x) else x);
    # brats_data['vol(TC|_#c)_r(#c=1)'] = brats_data['vol(TC|_#c)_r(#c=1)'].apply(lambda x: -1 if pd.isna(x) else x);
    # brats_data['vol(TC|_#c)_r(#c=2)'] = brats_data['vol(TC|_#c)_r(#c=2)'].apply(lambda x: -1 if pd.isna(x) else x);
    # brats_data['vol(NEC|_#c)_r(#c=0)'] = brats_data['vol(NEC|_#c)_r(#c=0)'].apply(lambda x: -1 if pd.isna(x) else x);
    # brats_data['vol(NEC|_#c)_r(#c=1)'] = brats_data['vol(NEC|_#c)_r(#c=1)'].apply(lambda x: -1 if pd.isna(x) else x);
    # brats_data['vol(NEC|_#c)_r(#c=2)'] = brats_data['vol(NEC|_#c)_r(#c=2)'].apply(lambda x: -1 if pd.isna(x) else x);

    # process columns that cannot be interprete directly
    if len(cols_p) > 0:
        for col in cols_p:
            print("preprocessing column", col)
            brats_data[col+'[0]'] = brats_data[col].apply(lambda x: -1 if pd.isna(x) else float(x.split('(')[-1].split(',')[0]));
            brats_data[col+'[1]'] = brats_data[col].apply(lambda x: -1 if pd.isna(x) else float(x.split(',')[1]));
            brats_data[col+'[2]'] = brats_data[col].apply(lambda x: -1 if pd.isna(x) else float(x.split(',')[2].split(')')[0]));
            cols.extend([col+'[0]',col+'[1]',col+'[2]'])

    if  'physics_based' in type and purpose == 'prediction':
        brats_data['cm(c(0)|_#c) (#c=0,aspace)[0]'] = brats_data['cm(c(0)|_#c) (#c=0,aspace)[0]'].apply(lambda x: -1 if pd.isna(x) else x);
        brats_data['cm(c(0)|_#c) (#c=0,aspace)[1]'] = brats_data['cm(c(0)|_#c) (#c=0,aspace)[1]'].apply(lambda x: -1 if pd.isna(x) else x);
        brats_data['cm(c(0)|_#c) (#c=0,aspace)[2]'] = brats_data['cm(c(0)|_#c) (#c=0,aspace)[2]'].apply(lambda x: -1 if pd.isna(x) else x);
        # brats_data['cm(c(0)|_#c) (#c=1,aspace)[0]'] = brats_data['cm(c(0)|_#c) (#c=1,aspace)[0]'].apply(lambda x: -1 if pd.isna(x) else x);
        # brats_data['cm(c(0)|_#c) (#c=1,aspace)[1]'] = brats_data['cm(c(0)|_#c) (#c=1,aspace)[1]'].apply(lambda x: -1 if pd.isna(x) else x);
        # brats_data['cm(c(0)|_#c) (#c=1,aspace)[2]'] = brats_data['cm(c(0)|_#c) (#c=1,aspace)[2]'].apply(lambda x: -1 if pd.isna(x) else x);
        # brats_data['cm(c(0)|_#c) (#c=2,aspace)[0]'] = brats_data['cm(c(0)|_#c) (#c=2,aspace)[0]'].apply(lambda x: -1 if pd.isna(x) else x);
        # brats_data['cm(c(0)|_#c) (#c=2,aspace)[1]'] = brats_data['cm(c(0)|_#c) (#c=2,aspace)[1]'].apply(lambda x: -1 if pd.isna(x) else x);
        # brats_data['cm(c(0)|_#c) (#c=2,aspace)[2]'] = brats_data['cm(c(0)|_#c) (#c=2,aspace)[2]'].apply(lambda x: -1 if pd.isna(x) else x);
        pass;

    print()
    print("active features:\n", cols)

    brats_data['is_na'] = brats_data[cols].isnull().apply(lambda x: any(x), axis=1)
    # brats_data = brats_data.loc[brats_data['is_na']]
    # print(tabulate(brats_data[cols], headers='keys', tablefmt='psql'))
    brats_data          = brats_data.loc[brats_data['is_na'] == False]
    dat_out             = brats_data.loc[brats_data['is_na'] == True]
    dat_out["filter-reason"] = "nan values"
    if len(dat_out) >=1:
        print("\n\n### BraTS simulation data [discarded] ### ")
        print(tabulate(dat_out[["BID", "filter-reason"]], headers='keys', tablefmt='psql'))


    # pair plot
    # v_cols = cols.copy();
    # v_cols.append('survival_class');
    # v_data = brats_data[v_cols];
    # sns.set(style="ticks", color_codes=True);
    # sns.pairplot(data=v_data,
    #     hue = 'survival_class', diag_kind = 'bar',
    #     palette=sns.xkcd_palette(['dark blue', 'dark green', 'gold', 'orange']),
    #     plot_kws=dict(alpha = 0.7),
    #     diag_kws=dict(shade=True))
    # plt.show()

    X = brats_data[cols].values.astype('float');
    Y = np.ravel(brats_data[['survival_class']].values).astype('int');
    if estimator == 'regressor':
        Y = np.ravel(brats_data[['survival(days)']].values).astype('float');
    return X, Y, cols;

def feature_selection(clf, feature_list):
    pass;

###
### ------------------------------------------------------------------------ ###
def preprocess_features(X_train, X_test=None, normalize=True, reduce_dims=None):
    # scaler     = preprocessing.StandardScaler()
    scaler     = preprocessing.MinMaxScaler()
    # scaler     = preprocessing.RobustScaler()
    normalizer = preprocessing.Normalizer()

    # Normalization
    if normalize:
        X_train = scaler.fit_transform(X_train)
        # normalizer.fit(X_train)
        # X_train = normalizer.transform(X_train)
        pass
        if X_test is not None:
            X_test = scaler.transform(X_test)
            # X_test = normalizer.transform(X_test)
            pass

    X_train_pca = X_train.copy();
    X_test_pca = X_test.copy() if X_test is not None else None;
    pca = None;
    # PCA dimensionality reduction
    if reduce_dims is not None:
        pca = PCA(n_components=reduce_dims, svd_solver='randomized', whiten=True).fit(X_train)
        # projecting data
        X_train_pca = pca.transform(X_train_pca)
        if X_test is not None:
            X_test_pca = pca.transform(X_test_pca)
        print()
        print("dim(X_train) before PCA: {}, and after PCA: {} ".format(X_train.shape, X_train_pca.shape))
    return X_train_pca, X_test_pca, pca


###
### ------------------------------------------------------------------------ ###
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=300)))
    return estimator;


###
### ------------------------------------------------------------------------ ###
def model_selection_RFC(X_train, y_train, type='random_search'):
    # randomized search
    if type == 'random_search':
        from sklearn.model_selection import RandomizedSearchCV
        # number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # method of selecting samples for training each tree
        bootstrap = [True, False]
        # create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        pprint(random_grid)

        # random forrest
        rf = RandomForestRegressor();
        # rf = RandomForestClassifier();
        # random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # fit the random search model
        rf_random.fit(X_train, y_train)

        print("best parameters:")
        pprint(rf_random.best_params_);
        return rf_random.best_params_, rf_random.best_estimator_;

    # grid-search
    elif type == 'grid_search':
        from sklearn.model_selection import GridSearchCV
        # create the parameter grid based on the results of random search
        # params from random grid search:
        # {'bootstrap': False,
        #  'max_depth': 30,
        #  'max_features': 'auto',
        #  'min_samples_leaf': 4,
        #  'min_samples_split': 2,
        #  'n_estimators': 2000}
        param_grid = {
            'bootstrap': [False],
            'max_depth': [20, 30, 40, 50, 100],
            'max_features': [2, 3, 'auto'],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'n_estimators': [1000, 1125, 1250, 1500 ]
        }
        # create a based model
        # rf = RandomForestRegressor();
        rf = RandomForestClassifier();
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
        grid_search.fit(X_train, y_train)

        return grid_search.best_params_, grid_search.best_estimator_;


###
### ------------------------------------------------------------------------ ###
def evaluate(clf, X_test, y_test, feature_list, estimator='classifier'):
    # print list of feature importance
    importances = list(clf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    [print('feature: {:30} importance: {}'.format(*pair)) for pair in feature_importances]
    # plot feature importance
    x_values = list(range(len(importances)))
    fig  = plt.figure();
    ax = fig.add_subplot(1,2,1);
    ax.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title('Feature Importances');
    # plot accumulative importance
    sorted_importances = [importance[1] for importance in feature_importances]
    sorted_features = [importance[0] for importance in feature_importances]
    cumulative_importances = np.cumsum(sorted_importances)
    ax = fig.add_subplot(1,2,2);
    ax.plot(x_values, cumulative_importances, 'g-')
    ax.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
    plt.xticks(x_values, sorted_features, rotation = 'vertical')
    plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');
    plt.tight_layout();
    # plt.show();

    print();
    print("Detailed classification report:");
    print();
    y_true, y_pred = y_test, clf.predict(X_test);

    if estimator == 'regressor':
        y_true_class = [getSurvivalClass(x) for x in y_true];
        y_pred_class = [getSurvivalClass(x) for x in y_pred];
        print(classification_report(y_pred_class, y_true_class));
    else:
        print(classification_report(y_true, y_pred));
    print()
    print("y_true: ", y_true)
    print("y_pred: ", y_pred)
    if estimator == 'regressor':
        print()
        print("y_true[class]: ", y_true_class)
        print("y_pred[class]: ", y_pred_class)

    print("score:", clf.score(X_test, y_test))
    print("mean squared error (MSE): %.2f" % sklearn.metrics.mean_squared_error(y_true, y_pred))
    print('variance score: %.2f' % sklearn.metrics.r2_score(y_true, y_pred))
    if estimator == 'classifier':
        print('accuracy score: %.2f' % metrics.accuracy_score(y_true, y_pred));
        # print('f1 score: %.2f' % metrics.f1_score(y_true, y_pred));
    if estimator == 'regressor':
        print('accuracy score[class]: %.2f' % metrics.accuracy_score(y_true_class, y_pred_class));
        print("confusion matrix:\n", sklearn.metrics.confusion_matrix(y_true_class, y_pred_class))
        # print('f1 score: %.2f' % metrics.f1_score(y_true_class, y_pred_class));
    else:
        print("confusion matrix:\n", sklearn.metrics.confusion_matrix(y_true, y_pred))



###
### ------------------------------------------------------------------------ ###
def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='merror', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='merror')

    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    # plt.show()

###
### ------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------ ###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    # parse arguments
    basedir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',    '--dir',   type = str, help = 'path to the results folder');
    parser.add_argument ('-l',    '--level', type = int, help = 'path to the results folder');
    parser.add_argument ('-file', '--f',     type = str, help = 'path to the csv brats file');
    args = parser.parse_args();

    brats_data = read_data(args.dir, args.level, args.f);
    brats_clustering, brats_survival = clean_data(brats_data, filter_GTR=False);

    CLUSTER_BRAINS = False;
    PREDICT_SURVIVAL = True;

    ### ------------------------------------------------------------------ ###
    # a) cluster brains
    if CLUSTER_BRAINS:
        X_ib, Y_ib, cols = get_feature_subset(brats_clustering, type=["image_based"], purpose='clustering', estimator='classifier');
        X_ib_n , d, d    = preprocess_features(X_ib, normalize=True);

        # k-means clustering
        COLORS = {}
        X_ib_cluster = X_ib_n.copy();
        range_n_clusters = [2, 3, 4, 10] # last entry not used, dummy
        for j, n_clusters in zip(range(len(range_n_clusters)), range_n_clusters):
            COLORS[j] = {}
            if j < len(range_n_clusters)-1:
                X_ib_pca_reduced, dummy, pca = preprocess_features(X_ib_cluster, normalize=False, reduce_dims=None);
                print("n_clusters: {}".format(n_clusters))
                print('init\t\ttime\tinertia\tsilhouette')
                kmeans = bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=100), name="k-means++ (d)", data=X_ib_pca_reduced)
                # kmeans = bench_k_means(KMeans(init='k-means++',     n_clusters=n_clusters, n_init=100), name="k-means++ (d)", data=X_ib)
                # kmeans = bench_k_means(KMeans(init='random',        n_clusters=n_clusters, n_init=100), name="random (d)",    data=X_ib)
                # kmeans = bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),  name="PCA-based (d)", data=X_ib)
                # kmeans = bench_k_means(KMeans(init='k-means++',     n_clusters=n_clusters, n_init=100), name="k-means++ (rd)", data=X_ib_pca_reduced)
                # kmeans = bench_k_means(KMeans(init='random',        n_clusters=n_clusters, n_init=100), name="random (rd)",    data=X_ib_pca_reduced)
                print("normalized mutual information (survival class): {}".format(sklearn.metrics.cluster.normalized_mutual_info_score(kmeans.labels_, Y_ib)));
                print("confusion matrix:\n", sklearn.metrics.confusion_matrix(Y_ib, kmeans.labels_))
                labels = kmeans.labels_;
            else:
                labels = Y_ib;
            for c in range(np.max(labels)+1):
                COLORS[j][c] = labels==c;

        # T-SNE visualization
        Y = {}
        k = 0
        X_ib_n , d, d    = preprocess_features(X_ib, normalize=False);
        for reduced_dims in [None, 4, 2]:
            X_ib_vis = X_ib_n.copy();
            X_ib_vis, dummy, pca = preprocess_features(X_ib_vis, normalize=False, reduce_dims=reduced_dims);
            fig, axx = plt.subplots(4, 6, figsize=(15, 10), squeeze=False);
            for i, perplexity in enumerate([5,10,30,40,50,100]):
                t0 = time()
                tsne = manifold.TSNE(n_components=2, init='random',
                                     random_state=0,
                                     early_exaggeration=12, n_iter_without_progress=1000,
                                     n_iter=10000, learning_rate=100,
                                     perplexity=perplexity)
                Y[k] = tsne.fit_transform(X_ib_vis)
                t1 = time()
                print("perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

                # coloring from k-means
                for j in range(len(COLORS)):
                    ax = axx[j][i]
                    pal = sns.color_palette("tab10", n_colors=6);
                    for c in range(len(COLORS[j])):
                        ax.scatter(Y[k][COLORS[j][c],0], Y[k][COLORS[j][c],1], color=pal[c]);
                    ax.set_title("Perplexity=%d" % perplexity)
                    ax.xaxis.set_major_formatter(NullFormatter())
                    ax.yaxis.set_major_formatter(NullFormatter())
                    ax.axis('tight')
            k += 1;
        plt.show()


    ### ------------------------------------------------------------------------ ###
    # b) survival (image based)
    if PREDICT_SURVIVAL:
        # X_ib, Y_ib, cols = get_feature_subset(brats_survival, type=["image_based"], purpose='prediction', estimator='classifier');
        X_ib, Y_ib, cols = get_feature_subset(brats_survival, type=["image_based", "physics_based"], purpose='prediction', estimator='classifier');


        from sklearn.model_selection import cross_val_score

        print()
        print("basic data statistics:\n")
        brats_survival[cols].describe()
        print()
        print(bcolors.OKBLUE, "predicting survival using {} samples and {} features".format(*X_ib.shape),bcolors.ENDC)
        print(len(brats_survival), X_ib.shape)

        X_train, X_test, y_train, y_test = train_test_split(X_ib, Y_ib, random_state = 0, test_size=0.25)
        X_ib_n , X_ib_n_test, d = preprocess_features(X_train, X_test, normalize=False);

        # randomized grid search with cross-validation for hyper parameter tuning
        # hyper_dict, rf_best_rs = model_selection_RFC(X_ib_n, y_train, type='random_search');
        # rf_best_rs.fit(X_ib_n, y_train);
        #
        # # grid search in narrowed down param space with cross-validation for hyper parameter tuning
        # hyper_dict, rf_best_gs = model_selection_RFC(X_ib_n, y_train, type='grid_search');
        # rf_best_gs.fit(X_ib_n, y_train);

        """
        # base model (random forrest classifier)
        rf_base = RandomForestClassifier(n_estimators=1000, random_state=42);
        rf_base.fit(X_ib_n, y_train);

        print("cross validation score:\n",sklearn.model_selection.cross_val_score(rf_base, X_ib, Y_ib, cv=10).mean());


        print()
        print(" ################################################################################# ")
        print("evaluate base model (random forrest classifier):")
        pprint(rf_base.get_params())
        evaluate(rf_base, X_ib_n_test, y_test, cols, estimator='classifier');

        scores = cross_val_score(rf_base, X_ib, Y_ib, cv=10)
        print(" ############## ")
        print("CV Scores: ", scores)
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print(" ############## ")
        """

        # --------------------

        rf_xgboost = XGBClassifier(silent = 0, learning_rate = 0.01, n_estimators = 1000, max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, objective= 'multi:softmax', nthread = 4, scale_pos_weight = 1, seed = 27, num_class=3, verbosity=1)
        modelfit(rf_xgboost, X_ib_n, y_train, useTrainCV=True)

        opt_nb_rounds = rf_xgboost.get_params()['n_estimators']
        print("found optimal number of rouns:", opt_nb_rounds)
        rf_xgboost = XGBClassifier(eta=0.01, objective="multi:softprob", random_state=42, verbosity=1, num_class=3);
        rf_xgboost.fit(X_ib_n, y_train)#, early_stopping_rounds=1000, eval_set=[(X_ib_n_test, y_test)]);

        param_test1 = {
            'max_depth': range(2,10,1),
            'min_child_weight': range(2,10,1)
        }
        # param_test1 = {
        #     'max_depth': [4,5,6],
        #     'min_child_weight': [4,5,6]
        # }
        # gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.01, n_estimators=opt_nb_rounds, max_depth=5,
        #                         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
        #                         objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, num_class=3),
        #                         param_grid = param_test1, n_jobs=4,iid=False, cv=5)
        # gsearch1.fit(X_ib_n,y_train)
        # print("grid scores:", gsearch1.cv_results_)
        # print("best params:", gsearch1.best_params_)
        # print("best score:", gsearch1.best_score_)
        #
        # param_test3 = {
        #     # 'gamma':[i/10.0 for i in range(0,5)]
        #     'reg_lambda' : [1,2,4,8,16,32,64,128,256,512,1024]
        # }
        # gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.01, n_estimators=opt_nb_rounds, max_depth=2,
        #                         min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
        #                         objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, num_class=3),
        #                         param_grid = param_test3, n_jobs=4,iid=False, cv=5)
        # gsearch3.fit(X_ib_n,y_train)
        # print("grid scores:", gsearch3.cv_results_)
        # print("best params:", gsearch3.best_params_)
        # print("best score:",  gsearch3.best_score_)


        print()
        print(" ################################################################################# ")
        print("evaluate XGboost model (random forrest classifier):")
        pprint(rf_xgboost.get_params())
        evaluate(rf_xgboost, X_ib_n_test, y_test, cols, estimator='classifier');
        scores = cross_val_score(rf_xgboost, X_ib, Y_ib, cv=5)
        print(" ############## ")
        print("CV Scores: ", scores)
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print(" ############## ")

        rf_xgboost = XGBClassifier( learning_rate=0.01, n_estimators=opt_nb_rounds, max_depth=5,
                                    min_child_weight=3, gamma=0, reg_lambda=2, subsample=0.8, colsample_bytree=0.8,
                                    objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, num_class=3)

        # # physics based CV accuracy: 0.5
        # rf_xgboost = XGBClassifier( learning_rate=0.01, n_estimators=40, max_depth=5,
        #                             min_child_weight=3, gamma=0, reg_lambda=2, subsample=0.8, colsample_bytree=0.8,
        #                             objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, num_class=3)
        rf_xgboost.fit(X_ib_n, y_train);

        print()
        print(" ################################################################################# ")
        print("evaluate XGboost model (random forrest classifier):")
        pprint(rf_xgboost.get_params())
        evaluate(rf_xgboost, X_ib_n_test, y_test, cols, estimator='classifier');
        scores = cross_val_score(rf_xgboost, X_ib, Y_ib, cv=5)
        print(" ############## ")
        print("CV Scores: ", scores)
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print(" ############## ")


        # rf_xgboost = XGBClassifier( learning_rate=0.01, n_estimators=opt_nb_rounds, max_depth=2,
        #                             min_child_weight=9, gamma=0.0, reg_lambda=1, subsample=0.8, colsample_bytree=0.8,
        #                             objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27, num_class=3)
        # rf_xgboost.fit(X_ib_n, y_train);


        # print("best score: {0}, best iteration: {1}, best ntree limit {2}".format(rf_xgboost.best_score, rf_xgboost.best_iteration, rf_xgboost.best_ntree_limit))

        # rf_xgboost = XGBClassifier(n_estimators=rf_xgboost.best_iteration, objective="multi:softprob", random_state=42, verbosity=1, num_class=3);
        # rf_xgboost.fit(X_ib_n, y_train);

        # xgb.plot_importance(rf_xgboost)
        # xgb.to_graphviz(rf_xgboost, num_trees=rf_xgboost.best_iteration)
        # plt.show()



        # print()
        # print(" ################################################################################# ")
        # print("evaluate XGboost model (random forrest classifier):")
        # pprint(rf_xgboost.get_params())
        # evaluate(rf_xgboost, X_ib_n_test, y_test, cols, estimator='classifier');

        scores = cross_val_score(rf_xgboost, X_ib, Y_ib, cv=5)
        print(" ############## ")
        print("CV Scores: ", scores)
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print(" ############## ")


        # --------------------

        """
        # X_ib, Y_ib, cols = get_feature_subset(brats_survival, type=["image_based"], purpose='prediction', estimator='regressor');
        X_ib, Y_ib, cols = get_feature_subset(brats_survival, type=["image_based", "physics_based"], purpose='prediction', estimator='regressor');

        # plot_feature_subset(brats_survival, cols);

        X_train, X_test, y_train, y_test = train_test_split(X_ib, Y_ib, random_state = 0, test_size=0.25)
        X_ib_n , X_ib_n_test, d = preprocess_features(X_train, X_test, normalize=False);

        # base model (random forrest regressor)
        from sklearn.ensemble import RandomForestRegressor
        rf_reg_base = RandomForestRegressor(n_estimators=1000, random_state=42);
        rf_reg_base.fit(X_ib_n, y_train);

        print()
        print(" ################################################################################# ")
        print("evaluate base model (random forrest regressor):")
        pprint(rf_reg_base.get_params())
        evaluate(rf_reg_base, X_ib_n_test, y_test, cols, estimator='regressor');
        print()
        """

        # print("tune hyperparams (best random search model):")
        # pprint(rf_best_rs.get_params())
        # evaluate(rf_best_rs, X_ib_n_test, y_test, cols);
        # print()
        # print("evaluate best grid search model:")
        # pprint(rf_best_gs.get_params())
        # evaluate(rf_best_gs, X_ib_n_test, y_test, cols);















# ### ======================================================================================+ ### #
# ### ======================================================================================+ ### #
# ### ======================================================================================+ ### #


# active features:



 #################################################################################
# evaluate XGboost model (random forrest classifier):
# {'base_score': 0.5,
#  'booster': 'gbtree',
#  'colsample_bylevel': 1,
#  'colsample_bynode': 1,
#  'colsample_bytree': 0.8,
#  'gamma': 0,
#  'learning_rate': 0.01,
#  'max_delta_step': 0,
#  'max_depth': 5,
#  'min_child_weight': 3,
#  'missing': None,
#  'n_estimators': 40,
#  'n_jobs': 1,
#  'nthread': 4,
#  'num_class': 3,
#  'objective': 'multi:softprob',
#  'random_state': 0,
#  'reg_alpha': 0,
#  'reg_lambda': 2,
#  'scale_pos_weight': 1,
#  'seed': 27,
#  'silent': None,
#  'subsample': 0.8,
#  'verbosity': 1}
# feature: rho-over-k                     importance: 0.14000000059604645
# feature: area(TC)_r                     importance: 0.10000000149011612
# feature: age                            importance: 0.10000000149011612
# feature: vol(TC)_r                      importance: 0.09000000357627869
# feature: area(ED)_r                     importance: 0.09000000357627869
# feature: cm(c(0)|_#c) (#c=0,aspace)[2]  importance: 0.09000000357627869
# feature: vol(ED)_r                      importance: 0.07000000029802322
# feature: vol(NEC)_r                     importance: 0.07000000029802322
# feature: cm(c(0)|_#c) (#c=0,aspace)[0]  importance: 0.05999999865889549
# feature: cm(c(0)|_#c) (#c=0,aspace)[1]  importance: 0.05999999865889549
# feature: area(NEC)_r                    importance: 0.05000000074505806
# feature: n_comps                        importance: 0.05000000074505806
# feature: vol(EN)/vol(TC)                importance: 0.03999999910593033
#
# Detailed classification report:
#
#               precision    recall  f1-score   support
#
#            0       0.57      0.86      0.69        14
#            1       0.33      0.08      0.13        12
#            2       0.47      0.53      0.50        15
#
#     accuracy                           0.51        41
#    macro avg       0.46      0.49      0.44        41
# weighted avg       0.46      0.51      0.46        41
#
#
# y_true:  [2 0 1 0 2 2 1 0 1 2 1 1 1 2 1 1 0 2 0 1 0 2 2 2 1 2 0 2 2 2 0 0 0 0 1 0 0
#  0 1 2 2]
# y_pred:  [2 0 0 0 0 0 2 0 2 1 2 1 2 1 2 0 2 2 0 2 0 2 2 0 0 2 0 2 0 2 0 0 0 0 2 2 0
#  0 0 2 0]
# score: 0.5121951219512195
# mean squared error (MSE): 1.00
# variance score: -0.41
# accuracy score: 0.51
# confusion matrix:
#  [[12  0  2]
#  [ 4  1  7]
#  [ 5  2  8]]
#  ##############
# CV Scores:  [0.35294118 0.53125    0.53125    0.5625     0.51612903]
# CV Accuracy: 0.50 (+/- 0.15)
#  ##############
