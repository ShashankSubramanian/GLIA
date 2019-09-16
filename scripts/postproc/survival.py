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
import scipy
from os import listdir
import numpy as np
import os
from tabulate import tabulate

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
        return 1;# np.array([1,0,0])
    elif m < 15:
        return 2;# np.array([0,1,0])
    else:
        return 3;# np.array([0,0,1])


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
            patient_data = pd.read_csv(os.path.join(patient_path,"features_"+str(level)+".csv"), index_col=0, header = None, error_bad_lines=True, sep=';', skipinitialspace=True).transpose();
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

    # read brats data
    if dir is None and file is not None:
        brats_data = pd.read_csv(os.path.join(basedir,file), header = 0, error_bad_lines=True, skipinitialspace=True)
        print("read brats simulation data of length %d" % len(brats_data))

    nbshort = len(brats_data.loc[brats_data['survival_class'] ==  1])
    nbmid   = len(brats_data.loc[brats_data['survival_class'] ==  2])
    nblong  = len(brats_data.loc[brats_data['survival_class'] ==  3])
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
def clean_data(brats_data, max_l2c1error = 0.8):
    # 1. add rho-over-k
    # brats_data['rho-inv'] = brats_data['rho-inv'].astype('float')
    # brats_data['k-inv']   = brats_data['k-inv'].astype('float')
    # dat_out  = brats_data.loc[brats_data['k-inv'] <= 0]
    # brats_data  = brats_data.loc[brats_data['k-inv'] >  0]
    # dat_out["filter-reason"] = "k zero"
    # dat_filtered_out = dat_out;
    # brats_data["rho-over-k"] = brats_data["rho-inv"]/brats_data["k-inv"]

    # 2. filter data with too large misfit
    brats_data['l2[Oc(1)-TC]'] = brats_data['l2[Oc(1)-TC]'].astype('float')
    dat_out = brats_data.loc[brats_data['l2[Oc(1)-TC]'] >= max_l2c1error]
    brats_data = brats_data.loc[brats_data['l2[Oc(1)-TC]'] <  max_l2c1error]
    dat_out["filter-reason"] = "l2(Oc1-d)err > "+str(max_l2c1error)
    dat_filtered_out = dat_out;
    # dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)

    brats_clustering = brats_data.copy();
    brats_survival = brats_data.copy();

    # 3. filter survival data
    brats_survival['age'] = brats_survival['age'].astype('float')
    dat_out = brats_survival.loc[brats_survival['age'] <= 0]
    brats_survival = brats_survival.loc[brats_survival['age'] >  0]
    dat_out["filter-reason"] = "no survival data"
    # dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)

    # 4. filter GTR resection status
    dat_out = brats_survival.loc[brats_survival['resection_status'] != 'GTR']
    brats_survival = brats_survival.loc[brats_survival['resection_status'] ==  'GTR']
    dat_out["filter-reason"] = "no GTR"
    dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)



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
def get_feature_subset(brats_data, type, purpose):

    cols = []
    # features used for brain clustering
    if purpose == 'clustering':
        cols.append('vol(TC)_r');                # ib.01 TC rel. volume
        cols.append('area(TC)_r');               # ib.02 TC rel. surface (rel. to sphere with volume of TC)
        cols.append('vol(ED)_r');                # ib.03 ED rel. volume
        cols.append('area(ED)_r');               # ib.04 ED rel. surface (rel. to sphere with volume of ED)
        cols.append('vol(NE)_r');                # ib.05 NE rel. volume
        cols.append('area(NE)_r');               # ib.06 NE rel. surface (rel. to sphere with volume of NE)
        cols.append('age');                      # ib.07 patient age
        cols.append('survival(days)');
        cols.append('n_comps');                  # number of components with rel. mass larger 1E-3
    # image based features used for survival prediciton
    if type == 'image_based' and purpose == 'prediction':
        cols.append('vol(TC)_r');                # ib.01 TC rel. volume
        cols.append('area(TC)_r');               # ib.02 TC rel. surface (rel. to sphere with volume of TC)
        cols.append('vol(ED)_r');                # ib.03 ED rel. volume
        cols.append('area(ED)_r');               # ib.04 ED rel. surface (rel. to sphere with volume of ED)
        cols.append('vol(NE)_r');                # ib.05 NE rel. volume
        cols.append('area(NE)_r');               # ib.06 NE rel. surface (rel. to sphere with volume of NE)
        cols.append('age');                      # ib.07 patient age
        # cols.append('resection_status');         # ib.08 resection status TODO
        cols.append('cm(NEC|_#c) (#c=0,aspace)') # ib.09 center of mass of NE of largest TC component, in a-space
        cols.append('vol(TC|_#c)_r(#c=0)')       # ib.10 vol(TC) in comp #0 rel. to toatal vol(TC)
        cols.append('vol(TC|_#c)_r(#c=1)')       # ib.10 vol(TC) in comp #1 rel. to toatal vol(TC)
        cols.append('vol(TC|_#c)_r(#c=2)')       # ib.10 vol(TC) in comp #2 rel. to toatal vol(TC)
        cols.append('vol(NE|_#c)_r(#c=0)')       # ib.10 vol(NE) in comp #0 rel. to toatal vol(NE)
        cols.append('vol(NE|_#c)_r(#c=1)')       # ib.10 vol(NE) in comp #1 rel. to toatal vol(NE)
        cols.append('vol(NE|_#c)_r(#c=2)')       # ib.10 vol(NE) in comp #2 rel. to toatal vol(NE)
        cols.append('n_comps');                  # number of components with rel. mass larger 1E-3
    # physics based features used for survival prediciton
    if type == 'physics_based' and purpose == 'prediction':
        cols.append('l2[c(0)|_#c]_r(#c=0)');     # ph.01 l2norm of c(0) in comp #0 rel. to total l2norm of c(0)
        cols.append('l2[c(0)|_#c]_r(#c=1)');     # ph.01 l2norm of c(0) in comp #1 rel. to total l2norm of c(0)
        cols.append('l2[c(0)|_#c]_r(#c=2)');     # ph.01 l2norm of c(0) in comp #2 rel. to total l2norm of c(0)
        cols.append('cm(c(0)_#c)_(#c=0,aspace)') # ph.02 center of mass of c(0) in comp #0 (in a-space)
        cols.append('cm(c(0)_#c)_(#c=1,aspace)') # ph.02 center of mass of c(0) in comp #1 (in a-space)
        cols.append('cm(c(0)_#c)_(#c=2,aspace)') # ph.02 center of mass of c(0) in comp #2 (in a-space)
        cols.append('rho-inv');                  # ph.03 inversion variables prolifaration, migration
        cols.append('k-inv');                    # ph.03 inversion variables prolifaration, migration
        cols.append('rho-over-k');               # ph.03 inversion variables prolifaration, migration
        cols.append('l2[c(1)|_TC-TC,scaled]_r')  # ph.04 rescaled misfit, i.e., recon. 'Dice' of TC

    brats_data['is_na'] = brats_data[cols_bio].isnull().apply(lambda x: any(x), axis=1)
    brats_data          = brats_data.loc[brats_data['is_na'] == False]
    dat_out             = brats_data.loc[brats_data['is_na'] == True]
    dat_out["filter-reason"] = "nan values"
    if len(dat_out) >=1:
        print("\n\n### BraTS simulation data [discarded] ### ")
        print(tabulate(dat_out[["BID", "filter-reason"]], headers='keys', tablefmt='psql'))

    X = brats_data[cols].values
    Y = np.ravel(brats_data[['survival_class']].values).astype('int')
    return X, Y


###
### ------------------------------------------------------------------------ ###
def preprocess_features(X_train, X_test):
    # === normalization/scaling ===
    # scaler    = preprocessing.StandardScaler().fit(X)
    scaler     = preprocessing.MinMaxScaler()
    normalizer = preprocessing.Normalizer()
    scaler.fit_transform(X_train)
    normalizer.fit(X_train)

    scaler.transform(X_test)
    normalizer.transform(X_train)
    normalizer.transform(X_test)

    # PCA dimensionality reduction
    pca = PCA(n_components=3, svd_solver='randomized', whiten=True).fit(X_train)
    # projecting data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # X_train_pca = X_train
    # X_test_pca  = X_test

    print("dim(X_train) before PCA: {} and after {} ".format(X_train.shape, X_train_pca.shape))
    return X_train_pca, X_test_pca


###
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
    brats_clustering, brats_survival = clean_data(brats_data);

    # a) cluster brains
    X_ib, Y_ib = get_feature_subset(brats_clustering, type="image_based");


    # X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)
    # preprocess_features(features);



    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_test, clf.predict(X_test_pca)
    # # print(classification_report(y_true, y_pred))
    # print()
    # print("y_true: ", y_true)
    # print("y_pred: ", y_pred)
    #
    # print(classification_report(y_test, y_pred))
    # accuracy = clf.score(X_test_pca, y_test)
