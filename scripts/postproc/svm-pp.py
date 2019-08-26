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
        return 1
    elif m < 15:
        return 2
    else:
        return 3

###
### ------------------------------------------------------------------------ ###
def svc_param_selection(X, y, nfolds, method, w):
    # Cs = [1E-5,1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2, 1E3, 1E4, 1E5]
    # gammas = [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2]
    # param_grid = {'C': Cs, 'gamma' : gammas} if kernel == 'rbf' or kernel == 'poly' else  {'C': Cs}
    # grid_search = GridSearchCV(svm.SVC(kernel=kernel, class_weight=w), param_grid, cv=nfolds)
    param_grid = [
    {'C': [1E-5,1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2, 1E3, 1E4, 1E5], 'kernel': ['linear']},
    {'C': [1E-5,1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2, 1E3, 1E4, 1E5], 'gamma': [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2], 'kernel': ['rbf']},
    ]
    if method=='SVM':
        clf = GridSearchCV(SVC(), param_grid, cv=nfolds)
    elif method=='SVR':
        clf = GridSearchCV(SVR(epsilon=0.2), param_grid, cv=nfolds)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    return clf, clf.best_params_

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


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    # parse arguments
    basedir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',  '--dir',  type = str, help = 'path to the results folder');
    parser.add_argument ('-file',  '--f', type = str, help = 'path to the csv brats file');
    args = parser.parse_args();

    max_l2c1error=0.8

    # read survival data
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True)
    # read brats data
    brats_data    = pd.read_csv(os.path.join(basedir,args.f), header = 0, error_bad_lines=True, skipinitialspace=True)
    print("read brats simulation data of length %d" % len(brats_data))

    # 1. convert categorial data
    brats_data['srvl[]'] = brats_data['srvl[]'].astype('category')
    brats_data['srvl[]'].cat.categories = [3,2,1]   # alpha numeric sorting: [NA, long, mid, short] -> [-1, 2,1,0]
    brats_data['srvl[]'] = brats_data['srvl[]'].astype('float')
    brats_data['srgy'] = brats_data['srgy'].astype('category')
    brats_data['srgy'].cat.categories = [3,2,1] # [tuple([1,0,0]),tuple([0,1,0]), tuple([0,0,1])]
    # 2. filter survival data
    brats_data['age'] = brats_data['age'].astype('float')
    dat_out = brats_data.loc[brats_data['age'] <= 0]
    brats_data = brats_data.loc[brats_data['age'] >  0]
    dat_out["filter-reason"] = "no survival data"
    dat_filtered_out = dat_out;
    # 3. filter data with too large misfit
    brats_data['l2c1'] = brats_data['l2Oc1'].astype('float')
    dat_out = brats_data.loc[brats_data['l2Oc1'] >= max_l2c1error]
    brats_data = brats_data.loc[brats_data['l2Oc1'] <  max_l2c1error]
    dat_out["filter-reason"] = "l2err > "+str(max_l2c1error)
    dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)
    # 4. add rho-over-k
    brats_data['rho-inv'] = brats_data['rho-inv'].astype('float')
    brats_data['k-inv']   = brats_data['k-inv'].astype('float')
    dat_out  = brats_data.loc[brats_data['k-inv'] <= 0]
    brats_data  = brats_data.loc[brats_data['k-inv'] >  0]
    dat_out["filter-reason"] = "k zero"
    dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)
    brats_data["rho-over-k"] = brats_data["rho-inv"]/brats_data["k-inv"]


    cols_bio   = ['#comp', 'age', 'srgy', '#wt/#b', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'l2Oc1', 'l2c1(TC,s)', 'I_EDc1', 'I_TCc1', 'I_B\WTc1', 'Ic0/Ic1']
    cols_bio0  = ['rho-inv', 'k-inv', 'l2Oc1', 'l2c1(TC,s)', 'I_EDc1', 'I_TCc1', 'I_B\WTc1', 'Ic0/Ic1']
    cols_bio1  = ['#comp', 'age', 'srgy', '#wt/#b', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'l2Oc1', 'l2c1(TC,s)', 'I_EDc1', 'I_TCc1', 'I_B\WTc1', 'Ic0/Ic1']
    cols_bio2  = ['#comp', 'age', 'srgy', '#wt/#b', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'I_EDc1', 'I_TCc1', 'I_B\WTc1', 'Ic0/Ic1']
    cols_bio3  = ['#comp', 'age', 'srgy', '#wt/#b', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'l2Oc1', 'l2c1(TC,s)', 'I_EDc1', 'I_TCc1', 'I_B\WTc1']
    cols_bio4  = ['#comp', 'age', 'srgy', '#wt/#b', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'I_EDc1', 'I_TCc1', 'I_B\WTc1', 'Ic0/Ic1']
    cols_bio5  = ['#comp', 'age', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'Ic0/Ic1']
    cols_bio6  = ['#comp', 'age', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'rho-over-k']
    cols_bio7  = ['#comp', 'age', '#ed/#b', '#tc/#b',            'k-inv', 'rho-over-k']
    # cols_stat  = ['#comp', 'age', '#wt/#b', '#ed/#b', '#tc/#b']
    cols_stat  = ['#comp', 'age', '#ed/#b', '#tc/#b']

    brats_data['is_na'] = brats_data[cols_bio].isnull().apply(lambda x: any(x), axis=1)
    brats_data          = brats_data.loc[brats_data['is_na'] == False]
    dat_out             = brats_data.loc[brats_data['is_na'] == True]
    dat_out["filter-reason"] = "nan values"
    dat_out             = pd.concat([dat_filtered_out, dat_out], axis=0)

    print("\n\n### BraTS simulation data [filtered] ### ")
    print(tabulate(brats_data, headers='keys', tablefmt='psql'))
    print("\n\n### BraTS simulation data [filtered out] ### ")
    print(tabulate(dat_filtered_out, headers='keys', tablefmt='psql'))

    nbshort = len(brats_data.loc[brats_data['srvl[]'] ==  1])
    nbmid   = len(brats_data.loc[brats_data['srvl[]'] ==  2])
    nblong  = len(brats_data.loc[brats_data['srvl[]'] ==  3])
    sum     = nbshort + nbmid + nblong
    weights = {}
    # weights[1] = 1. / nbshort;
    # weights[2] = 1. / nbmid;
    # weights[3] = 1. / nblong;
    for i in range(1,4):
        weights[i] = 1.
    print("work dataset of length %d/%d. short: %d, mid: %d, long: %d" % (len(brats_data),len(survival_data),nbshort,nbmid,nblong))

    # select features
    df_stat = brats_data[cols_stat]
    df_bio  = brats_data[cols_bio5]

    # df_bio.hist
    # plt.show()

    p_dict_svm = []
    # for cols, i in zip([cols_bio0,cols_bio1,cols_bio2,cols_bio3,cols_bio4,cols_bio5,cols_bio6,cols_stat], range(8)):
    for cols, i in zip([cols_bio6,cols_stat], range(2)):
    # for cols in [cols_bio5]:
        cc = "bio %d" % i if i < 7 else "stat";
        print(bcolors.OKBLUE, "=== %s ===" % cc, bcolors.ENDC)

        X = brats_data[cols].values
        Y = np.ravel(brats_data[['srvl[]']].values).astype('int')
        # Y = np.ravel(brats_data[['srvl']].values).astype('int')
        nfolds=5;


        # === data split ===
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)

        # === normalization/scaling ===
        # scaler    = preprocessing.StandardScaler().fit(X)
        scaler     = preprocessing.MinMaxScaler()
        normalizer = preprocessing.Normalizer()
        scaler.fit_transform(X_train)
        normalizer.fit(X_train)

        scaler.transform(X_test)
        normalizer.transform(X_train)
        normalizer.transform(X_test)

        # === dimensionality reduction ===
        pca = PCA(n_components=3, svd_solver='randomized', whiten=True).fit(X_train)
        # projecting data
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # X_train_pca = X_train
        # X_test_pca  = X_test

        print(X_train.shape, X_train_pca.shape)


        # #############################################################################
        # Train a SVM classification model
        # print("Fitting the classifier to the training set")
        # param_grid = {'C':  [1E-5,1E-4, 1E-3, 1E-2, 1E-1, 1, 1E1, 1E2, 1E3, 1E4, 1E5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        # clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
        # clf = clf.fit(X_train_pca, y_train)
        # print("Best estimator found by grid search:")
        # print(clf.best_estimator_)


        # === grid search for SVM ===
        # clf, opt_params = svc_param_selection(X_train_pca,y_train,nfolds,method='SVM',w=weights);
        clf             = SVC(kernel='linear', C=0.1, random_state=0).fit(X_train_pca, y_train)


        # === SVM Regressor ===
        # from sklearn.svm import SVR
        # clf, opt_params = svc_param_selection(X_train,y_train,nfolds,method='SVR',w=weights);
        # clf = SVR(gamma='scale', C=1.0, kernel='rbf', epsilon=0.2).fit(X_train, y_train)

        # ==== one vs rest linear svm ===
        # from sklearn import datasets
        # from sklearn.multiclass import OneVsOneClassifier
        # from sklearn.multiclass import OneVsRestClassifier
        # from sklearn.svm import LinearSVC
        # # clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
        # clf = OneVsRestClassifier(clf).fit(X_train, y_train)

        # === MLP neural network ===
        # from sklearn.neural_network import MLPClassifier
        # param_grid = {
        #     'hidden_layer_sizes': [(15,15,15), (15,), (100,), (100,5,5), (15,2), (10,), (10,5), (10,5,5), (5,5,5)],
        #     'activation': ['tanh', 'relu'],
        #     'solver': ['lbfgs', 'adam'],
        #     'alpha': 10.0 ** -np.arange(1, 7),
        #     'learning_rate': ['constant', 'invscaling', 'adaptive']
        # }
        # clf = GridSearchCV(MLPClassifier(random_state=1, max_iter=1000), param_grid, n_jobs=3, cv=nfolds)
        # # clf = MLPClassifier(random_state=1, max_iter=10000, activation= 'relu', alpha=1E-6, hidden_layer_sizes=(15,), learning_rate='constant', solver='lbfgs')
        # clf.fit(X_train, y_train)

        # print("Best parameters set found on development set:")
        # print()
        # print(clf.best_params_)
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            # print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test_pca)
        # print(classification_report(y_true, y_pred))
        print()
        print("y_true: ", y_true)
        print("y_pred: ", y_pred)

        # plt.step(np.arange(0,len(y_true),1), y_true-y_pred, color='b',where='mid')
        # plt.step(np.arange(0,len(y_true),1), y_pred, color='r',where='mid')
        # plt.show()

        print()
        # class_true = [getSurvivalClass(x) for x in y_true]
        # class_pred = [getSurvivalClass(x) for x in y_pred]
        # print("y_true: ", class_true)
        # print("y_pred: ", class_pred)
        # y_test = class_true
        # y_pred = class_pred
        print(classification_report(y_test, y_pred))

        accuracy = clf.score(X_test_pca, y_test)

        p_dict_svm.append(accuracy);

        print(bcolors.OKGREEN + "accuracy: ", accuracy, bcolors.ENDC)
        # print(bcolors.OKGREEN + "accuracy (rbf):    ", accuracy_rbf, bcolors.ENDC)

        # creating a confusion matrix
        class_names = [1,2,3]
        print("normalized confusion matrix")
        plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,title='normalized confusion matrix')

        print("\n========================================\n")
        # plt.show()

    p_dict = {}
    p_dict['svm'] = p_dict_svm;
    ddd =pd.DataFrame(p_dict);
    print("\n\n### Model Evaluation ### ")
    print(tabulate(ddd, headers='keys', tablefmt='psql'))







#
# ### Model Evaluation ###
# +----+----------+
# |    |      svm |
# |----+----------|
# |  0 | 0.5      |   # {'C': 0.001, 'kernel': 'linear'}                   # ['#comp', 'age', 'srgy', '#wt/#b', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'l2Oc1', 'l2c1(TC,s)', 'I_EDc1', 'I_TCc1', 'I_B\WTc1', 'Ic0/Ic1']
# |  1 | 0.558824 |   # {'C': 10.0, 'kernel': 'linear'}                    # ['#comp', 'age', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'Ic0/Ic1']
# |  2 | 0.294118 |   # {'C': 100000.0, 'gamma': 1e-06, 'kernel': 'rbf'}   # ['#comp', 'age', '#ed/#b', '#tc/#b', 'rho-inv', 'k-inv', 'rho-over-k']
# |  3 | 0.352941 |   # {'C': 1, 'gamma': 1e-05, 'kernel': 'rbf'}          # ['#comp', 'age',            '#tc/#b',           'k-inv', 'rho-over-k', 'Ic0/Ic1']
# |  4 | 0.5      |   # {'C': 0.001, 'kernel': 'linear'}                   # ['#comp', 'age', '#ed/#b', '#tc/#b']
# +----+----------+


# ### Model Evaluation ###
# +----+----------+
# |    |      svm |
# |----+----------|
# |  0 | 0.588235 | # {'C': 0.001, 'kernel': 'linear'}                   # ['#comp', 'age', '#ed/#b', '#tc/#b',            'k-inv', 'rho-over-k', 'Ic0/Ic1']
# |  1 | 0.5      | # {'C': 0.001, 'kernel': 'linear'}                   # ['#comp', 'age', '#ed/#b', '#tc/#b']
# +----+----------+
