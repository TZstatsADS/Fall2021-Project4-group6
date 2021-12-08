# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:22:50 2021

@author: olp
"""
#####################generate_synthetic_data
import math
import numpy as np
import matplotlib.pyplot as plt # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def generate_synthetic_data(plot_data=False):

    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """

    n_samples = 1000 # generate these many data points per class
    disc_factor = math.pi / 4.0 # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1) # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1) # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = range(0,n_samples*2)
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    
    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)


    """ Generate the sensitive feature here """
    x_control = [] # this array holds the sensitive feature value
    for i in range (0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)
        
        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s
        p2 = p2/s
        
        r = np.random.uniform() # generate a random number from 0 to 1

        if r < p1: # the first cluster is the positive class
            x_control.append(1.0) # 1.0 means its male
        else:
            x_control.append(0.0) # 0.0 -> female

    x_control = np.array(x_control)

    """ Show the data """
    if plot_data:
        num_to_draw = 200 # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]
        plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label= "Prot. +ve")
        plt.scatter(X_s_0[y_s_0==-1.0][:, 0], X_s_0[y_s_0==-1.0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label = "Prot. -ve")
        plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label = "Non-prot. +ve")
        plt.scatter(X_s_1[y_s_1==-1.0][:, 0], X_s_1[y_s_1==-1.0][:, 1], color='red', marker='o', facecolors='none', s=30, label = "Non-prot. -ve")

        
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        plt.xlim((-15,10))
        plt.ylim((-10,15))
        plt.savefig("img/data.png")
        plt.show()

    x_control = {"s1": x_control} # all the sensitive features are stored in a dictionary
    return X,y,x_control



#####decision_boundary_demo
from __future__ import division
import os,sys
import numpy as np
from generate_synthetic_data import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # for plotting stuff
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
from plot_synthetic_boundaries import plot_data

import stats_pref_fairness as compute_stats # for computing stats
from linear_clf_pref_fairness import LinearClf


def print_stats_and_plots(x,y,x_sensitive, clf, fname):

    dist_arr, dist_dict = clf.get_distance_boundary(x, x_sensitive)
    acc, _, acc_stats = compute_stats.get_clf_stats(dist_arr, dist_dict, y, x_sensitive, print_stats=True)

    if isinstance(clf.w, dict):
        w_arr = [clf.w[0], clf.w[1]]
        lt_arr = ['c--', 'b--']
        label_arr = [
                        "$\mathcal{B}_0: %0.2f; \mathcal{B}_1: %0.2f$" % (acc_stats[0][0]["frac_pos"], acc_stats[1][0]["frac_pos"]),
                        "$\mathcal{B}_0: %0.2f; \mathcal{B}_1: %0.2f$" % (acc_stats[0][1]["frac_pos"], acc_stats[1][1]["frac_pos"]),
                    ]
    else:
        w_arr = [clf.w]
        lt_arr = ['k--']
        label_arr = [
                "$\mathcal{B}_0: %0.2f; \mathcal{B}_1: %0.2f$" % (acc_stats[0][0]["frac_pos"], acc_stats[1][1]["frac_pos"]),
            ]

    title = "$Acc: %0.2f$" % acc
    plot_data(x, y, x_sensitive, w_arr, label_arr, lt_arr, fname, title)




def test_synthetic_data():
    
    """ Generate the synthetic data """
    data_type = 1
    X, y, x_sensitive = generate_synthetic_data(data_type=data_type, n_samples=1000) # set plot_data to False to skip the data plot
    X = compute_stats.add_intercept(X)

    """ Split the data into train and test """
    TEST_FOLD_SIZE = 0.3
    x_train, x_test, y_train, y_test, x_sensitive_train, x_sensitive_test =  train_test_split(X, y, x_sensitive, test_size=TEST_FOLD_SIZE, random_state=1234, shuffle=False)
    
    # show the data
    plot_data(x_test, y_test, x_sensitive_test, None, None, None, "img/data.png", None)





    """ Training the classifiers """
    # Classifier parameters 
    loss_function = "logreg" # perform the experiments with logistic regression
    EPS = 1e-4

    

    cons_params = {}
    cons_params["EPS"] = EPS
    cons_params["cons_type"] = -1 # no constraint



    print "\n\n== Unconstrained classifier =="
    # Train a classifier for each sensitive feature group separately optimizing accuracy for the respective group    
    clf_group = {} # will store the classifier for eah group here
    lam = {0:0.01, 1:0.01} # the regularization parameter -- we set small values here, in the paper, we cross validate all of regularization parameters
    for s_attr_val in set(x_sensitive_train):
        idx = x_sensitive_train==s_attr_val # the index for the current sensitive feature group
        clf = LinearClf(loss_function, lam=lam[s_attr_val], train_multiple=False) # for more info, see the clf documentation in the respective file
        clf.fit(x_train[idx], y_train[idx], x_sensitive_train[idx], cons_params)
        clf_group[s_attr_val] = clf

    # For simplicity of computing stats, we merge the two trained classifiers
    clf_merged = LinearClf(loss_function, lam=lam, train_multiple=True) 
    clf_merged.w = {0:None, 1:None}
    for s_attr_val in set(x_sensitive_train):
        clf_merged.w[s_attr_val] = clf_group[s_attr_val].w

    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf_merged, "img/unconstrained.png")



    
    print "\n\n== Parity classifier =="
    cons_params["cons_type"] = 0
    clf = LinearClf(loss_function, lam=0.01, train_multiple=False)
    clf.fit(x_train, y_train, x_sensitive_train, cons_params)
    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf, "img/parity.png")

    # compute the proxy value, will need this for the preferential classifiers
    dist_arr,dist_dict=clf.get_distance_boundary(x_train, x_sensitive_train)
    s_val_to_cons_sum_di = compute_stats.get_sensitive_attr_cov(dist_dict) # will need this for applying preferred fairness constraints



    
    print "\n\n\n\n== Preferred impact classifier =="


    # Not all values of the lambda satisfy the constraints empirically (in terms of acceptace rates)
    # This is because the scale (or norm) of the group-conditional classifiers can be very different from the baseline parity classifier, and from each other. This affects the distance from boundary (w.x) used in the constraints.
    # We use a hold out set with different regaularizer values to validate the norms that satisfy the constraints. Check the appendix of our NIPS paper for more details. 
    
    cons_params["cons_type"] = 1
    cons_params["tau"] = 1 # the tau value varies based on the dataset we look at. See the DCCP documentation for details
    cons_params["s_val_to_cons_sum"] = s_val_to_cons_sum_di
    lam = {0:2.5, 1:0.01} 
    clf = LinearClf(loss_function, lam=lam, train_multiple=True)
    clf.fit(x_train, y_train, x_sensitive_train, cons_params)
    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf, "img/preferred_impact.png")    


    
    print "\n\n\n\n== Preferred treatment AND preferred impact classifier =="
    cons_params["cons_type"] = 3
    cons_params["s_val_to_cons_sum"] = s_val_to_cons_sum_di
    lam = {0:2.5, 1:0.35}
    clf = LinearClf(loss_function, lam=lam, train_multiple=True)
    clf.fit(x_train, y_train, x_sensitive_train, cons_params)
    print_stats_and_plots(x_test, y_test, x_sensitive_test, clf, "img/preferred_both.png")    
    


def main():
    test_synthetic_data()


if __name__ == '__main__':
    main()


###generate_synthetic_data
import sys
import math
import numpy as np
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data

def generate_synthetic_data(data_type, n_samples):

    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """

    # in the same program run, we might generate this data multiple times (once for single and once for two models)
    # so make sure that the seed is always the same
    SEED = 1234
    seed(SEED)
    np.random.seed(SEED)

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y
    

    if data_type == 1:
    
        disc_factor = math.pi / 8.0 # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination



        """ Generate the non-sensitive features randomly """
        # We will generate one gaussian cluster for each class
        mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
        mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
        nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1) # positive class
        nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1) # negative class

        # join the posisitve and negative class clusters
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))

        # shuffle the data
        perm = range(0,n_samples*2)
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        
        rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
        X_aux = np.dot(X, rotation_mult)


        """ Generate the sensitive feature here """
        x_control = [] # this array holds the sensitive feature value
        for i in range (0, len(X)):
            x = X_aux[i]

            # probability for each cluster that the point belongs to it
            p1 = nv1.pdf(x)
            p2 = nv2.pdf(x)
            
            # normalize the probabilities from 0 to 1
            s = p1+p2
            p1 = p1/s
            p2 = p2/s
            
            r = np.random.uniform() # generate a random number from 0 to 1

            if r < p1: # the first cluster is the positive class
                x_control.append(1.0) # 1.0 means its male
            else:
                x_control.append(0.0) # 0.0 -> female

        x_control = np.array(x_control)

    elif data_type == 2:

        ## More datasets here.
        pass



    return X,y,x_control


###plot_synthetic_boundaries
import matplotlib
import matplotlib.pyplot as plt # for plotting stuff
import os
import numpy as np

matplotlib.rcParams['text.usetex'] = True # for type-1 fonts

def get_line_coordinates(w, x1, x2):
    y1 = (-w[0] - (w[1] * x1)) / w[2]
    y2 = (-w[0] - (w[1] * x2)) / w[2]    
    return y1,y2

def plot_data(X, y, x_sensitive, w_arr, label_arr, lt_arr, fname, title, group=None):


    # print fp_fn_arr
    plt.figure()
    num_to_draw = 200 # we will only draw a small number of points to avoid clutter
    fs = 20 # font size for labels and legends

    x_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]
    x_sensitive_draw = x_sensitive[:num_to_draw]


    x_lim = [min(x_draw[:,-2]) - np.absolute(0.3*min(x_draw[:,-2])), max(x_draw[:,-2]) + np.absolute(0.5 * max(x_draw[:,-2]))]
    y_lim = [min(x_draw[:,-1]) - np.absolute(0.3*min(x_draw[:,-1])), max(x_draw[:,-1]) + np.absolute(0.7 * max(x_draw[:,-1]))]

    X_s_0 = x_draw[x_sensitive_draw == 0.0]
    X_s_1 = x_draw[x_sensitive_draw == 1.0]
    y_s_0 = y_draw[x_sensitive_draw == 0.0]
    y_s_1 = y_draw[x_sensitive_draw == 1.0]

    if w_arr is not None: # we are plotting the boundaries of a trained classifier
        plt.scatter(X_s_0[y_s_0==1.0][:, -2], X_s_0[y_s_0==1.0][:, -1], color='green', marker='x', s=70, linewidth=2)
        plt.scatter(X_s_0[y_s_0==-1.0][:, -2], X_s_0[y_s_0==-1.0][:, -1], color='red', marker='x', s=70, linewidth=2)
        plt.scatter(X_s_1[y_s_1==1.0][:, -2], X_s_1[y_s_1==1.0][:, -1], color='green', marker='o', facecolors='none', s=70, linewidth=2)
        plt.scatter(X_s_1[y_s_1==-1.0][:, -2], X_s_1[y_s_1==-1.0][:, -1], color='red', marker='o', facecolors='none', s=70, linewidth=2)


        for i in range(0, len(w_arr)):
            w = w_arr[i]
            l = label_arr[i]
            lt = lt_arr[i]

            x1,x2 = min(x_draw[:,1]), max(x_draw[:,1])
            y1,y2 = get_line_coordinates(w, x1, x2)

            plt.plot([x1,x2], [y1,y2], lt, linewidth=3, label = l)


        plt.title(title, fontsize=fs)

    else: # just plotting the data
        plt.scatter(X_s_0[y_s_0==1.0][:, -2], X_s_0[y_s_0==1.0][:, -1], color='green', marker='x', s=70, linewidth=2, label= "group-0 +ve")
        plt.scatter(X_s_0[y_s_0==-1.0][:, -2], X_s_0[y_s_0==-1.0][:, -1], color='red', marker='x', s=70, linewidth=2, label= "group-0 -ve")
        plt.scatter(X_s_1[y_s_1==1.0][:, -2], X_s_1[y_s_1==1.0][:, -1], color='green', marker='o', facecolors='none', s=70, linewidth=2, label= "group-1 +ve")
        plt.scatter(X_s_1[y_s_1==-1.0][:, -2], X_s_1[y_s_1==-1.0][:, -1], color='red', marker='o', facecolors='none', s=70, linewidth=2, label= "group-1 -ve")


    if True: # turn the ticks on or off
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.legend(loc=2, fontsize=fs)
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    

    plt.savefig(fname)
    

    plt.show()





















