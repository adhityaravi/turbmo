from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import time as ti
from os import listdir
from os import path
import re
from math import sqrt
import os, sys

def compare_oob_score(data_set, target, max_estimators=500, plot_data=False, show_type='show', save_path=None,
                      save_format='png', run_name=None):
    """
    Compare random forests with different number of estimators using oob score

    Input
    =====
    * data_set - data set to train the model
    * target - target to train the model
    * max_estimators - maximum number of trees to be used to build the random forest
        (optional - default = 500)
    * plot_data - plotting option to visualize the results of comparison
        (optional - default = False)

    Output
    ======
    * x_estimators - array of number of estimators used
    * oob_score - out of bag score for each number of estimator used
    * req_time - time required to build each number of estimators

    """

    n_rows = data_set.shape[0]
    n_columns = data_set.shape[1]

    # --> Initializing score and time list
    oob_score = []
    req_time = []
    x_estimators = []
    ctr = 0

    for i_estimators in range(20, max_estimators+1, 20):

        oob_score.append([])
        req_time.append([])
        x_estimators.append([])

        start_time = ti.time()
        regr = RandomForestRegressor(n_estimators=i_estimators, random_state=0, oob_score=True)
        regr.fit(data_set, target)
        score = regr.oob_score_
        stop_time = ti.time()

        oob_score[ctr] = score
        req_time[ctr] = stop_time - start_time
        x_estimators[ctr] = i_estimators
        ctr = ctr + 1

    if plot_data is True:
        plt.plot(x_estimators, oob_score, 'r-')
        plt.xlabel('Number of trees')
        plt.ylabel('Out of bag score')
        plt.title('Oob Score Comparison for a data set with {} rows and {} columns'.format(n_rows, n_columns))
        plt.ticklabel_format(useOffset=False)
        if show_type is 'show':
            plt.show()
        elif show_type is 'save':
            plt_name = 'OOB_score_comparison'
            if run_name is not None:
                plt_name = run_name + '_' + plt_name
            if save_path is not None:
                plt_name = path.join(save_path, plt_name)
            plt.savefig(plt_name, format=save_format)
        plt.close()

        plt.plot(x_estimators, req_time, 'b-')
        plt.xlabel('Number of trees')
        plt.ylabel('Time taken (in s)')
        plt.title('Timing Comparison')
        if show_type is 'show':
            plt.show()
        elif show_type is 'save':
            plt_name = 'Timing_comparison'
            if run_name is not None:
                plt_name = run_name + '_' + plt_name
            if save_path is not None:
                plt_name = path.join(save_path, plt_name)
            plt.savefig(plt_name, format=save_format)
        plt.close()

    return x_estimators, oob_score, req_time


def dim_red_rds(n_features, sample_size=1000):
    """
    Dimensionality reduction using random discreet sampling of the nodes

    Input
    =====
    * n_features - number of features in the data set
    * sample_size - number of features required
        (optional - default = 1000)

    Output
    ======
    * idx - indices of the randomly selected features

    """

    # --> Random sampling of the features
    idx = np.random.randint(0, n_features-1, sample_size)

    return idx


def multi_dim_lhs(n, samples=None, lim=None):
    """
    Generate randomized latin hypercube samples between the specified limits modified from pyDOE.lhs

    Input
    =====
    * n - dimension for latin hyper cube sampling
    * samples - number of lhs required
        (optional - default = n)
    * lim - limits between which lhs have to be generated
        (optional - default = [0, 1]

    Output
    ======
    * H - latin hyper cube samples

    Note: limits can be specified for each dimension, or a single limit can be specified which will be used for all the
    dimensions

    Eg: lim = [2, 3]
        lhs = multi_dim_lhs(n=2, samples=3, lim=lim)
        In this case for both dimensions, the limit [2, 3] will be used

        lim = [[1, 2], [2, 3]]
        lhs = multi_dim_lhs(n=2, samples=3, lim=lim)
        In this case, lim[0, :] will be used for 1st dimension and lhs[1, :] will be used for 2nd.

    """

    if lim is None:
        lim = np.ndarray([n, 2])
        lim[:, 0] = 0
        lim[:, 1] = 1

    if lim is not None:
        lim = np.asarray(lim)
        if len(lim.shape) == 1:
            li = lim
            lim = np.ndarray([n, 2])
            lim[:, 0] = li[0]
            lim[:, 1] = li[1]

    if samples is None:
        samples = n

    # --> Initializing points
    rdpoints = np.ndarray([samples, n])

    for j in range(n):
        cut = np.linspace(lim[j, 0], lim[j, 1], samples+1)

        # Fill points uniformly in each interval
        u = np.random.rand(samples)
        a = cut[:samples]
        b = cut[1:samples + 1]

        rdpoints[:, j] = u * (b - a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    return H


def dim_red_lhs(x_coords, z_coords, sample_size=1000):
    """
    Dimensionality reduction using latin hypercube sampling

    Input
    =====
    * x_coords - x coordinates of the mesh
    * z_coords - z coordinates of the mesh
    * sample_size - number of features required

    Output
    ======
    * idx - indices of the features selected using latin hypercube sampling

    """

    # --> Generate latin hyper cube samples between required limits
    x_lim = [50, 100]
    z_lim = [0, 21.60]
    lim = [x_lim, z_lim]

    lhs = multi_dim_lhs(2, sample_size, lim)
    pts = zip(lhs[:, 0], lhs[:, 1])

    # --> Using KDTree to find the points in the grid closest to the generated latin hypercube samples
    data = zip(x_coords, z_coords)
    tree = KDTree(data)
    idx = tree.query(pts)[1]

    return idx


def humanSort(list):
    """
    Perform alphanumeric human sort or natural sort on a list
    """

    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    list.sort(key=alphanum)
    return list


def blockPrint():
    """
    Ignore the printing statements
    """

    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    """Restore printing capability"""

    sys.stdout = sys.__stdout__


def plotPredictions(target, trgtPredicted, respName, showType, runName, savePath):
    """
    Plotting predictions
    """

    # --> Prediction accuracy plot
    target = target/1e5
    trgtPredicted = trgtPredicted/1e5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trgtPredicted, 'r-', label='Predicted')
    line, = ax.plot(target, 'b-', label='True')
    line.set_dashes([8, 4, 2, 4, 2, 4])
    ax.legend(loc=1)
    #plt.title('Original {} Vs Predicted {}'.
     #         format(respName, respName))
    plt.xlabel('Time Step')
    plt.ylabel('{} Value'.format(respName))
    plt.legend(['Predicted', 'Original'])

    if showType is 'show':
        plt.show()
    elif showType is 'save':
        pltName = 'Accuracy.png'
        pltName = runName+'_'+pltName
        pltName = path.join(savePath, pltName)
        plt.savefig(pltName, dpi=600, bbox_inches='tight')

    plt.close()


def findMinMaxScore(topFeatures):
    """
    Find the minimum and maximum score of selected top features
    """
    score = np.array([])

    for key in topFeatures.keys():
        score = np.append(score, topFeatures[key]['ftrScore'])

    sMin = min(score)
    sMax = max(score)

    return sMin, sMax


def plotRanking(xMin, xMax, zMin, zMax, topFeatures, nFeatureTypes, markers, featureName, showType, runName, savePath):
    """
    Plotting the score of top features
    """

    # --> Feature score plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --> Boundary plot
    xB = [xMin, xMax, xMax, xMin, xMin]
    zB = [zMin, zMin, zMax, zMax, zMin]
    ax.plot(xB, zB, 'k')

    # --> Fetch the colorbar limits for scatter plot
    cBarMin, cBarMax = findMinMaxScore(topFeatures)

    # --> Score plot
    for iFeatureType in range(nFeatureTypes):
        try:
            xS = topFeatures[iFeatureType]['x']
            zS = topFeatures[iFeatureType]['z']
            cS = topFeatures[iFeatureType]['ftrScore']

            sctr = ax.scatter(xS, zS, c=cS, cmap='rainbow', vmin=cBarMin, vmax=cBarMax,
                              marker=markers[iFeatureType], label=featureName[iFeatureType], s=4)
        except KeyError:
            continue

    # --> Add legend
    ax.legend(bbox_to_anchor=(0.95, 1), fontsize=5, markerscale=0.8, framealpha=1)

    plt.axis('scaled')

    # --> Add color bar
    cbar = plt.colorbar(sctr, fraction=0.016, pad=0.1)
    cbar.set_label('Feature score')

    # --> Title
    plt.title('Feature Ranking')
    plt.xlabel('x')
    plt.ylabel('z')

    if showType is 'show':
        plt.show()
    elif showType is 'save':
        pltName = 'Ranking.png'
        pltName = runName+'_'+pltName
        pltName = path.join(savePath, pltName)
        plt.savefig(pltName, dpi=600, bbox_inches='tight')

    plt.close()


def dist(x1, y1, x2, y2):
    """Distance between 2 points (x1, y1) and (x2, y2)"""

    return sqrt(((x2 - x1)**2) + ((y2 - y1)**2))


def selectNTopFeatures(ftrImp, limit, tol):
    """
    Function to select the optimal number of top features

    Input:
    * limit - percentage of the top most feature score at which feature gathering will be stopped
    * tol - difference between 2 consecutive features
    """
    limit = max(ftrImp) * limit
    nTopFeatures = 0

    for i in range(ftrImp.shape[0]-1):

        if ftrImp[i] < limit:
            if (ftrImp[i] - ftrImp[i+1]) > tol:
                nTopFeatures += 1
            else:
                break
        else:
            nTopFeatures += 1

    return nTopFeatures
