# --> Project specific tools
from tools import *
from rfpimp import *

# --> sklearn tools
from sklearn.ensemble import RandomForestRegressor

# --> General tools
from os import path
import os, sys
import time as ti
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.spatial import KDTree
import warnings

class VarRanking:
    """
    Class to perform feature ranking on the turbulent boundary layer data set

    Functionality
    =============
    * Build regression random forest from the data set
    * Perform out of bag scoring for different number of trees
    * Perform feature ranking of the available features using mean decrease in impurity or using mean decrease in
      accuracy

    Input : A dictionary with the following keys
    =====
    * featureFile - numpy file containing feature data matrix
    * gridFile - numpy file containing grid information
    * targetFile - numpy file containing the target array
    * featureName - list of the name/names of the feature required
    * respName - name of the response used    
    * nSamples - required number od samples per feature type
    * cmpOob - option to perform a out of bag score comparison for different number of trees
        options: True/False
    * nTrees - number of  trees to grow in the random forest regressor
        options: non zero int
    * rankingCriteria - criteria based on which feature ranking will be performed
        options: 'mda'(mean decrease in accuracy), 'mdi'(mean decrease in impurity)
    * plotPred - option to plot the predictions vs actual response
        options: True/False
    * show_type - option to show or save plots
        options: 'show', 'save'
    * save_path - path to save the output data
        default: None
    * saveTopFeatures - option to save the top features
        options: True, False
    * nTopFeatures - number of top features to be stored. If nTopFeatures is None, optimal number of top features will be chosen.
	options: int, None
    * verbose - option to print the progress of the process
        options: True, False
    * plotRanking - option to plot the variable ranking
        options: True, False
    * runName - name to save plot files with
    """

    def __init__(self, inputDict):
        """
        Constructor
        """

        # --> Extracting input variables
        self._extractInput(inputDict)

        # --> Initialize output paths
        self._initializePaths()

        if self.verbose is False:
            blockPrint()

        # --> Prepare data set, mesh points and target
        self._prepare()

        # --> Oob scoring
        if self.cmpOob is not False:
            self._compareOob()
            return

        # --> Feature ranking
        self._rankFeatures()

        enablePrint()


    def _extractInput(self, inputDict):
        """
        Extract the input variables
        """

        self.featureFile = inputDict['featureFile']
        self.gridFile = inputDict['gridFile']
        self.targetFile = inputDict['targetFile']
        self.featureName = inputDict['featureName']
        self.respName = inputDict['respName']
        self.nSamples = inputDict['nSamples']
        self.cmpOob = inputDict['cmpOob']
        self.nTrees = inputDict['nTrees']
        self.rankingCriteria = inputDict['rankingCriteria']
        self.plotPred = inputDict['plotPred']
        self.showType = inputDict['showType']
        self.savePath = inputDict['savePath']
        self.saveTopFeatures = inputDict['saveTopFeatures']
        self.nTopFeatures = inputDict['nTopFeatures']
        self.verbose = inputDict['verbose']
        self.runName = inputDict['runName']
        self.plotRanking = inputDict['plotRanking']
        self.markers = ['o', '*', '+', '^', 'v', '<', '>', '8', 's', 'p', 'h', 'd']
        self.scalingFactor = 1e5

    def _initializePaths(self):
        """
        Initialize the required output paths
        """
        self.savePathData = path.join(self.savePath, 'Data')
        self.savePathPlotPred = path.join(self.savePath, 'Prediction_Plots')
        self.savePathRanking = path.join(self.savePath, 'Ranking_Plots')

        
    def _prepare(self):
        """
        Prepare data set, mesh points, and target 
        """

        # --> Loading the data
        print('\n\tLoading data_set and grid points...')
        start_time = ti.time()
        self.dataSet = np.load(self.featureFile)
        foo = np.load(self.gridFile)
        if len(foo['x'].shape) > 1:
            self.x = foo['x'][:,0]
            self.z = foo['z'][:,0]
        else:
            self.x = foo['x']
            self.z = foo['z']
        # --> For plotting boundary
        self.xMin = min(self.x)
        self.xMax = max(self.x)
        self.zMin = min(self.z)
        self.zMax = max(self.z)
        print('\t\tDone is {} s'.format(ti.time() - start_time))

        print('\n\tLoading target...')
        start_time = ti.time()
        self.target = np.load(self.targetFile)
        print('\t\tDone is {} s'.format(ti.time() - start_time))

        # --> Generating LHS
        print('\n\tGenerating LHS...')
        start_time = ti.time()
        idxKept = dim_red_lhs(self.x, self.z, sample_size=self.nSamples)
        print('\t\tDone is {} s'.format(ti.time() - start_time))

        # --> Re-shaping dataset with LHS
        # --> Initializing infos for new dataset
        self.nFeatureTypes = len(self.featureName)
        nTotalFeatures = self.dataSet.shape[0]
        self.nFeaturesPerType = nTotalFeatures/self.nFeatureTypes
        qux = np.array([[]])

        print('\n\tReducing dimensions...')
        start_time = ti.time()
        self.x = self.x[idxKept]
        self.z = self.z[idxKept]
        for iFeatureType in range(self.nFeatureTypes):
            selectedColumns = idxKept + (iFeatureType * self.nFeaturesPerType)
            if qux.size is 0:
                qux = self.dataSet[:, selectedColumns]
            else:
                qux = np.append(qux, self.dataSet[:, selectedColumns], axis=1)
        self.dataSet = qux
        print('\t\tDone is {} s'.format(ti.time() - start_time))

        # --> Scale the dataset
        print('\n\tScaling Dataset...')
        start_time = ti.time()
        self.dataSet = self.dataSet * self.scalingFactor
        self.target = self.target * self.scalingFactor
        print('\t\tDone is {} s'.format(ti.time() - start_time))

    def _compareOob(self):
        """
        Compare oob score for different tree numbers
        """

        # --> Determine the number of trees to be grown in random forests
        print('\n\tBuilding oob Vs no. of trees graph...')
        start_time = ti.time()
        cmp_n_trees, cmp_oob_score, cmp_req_time = compare_oob_score(self.dataSet, self.target, plot_data=True,
                                                                     show_type=self.showType, save_path=self.savePath,
                                                                     save_format='png')
        print('\t\tDone in {} s'.format(ti.time() - start_time))

    def _mdaImportances(self, model):
        """
        Perform feature ranking based on mean decrease in accuracy
        """

        X = pd.DataFrame(self.dataSet)
        y = pd.DataFrame(self.target)

        ftrImpDFrame = oob_importances(model, X, y)

        ftr_imp = ftrImpDFrame.values
        idx = ftrImpDFrame.index.values

        foo = zip(idx, ftr_imp)
        foo = np.asarray(foo)
        foo = foo[foo[:, 0].argsort()]
        ftrImp = zip(*foo)[1]
        ftrImp = np.asarray(ftrImp)

        del X, y

        return ftrImp

    def _extractTopFeatures(self, rankedIdx, rankedFtrScore):
        """
        Extract n distinct top features : implemented with a closest neighbour check
        """

        # --> Tolerance in the distance between 2 successive features
        tol = 0.5
        # --> a global counter to check if 'n' top features are chosen
        globalCtr = 0

        # --> top most feature
        # --> Initialize the selected indices dict
        selectedFeatures = {}
        # --> Fetch the feature id
        featureType = int(rankedIdx[0]/self.nSamples)
        selectedFeatures[featureType] = {}
        # --> Add feature score, x, and z value of the top most feature to the dict
        selectedFeatures[featureType]['ftrScore'] = np.array([])
        selectedFeatures[featureType]['ftrScore'] = np.append(selectedFeatures[featureType]['ftrScore'], rankedFtrScore[0])
        # --> Since there is only a singe grid for multiple features, it is necessary to adjust the index value
        #     when fetching the grid
        gridIdx = rankedIdx[0] - (self.nSamples * featureType)
        selectedFeatures[featureType]['x'] = np.array([])
        selectedFeatures[featureType]['x'] = np.append(selectedFeatures[featureType]['x'], self.x[gridIdx])
        selectedFeatures[featureType]['z'] = np.array([])
        selectedFeatures[featureType]['z'] = np.append(selectedFeatures[featureType]['z'],self.z[gridIdx])
        globalCtr += 1

        # --> Loop through the indices arranged in descending order
        for iIdx in range(rankedIdx.shape[0]):

            # --> Check if the required number of top features are already selected
            if globalCtr < self.nTopFeatures:

                # --> Fetch the feature type
                featureType = int(rankedIdx[iIdx] / self.nSamples)

                # --> Check if the featureId already exists in the selected dict
                if featureType in selectedFeatures:

                    gridIdx = rankedIdx[iIdx] - (self.nSamples * featureType)

                    # --> Closest neighbour check
                    if (dist(self.x[gridIdx], self.z[gridIdx],
                             selectedFeatures[featureType]['x'][-1], selectedFeatures[featureType]['z'][-1])) > tol:

                        selectedFeatures[featureType]['ftrScore'] = np.append(selectedFeatures[featureType]['ftrScore'],
                                                                          rankedFtrScore[iIdx])
                        selectedFeatures[featureType]['x'] = np.append(selectedFeatures[featureType]['x'], self.x[gridIdx])
                        selectedFeatures[featureType]['z'] = np.append(selectedFeatures[featureType]['z'], self.z[gridIdx])

                        globalCtr += 1

                    else:
                        continue

                # --> Else, add the feature id to the selected dict
                else:
                    selectedFeatures[featureType] = {}

                    gridIdx = rankedIdx[iIdx] - (self.nSamples * featureType)

                    selectedFeatures[featureType]['ftrScore'] = np.array([])
                    selectedFeatures[featureType]['ftrScore'] = np.append(selectedFeatures[featureType]['ftrScore'], rankedFtrScore[iIdx])
                    selectedFeatures[featureType]['x'] = np.array([])
                    selectedFeatures[featureType]['x'] = np.append(selectedFeatures[featureType]['x'], self.x[gridIdx])
                    selectedFeatures[featureType]['z'] = np.array([])
                    selectedFeatures[featureType]['z'] = np.append(selectedFeatures[featureType]['z'], self.z[gridIdx])

                    globalCtr += 1

            else:
                break

        return selectedFeatures

    def _topFeatures(self):
        """
        Save n top features as a numpy file
        """

        # --> Sort the features in descending order (rank them)
        idx = np.arange(self.ftrScore.shape[0])
        ftrIdx = zip(self.ftrScore, idx)
        ftrIdx = -np.asarray(ftrIdx)
        ftrIdx = ftrIdx[ftrIdx[:, 0].argsort()]

        # --> Select optimal number of top features
        rankedFtrScore = zip(*ftrIdx)[0]
        rankedFtrScore = -np.asarray(rankedFtrScore)

        # --> Extract info of top features
        rankedIdx = zip(*ftrIdx)[1]
        rankedIdx = -np.asarray(rankedIdx)
        rankedIdx = rankedIdx.astype(int)
        if self.nTopFeatures is None:
            self.nTopFeatures = selectNTopFeatures(rankedFtrScore, 0.4, 0.01)
        self.topFeatures = self._extractTopFeatures(rankedIdx, rankedFtrScore)

        # --> Save top feature infos
        if self.saveTopFeatures:
            saveName = '{}_topFeatures'.format(self.runName)
            saveName = path.join(self.savePathData, saveName)
            np.save(saveName, self.topFeatures)

    def _rankFeatures(self):
        """
        Perform feature ranking
        """

        # --> Initialize random forest regressor
        regr = RandomForestRegressor(n_estimators=self.nTrees, random_state=0, oob_score=True)

        # --> Fit the forest to the data set
        print('\n\tFitting Trees...')
        start_time = ti.time()
        regr.fit(self.dataSet, self.target)
        print('\t\tDone in {}s'.format(ti.time() - start_time))
        score = regr.oob_score_
        print('\n\tOut of bag score = {}'.format(score))

        # --> Feature ranking
        if self.rankingCriteria is 'mdi':
            self.ftrScore = regr.feature_importances_
        elif self.rankingCriteria is 'mda':
            self.ftrScore = self._mdaImportances(regr)
        else:
            msg = '\nOption {} is not yet available'.format(self.rankingCriteria)
            raise NotAvailableException(msg)

        # --> Saving top features
        self._topFeatures()

        if self.plotPred:
            self.trgtPredicted = regr.predict(self.dataSet)
            plotPredictions(self.target, self.trgtPredicted, self.respName,
                            self.showType, self.runName, self.savePathPlotPred)

        if self.plotRanking:
            plotRanking(self.xMin, self.xMax, self.zMin, self.zMax, self.topFeatures, self.nFeatureTypes, self.markers,
                        self.featureName, self.showType, self.runName, self.savePathRanking)


class CumulativeRanking():
    """
        Class to perform a cumulative feature ranking of a set of top features stored

        Functionality:
        =============
        * Collect top features
        * Build dataset with just the top features
        * Rank the gathered features

        Input:
        =====
        Input dict with following keys

        * topFeaturePath: Path of the folder containing the stored top features
        * targetFile: numpy file containing the target
        * featureName - list of the name/names of the feature required
        * respName - name of the response variable
        * masterFeatureFile - numpy file containing the main dataset
        * gridFile - numpy file containing the grid info
        * nTrees - number of trees to be grown in the random forest
        * plotPred - option to plot the predictions vs actual response
            options: True/False
        * show_type - option to show or save plots
            options: 'show', 'save'
        * save_path - path to save the output data
            default: None
	* saveTopFeatures - option to save the top features
            options: True, False
    	* nTopFeatures - number of top features to be stored. If nTopFeatures is None, optimal number of top features will be chosen.
	    options: int, None
        * plotRanking - option to plot the variable ranking
            options: True, False
        * verbose - option to print the progress of the process
            options: True, False
    """

    def __init__(self, inputDict):
        """
        Constructor
        """

        # --> Extracting input variables
        self._extractInput(inputDict)

        if not self.verbose:
            blockPrint()

        # --> Prepare data set, mesh points and target
        self._prepare()

        # --> Feature ranking
        self._rankFeatures()


    def _extractInput(self, inputDict):
        """
        Extract the input variables
        """

        self.topFeaturePath = inputDict['topFeaturePath']
        self.masterDataSet = inputDict['masterFeatureFile']
        self.gridFile = inputDict['gridFile']
        self.targetFile = inputDict['targetFile']
        self.featureName = inputDict['featureName']
        self.respName = inputDict['respName']
        self.nTrees = inputDict['nTrees']
        self.plotPred = inputDict['plotPred']
        self.showType = inputDict['showType']
        self.savePath = inputDict['savePath']
        self.saveTopFeatures = inputDict['saveTopFeatures']
        self.nTopFeatures = inputDict['nTopFeatures']
        self.verbose = inputDict['verbose']
        self.runName = inputDict['runName']
        self.plotRanking = inputDict['plotRanking']
        self.markers = ['o', '*', '+', '^', 'v', '<', '>', '8', 's', 'p', 'h', 'd']
        self.scalingFactor = 1e5

    def _prepare(self):
        """
        Prepare the dataset
        """

        featureFiles = os.listdir(self.topFeaturePath)
        featureFiles = [f for f in featureFiles if f.endswith('.npy')]
        self.nFeatureType = len(self.featureName)

        # --> Load main dataset
        print('\n\tLoading data...')
        startTime = ti.time()
        masterDataSet = np.load(self.masterDataSet)
        masterNFeaturesPerType = masterDataSet.shape[1] / self.nFeatureType

        # --> Load master grid
        foo = np.load(self.gridFile)
        if len(foo['x'].shape) > 1:
            masterX = foo['x'][:,0]
            masterZ = foo['z'][:,0]
        else:
            masterX = foo['x']
            masterZ = foo['z']
        self.xMin = min(masterX)
        self.xMax = max(masterX)
        self.zMin = min(masterZ)
        self.zMax = max(masterZ)
        qux = zip(masterX, masterZ)
        gridTree = KDTree(qux)
        print('\t\tDone in {} s'.format(ti.time()-startTime))

        # Intialize feature information dict
        print('\n\tCumulating feature information...')
        startTime = ti.time()
        self.featureInfos = {}
        for iFeatureType in range(self.nFeatureType):
            self.featureInfos[iFeatureType] = {}
            self.featureInfos[iFeatureType]['x'] = np.array([])
            self.featureInfos[iFeatureType]['z'] = np.array([])
            self.featureInfos[iFeatureType]['nFtrsLocal'] = 0
            self.featureInfos[iFeatureType]['nFtrsGlobal'] = 0
            self.featureInfos[iFeatureType]['name'] = self.featureName[iFeatureType]
            self.featureInfos[iFeatureType]['idxKept'] = np.array([])

        # --> Loop through the saved feature files and extract the saved feature locations
        for iFile in featureFiles:
            iFile = os.path.join(self.topFeaturePath, iFile)
            data = np.load(iFile).item()

            for iFeatureType in range(self.nFeatureType):

                if iFeatureType in data:
                    self.featureInfos[iFeatureType]['x'] = np.append(self.featureInfos[iFeatureType]['x'], data[iFeatureType]['x'])
                    self.featureInfos[iFeatureType]['z'] = np.append(self.featureInfos[iFeatureType]['z'], data[iFeatureType]['z'])
                else:
                    continue

        # --> The grid locations of the gathered features
        self.x = np.array([])
        self.z = np.array([])

        # --> Find the index of the saved feature locations in the master dataset
        for iFeatureType in range(self.nFeatureType):
            fubar = zip(self.featureInfos[iFeatureType]['x'], self.featureInfos[iFeatureType]['z'])
            try:
                self.featureInfos[iFeatureType]['idxKept'] = np.append(self.featureInfos[iFeatureType]['idxKept'],
                                                                       gridTree.query(fubar)[1])
            except ValueError:
                msg = 'One or few of the mentioned features is not available in the data. If the results are weird' \
                      ', check the topFeaturePath and featureName keys.\n'
                warnings.warn(msg)
                continue
            self.featureInfos[iFeatureType]['idxKept'] = np.unique(self.featureInfos[iFeatureType]['idxKept'])
            self.featureInfos[iFeatureType]['idxKept'] = self.featureInfos[iFeatureType]['idxKept'].astype(int)
            self.x = np.append(self.x, masterX[self.featureInfos[iFeatureType]['idxKept']])
            self.z = np.append(self.z, masterZ[self.featureInfos[iFeatureType]['idxKept']])
            self.featureInfos[iFeatureType]['idxKept'] += (iFeatureType * masterNFeaturesPerType)
            self.featureInfos[iFeatureType]['nFtrsLocal'] = self.featureInfos[iFeatureType]['idxKept'].shape[0]
            self.featureInfos[iFeatureType]['nFtrsGlobal'] = self.featureInfos[iFeatureType]['nFtrsLocal']
            if iFeatureType > 0:
                self.featureInfos[iFeatureType]['nFtrsGlobal'] += self.featureInfos[iFeatureType-1]['nFtrsGlobal']
        print('\t\tDone in {} s'.format(ti.time()-startTime))

        # --> Build the new dataset
        print('\n\tBuilding new dataset...')
        startTime = ti.time()
        selectedColumns = np.array([])
        self.idxIdentifier = np.array([[]])
        for iFeatureType in range(self.nFeatureType):
            selectedColumns = np.append(selectedColumns, self.featureInfos[iFeatureType]['idxKept'])
        selectedColumns = selectedColumns.astype(int)
        self.dataSet = masterDataSet[:, selectedColumns]
        self.dataSet = self.dataSet * self.scalingFactor
        self.target = np.load(self.targetFile)
        self.target = self.target * self.scalingFactor
        print('\t\tDone in {} s'.format(ti.time() - startTime))

        masterDataSet = None

    def _fetchFeatureType(self, column):
        """
        Fetch the type to feature with the column number in the dataset
        """

        for iFeatureType in range(self.nFeatureType):
            if self.featureInfos[iFeatureType]['nFtrsLocal'] is not 0:
                if column < (self.featureInfos[iFeatureType]['nFtrsGlobal']):
                    featureType = iFeatureType
                    break
                else:
                    continue
            else:
                continue

        try:
            return featureType
        except UnboundLocalError:
            msg = 'Column number exceeds the dimension of the dataset'
            raise Exception(msg)

    def _extractTopFeatures(self, rankedIdx, rankedFtrScore):
        """
        Extract n distinct top features : implemented with a closest neighbour check
        """

        # --> Tolerance in the distance between 2 successive features
        tol = 0.5
        # --> a global counter to check if top features are chosen
        globalCtr = 0

        # --> top most feature
        # --> Initialize the selected indices dict
        selectedFeatures = {}
        # --> Fetch the feature id
        featureType = self._fetchFeatureType(rankedIdx[0])
        selectedFeatures[featureType] = {}
        # --> Add feature score, x, and z value of the top most feature to the dict
        selectedFeatures[featureType]['ftrScore'] = np.array([])
        selectedFeatures[featureType]['ftrScore'] = np.append(selectedFeatures[featureType]['ftrScore'],
                                                              rankedFtrScore[0])
        selectedFeatures[featureType]['x'] = np.array([])
        selectedFeatures[featureType]['x'] = np.append(selectedFeatures[featureType]['x'], self.x[rankedIdx[0]])
        selectedFeatures[featureType]['z'] = np.array([])
        selectedFeatures[featureType]['z'] = np.append(selectedFeatures[featureType]['z'], self.z[rankedIdx[0]])
        globalCtr += 1

        # --> Loop through the indices arranged in descending order
        for iIdx in range(1, rankedIdx.shape[0]):

            # --> Check if the required number of top features are already selected
            if globalCtr < self.nTopFeatures:
                # --> Fetch the feature type
                featureType = self._fetchFeatureType(rankedIdx[iIdx])

                # --> Check if the featureId already exists in the selected dict
                if featureType in selectedFeatures:

                    # --> Closest neighbour check
                    foo = np.array([])
                    for iPoint in range(selectedFeatures[featureType]['x'].shape[0]):
                        foo = np.append(foo, dist(selectedFeatures[featureType]['x'][iPoint], selectedFeatures[featureType]['z'][iPoint],
                                                  self.x[rankedIdx[iIdx]], self.z[rankedIdx[iIdx]]))
                    di = min(foo)

                    if di > tol:

                        selectedFeatures[featureType]['ftrScore'] = np.append(selectedFeatures[featureType]['ftrScore'],
                                                                              rankedFtrScore[iIdx])
                        selectedFeatures[featureType]['x'] = np.append(selectedFeatures[featureType]['x'],
                                                                       self.x[rankedIdx[iIdx]])
                        selectedFeatures[featureType]['z'] = np.append(selectedFeatures[featureType]['z'],
                                                                       self.z[rankedIdx[iIdx]])

                        globalCtr += 1

                    else:
                        continue

                # --> Else, add the feature id to the selected dict
                else:
                    selectedFeatures[featureType] = {}

                    selectedFeatures[featureType]['ftrScore'] = np.array([])
                    selectedFeatures[featureType]['ftrScore'] = np.append(selectedFeatures[featureType]['ftrScore'],
                                                                          rankedFtrScore[iIdx])
                    selectedFeatures[featureType]['x'] = np.array([])
                    selectedFeatures[featureType]['x'] = np.append(selectedFeatures[featureType]['x'], self.x[rankedIdx[iIdx]])
                    selectedFeatures[featureType]['z'] = np.array([])
                    selectedFeatures[featureType]['z'] = np.append(selectedFeatures[featureType]['z'], self.z[rankedIdx[iIdx]])

                    globalCtr += 1

            else:
                break

        return selectedFeatures

    def _topFeatures(self):
        """
        Select top features
        """
        # --> Sort the features in descending order (rank them)
        idx = np.arange(self.ftrScore.shape[0])
        ftrIdx = zip(self.ftrScore, idx)
        ftrIdx = -np.asarray(ftrIdx)
        ftrIdx = ftrIdx[ftrIdx[:, 0].argsort()]

        # --> Select optimal number of top features
        rankedFtrScore = zip(*ftrIdx)[0]
        rankedFtrScore = -np.asarray(rankedFtrScore)

        # --> Extract info of top features
        rankedIdx = zip(*ftrIdx)[1]
        rankedIdx = -np.asarray(rankedIdx)
        rankedIdx = rankedIdx.astype(int)

        if self.nTopFeatures is 'all':
            self.nTopFeatures = rankedIdx.shape[0]
        elif self.nTopFeatures is None:
            self.nTopFeatures = selectNTopFeatures(rankedFtrScore, 0.4, 0.01)
        self.topFeatures = self._extractTopFeatures(rankedIdx, rankedFtrScore)

        # --> Save top feature infos
        if self.saveTopFeatures:
            saveName = '{}_topFeatures'.format(self.runName)
            saveName = path.join(self.savePath, saveName)
            np.save(saveName, self.topFeatures)

    def _rankFeatures(self):
        """
        Perform feature ranking
        """

        # --> Initialize random forest regressor
        regr = RandomForestRegressor(n_estimators=self.nTrees, random_state=0, oob_score=True)

        # --> Fit the forest to the data set
        print('\n\tFitting Trees...')
        start_time = ti.time()
        regr.fit(self.dataSet, self.target)
        print('\t\tDone in {}s'.format(ti.time() - start_time))
        score = regr.oob_score_
        print('\n\tOut of bag score = {}'.format(score))

        # --> Feature ranking
        self.ftrScore = regr.feature_importances_

        # --> Saving top features
        self._topFeatures()

        # --> Prediction plot
        if self.plotPred:
            self.trgtPredicted = regr.predict(self.dataSet)
            plotPredictions(self.target, self.trgtPredicted, self.respName,
                            self.showType, self.runName, self.savePath)

        # --> Ranking plot
        if self.plotRanking:
            plotRanking(self.xMin, self.xMax, self.zMin, self.zMax, self.topFeatures, self.nFeatureType, self.markers,
                        self.featureName, self.showType, self.runName, self.savePath)


class NotAvailableException(Exception):
    pass

