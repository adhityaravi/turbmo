import numpy as np
from scipy.spatial import KDTree
from tools import *
import os


class PrepDataSet:
    """
       Class to facilitate the preparation of data set for building the machine learning model
       from the turbulent boundary layer data in numpy format. Shifts
       the sample along the surface wave in every time step.

       Input
       =====
       Initialize class with the following:
       * Path containing numpy files with feature data
       * Required features

       Data Structure
       ==============
       The data set will be structured as follows:

       [[F11(t1), F12(t1), .., F1n(t1), F21(t1), F22(t1), .., F2n(t1), .., Fx1(t1), Fx2(t1), .., Fxn(t1)],
        [F11(t2), F12(t2), .., F1n(t2), F21(t2), F22(t2), .., F2n(t2), .., Fx1(t1), Fx2(t1), .., Fxn(t1)],
        .
        .
        .
        .
        [F11(tm), F12(tm), .., F1n(tm), F21(tm), F22(tm), .., F2n(tm), .., Fx1(tm), Fx2(tm), .., Fxn(tm)]

        where,
        x - types of features
        n - number of features in every type
        m - number of snapshots

        So, Fab(c) represents an ath type Feature in of local index b in the time step c

        The prepared array is saved as a .npy file with a name 'dataset.npy'
    """

    def __init__(self, featureFilePath, gridFile, outputPath=None, feature=['p'], movePoints=False):
        """
            * featureFilePath - Path containing the simulation data in numpy format
            * gridFile - File containing the grid information
            * outputPath - Preferred output path for saving the data set (optional)
            * feature - List of required features in the data set
            * movePoints - Whether or not to move every point along the wave
        """

        self.featureFilePath = featureFilePath
        self.gridFile = gridFile
        self.outputPath = outputPath
        self.feature = feature
        self.movePoints = movePoints

        # --> Surface wave properties
        # --> Surface wave velocity
        uWave = 0.0553780945306
        # --> Time step
        dTWave = 9.40518328
        # --> Surface wave displacement
        self.dZ = uWave * dTWave

        # --> Load grid
        self._loadGrid()

        # --> Load flow data files
        self._loadFeatureFiles()

        # --> Build data set
        self._buildDataSet()

    def _loadGrid(self):
        """
        Load grid from the specified file
        """

        gridData = np.load(self.gridFile)
        self.x = gridData['x'][:, 0]
        self.z = gridData['z'][:, 0]

        self.xKept = self.x
        self.zKept = self.z

        self.zMin = min(self.z)
        self.zMax = max(self.z)

        data = zip(self.x, self.z)
        self.gridTree = KDTree(data)
        self.idxKept = self.gridTree.query(data)[1]

    def _loadFeatureFiles(self):
        """
        Load flow data files
        """

        self.featureFiles = os.listdir(self.featureFilePath)
        self.featureFiles = humanSort(self.featureFiles)

    def _updatePts(self):
        """
        Update grid points for the next time step
        Moves z coordinates by a length of dZ = Surface wave velocity / time step
        """

        # --> Move the z coordinates by dZ
        self.zKept = self.zKept + self.dZ

        # --> Find the points that are moved out of the grid and loop them back in from the front
        overShoot = np.argwhere(self.zKept > self.zMax)
        self.zKept[overShoot] = self.zMin + (self.zKept[overShoot] - self.zMax)

        # --> Find the nearest neighbour in the grid to the shifted points
        updatedPts = zip(self.xKept, self.zKept)
        selectedIdx = self.gridTree.query(updatedPts)[1]

        # --> Update the LHS
        self.zKept = self.z[selectedIdx]
        self.xKept = self.x[selectedIdx]
        self.idxKept = selectedIdx

    def _initializeDataSet(self):
        """
        Initialize data set matrix with a correct shape
        """

        self.nFeatureTypes = len(self.feature)
        foo = os.path.join(self.featureFilePath, self.featureFiles[0])
        foo = np.load(foo)
        self.nFeatures = foo[self.feature[0]].shape[0]

        nRows = len(self.featureFiles)
        nCols = self.nFeatures * self.nFeatureTypes

        self.dataSet = np.ndarray((nRows, nCols))

    def _buildDataSet(self):
        """
        Build data set
        """

        # --> Initialize the data set
        self._initializeDataSet()

        ctr = 1
        ctrEnd = len(self.featureFiles)

        # --> Loop over the snapshots
        for iFile in range(len(self.featureFiles)):
            print('\nWorking on {}. {} remaining'.format(iFile, (ctrEnd - ctr)))
            featureFile = os.path.join(self.featureFilePath, self.featureFiles[iFile])
            data = np.load(featureFile)

            # --> Loop over type different types of features
            for iType in range(self.nFeatureTypes):
                self.dataSet[iFile, (self.nFeatures * iType):(self.nFeatures * (iType + 1))] = \
                                                                            data[self.feature[iType]][self.idxKept, 0]

            # --> Shift the points along the moving wave for the next snapshot
            if self.movePoints:
                self._updatePts()
            ctr += 1

        # --> Save the dataset
        fileName = 'dataSet' if self.outputPath is None else os.path.join(self.outputPath, 'dataSet')
        np.save(fileName, self.dataSet)
