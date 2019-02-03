from twoD.varRanking import *
from mpi4py import MPI
import time as ti
import sys

# --> Parallel stuff
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
myRank = comm.Get_rank()

# --> Stage 1 ranking

inputDict = {}
inputDict['featureFile'] = '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L1000-T40-A40/y=1.5/All/dataSet.npy'
inputDict['gridFile'] = '/media/adhitya/Storage (A:)/TBL/L1000-T40-A40/numpyData/y=1.5/Grid/grid.npz'
inputDict['targetFile'] = '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L1000-T40-A40/target.npy'
inputDict['featureName'] = ['p', 'v', 'w', 'vorticityY']
inputDict['respName'] = 'Cf'
inputDict['nSamples'] = 100
inputDict['cmpOob'] = True
inputDict['nTrees'] = 200
inputDict['plotPred'] = False
inputDict['showType'] = 'show'
inputDict['rankingCriteria'] = 'mdi'
inputDict['savePath'] = '/home/adhitya/StudienArbeit/Results/TBL/Moving/L1000-T40-A40/y=1.5/All/'
inputDict['saveTopFeatures'] = True
inputDict['nTopFeatures'] = 20
inputDict['verbose'] = False
inputDict['runName'] = 'run0'
inputDict['plotRanking'] = False

# --> Number of iterations to run
nIter = 1

# --> Starting iteration number
start_iter = 0

# --> Thread specific iteration numbers
nIter_per_proc = nIter / world_size
start = (myRank * nIter_per_proc) + start_iter
stop = start + nIter_per_proc

if myRank is 0:
    print('\n------ Recursive Variable Ranking ------\n')

comm.Barrier()

# -- Variable ranking
for i in range(start, stop):

    inputDict['runName'] = 'run{}'.format(i)
    print('Thread {} starting stage 1 {}'.format(myRank, inputDict['runName']))
    start_time = ti.time()
    VarRanking(inputDict)
    print('\nThread {} completed {} in {} s\n'.format(myRank, inputDict['runName'], ti.time()-start_time))

comm.Barrier()

if myRank is 0:
    print('\n------ End ------\n')

# --> Stage 2 ranking

if myRank is 0:

    print('\n------ Cumulative Variable Ranking ------\n')

    inputDict['topFeaturePath'] = '/home/adhitya/StudienArbeit/Results/TBL/Moving/L1000-T40-A40/y=0.25/Pressure'
    inputDict['masterFeatureFile'] = '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L1000-T40-A40/y=0.25/Pressure/dataSet.npy'
    inputDict['featureName'] = ['p']
    inputDict['gridFile'] = '/media/adhitya/Storage (A:)/TBL/L1000-T40-A40/numpyData/y=0.25/Grid/grid.npz'
    inputDict['targetFile'] = '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L1000-T40-A40/target.npy'
    inputDict['respName'] = 'Cf'
    inputDict['nTrees'] = 200
    inputDict['saveTopFeatures'] = True
    inputDict['nTopFeatures'] = 40
    inputDict['plotPred'] = True
    inputDict['showType'] = 'save'
    inputDict['savePath'] = '/home/adhitya/StudienArbeit/Results/TBL/Moving/L1000-T40-A40/y=0.25/Pressure/'
    inputDict['verbose'] = True
    inputDict['runName'] = sys.argv[1]
    inputDict['plotRanking'] = True

    CumulativeRanking(inputDict)

    print('\n------ End ------\n')
