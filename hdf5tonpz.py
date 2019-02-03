import h5py as hd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI

# --> Parallel stuff
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
my_rank = comm.Get_rank()

def findNearest(array, value):
    """
    Find the index of the value nearest to the specified value in the given array
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if my_rank is 0:
    myY = 0.25
elif my_rank is 1:
    myY = 0.5
elif my_rank is 2:
    myY = 1.0
elif my_rank is 3:
    myY = 1.5

# --> HDF5 files
originalGrid = '/media/adhitya/Storage (A:)/TBL/grid/grid-cellCentered.hdf5'
originalFlow = '/media/adhitya/Storage (A:)/TBL/L1000-T40-A40/Data/boxes'
originalOut = '/media/adhitya/Storage (A:)/TBL/L1000-T40-A40/Data/out/Drag.dat'

# --> Numpy files
numpyGrid = '/media/adhitya/Storage (A:)/TBL/L1000-T40-A40/numpyData/y={}/Grid'.format(myY)
numpyFlow = '/media/adhitya/Storage (A:)/TBL/L1000-T40-A40/numpyData/y={}/Flow'.format(myY)
numpyOut = '/media/adhitya/Storage (A:)/TBL/L1000-T40-A40/numpyData/out'

# --> X-axis limits
xlim = [50, 100]

# --> y slice
y = myY

# --> Grid operations
# --> Loading hdf5 Grid
print('\nConverting Grid')
origGridData = hd.File(originalGrid, 'r')

# --> Loading x and z coordinates for y=0 slice
yC = origGridData['box0/y'][0,:,0]
yIdx = findNearest(yC, y)
xCoords = origGridData['box0/x'][:,yIdx,:]
zCoords = origGridData['box0/z'][:,yIdx,:]
#yCoords = origGridData['box0/y'][:,0:yIdx,:]

# --> Flatten the coordinates for ease of use
xCoords = xCoords.flatten()
zCoords = zCoords.flatten()
#yCoords = yCoords.flatten()

# --> Find the xCoords within specified limits
keepIdx = np.argwhere((xCoords>=xlim[0]) & (xCoords<=xlim[1]))

# --> Take the x and z coords within the limits
xCoords = xCoords[keepIdx]
zCoords = zCoords[keepIdx]
#yCoords = yCoords[keepIdx]

# --> Save the mesh
fileName = 'grid'
fileName = os.path.join(numpyGrid, fileName)
np.savez(fileName, x=xCoords, z=zCoords)

# --> Flow operations
fileNames = os.listdir(originalFlow)
globalTimeStep = []
iTime = 0

for fileName in fileNames:
    # --> Loading hdf5 Flow
    print('\nConverting {}'.format(fileName))
    fileName = os.path.join(originalFlow, fileName)
    origFlowData = hd.File(fileName, 'r')

    root = str(origFlowData.keys()[0])
    globalTimeStep.append([])
    globalTimeStep[iTime] = origFlowData[root].attrs['globalTimeStep'][0]
    iTime = iTime + 1

    # paths inside hdf5 file
    pPath = os.path.join(root, 'box0/p')
    rhoPath = os.path.join(root, 'box0/rho')
    voxPath = os.path.join(root, 'box0/vorticityX')
    voyPath = os.path.join(root, 'box0/vorticityY')
    vozPath = os.path.join(root, 'box0/vorticityZ')
    uPath = os.path.join(root, 'box0/u')
    vPath = os.path.join(root, 'box0/v')
    wPath = os.path.join(root, 'box0/w')

    # extract data
    p = origFlowData[pPath][:,yIdx,:]
    rho = origFlowData[rhoPath][:,yIdx,:]
    vorticityX = origFlowData[voxPath][:,yIdx,:]
    vorticityY = origFlowData[voyPath][:,yIdx,:]
    vorticityZ = origFlowData[vozPath][:,yIdx,:]
    u = origFlowData[uPath][:,yIdx,:]
    v = origFlowData[vPath][:,yIdx,:]
    w = origFlowData[wPath][:,yIdx,:]

    # --> Flatten the arrays
    p = p.flatten()
    rho = rho.flatten()
    vorticityX = vorticityX.flatten()
    vorticityY = vorticityY.flatten()
    vorticityZ = vorticityZ.flatten()
    u = u.flatten()
    v = v.flatten()
    w = w.flatten()

    # --> Fetch the values on the nodes within specified limits
    p = p[keepIdx]
    rho = rho[keepIdx]
    vorticityX = vorticityX[keepIdx]
    vorticityY = vorticityY[keepIdx]
    vorticityZ = vorticityZ[keepIdx]
    u = u[keepIdx]
    v = v[keepIdx]
    w = w[keepIdx]

    # --> Save the flow data
    fileName = os.path.basename(fileName)
    fileName = fileName.replace('.hdf5', '')
    fileName = os.path.join(numpyFlow, fileName)
    np.savez(fileName, p=p, rho=rho, vorticityY=vorticityY, u=u, v=v, w=w)

if my_rank is 0:
    # --> Response operations
    print('\nConverting Response')
    selectedResponses = []
    iResp = 0
    # --> Open the dat file
    with open(originalOut) as f:
        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            # --> Check if the global time step is available in the feature file
            if int(line[0]) in globalTimeStep:
                print('Time Step Selected - {}'.format(line[0]))
                selectedResponses.append([])
                # --> Response Cf = Cd / area (line[6] - Cd, line[9] - area)
                selectedResponses[iResp] = float(line[6]) / float(line[9])
                iResp = iResp + 1

    selectedResponses = np.asarray(selectedResponses)
    fileName = 'targetCf'
    fileName = os.path.join(numpyOut, fileName)
    np.save(fileName, selectedResponses)
