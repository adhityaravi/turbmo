from twoD.prepDataSet import PrepDataSet
from mpi4py import MPI

# --> Parallel stuff
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
my_rank = comm.Get_rank()

if my_rank is 0:

    # --> Flow data
    fFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=0.25/Flow'
    # --> Grid data
    gFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=0.25/Grid/grid.npz'

    PrepDataSet(fFile, gFile, '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L500-T40-A30/y=0.25/', ['p', 'vorticityY', 'v', 'w'], True)

elif my_rank is 2:

    fFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=0.5/Flow'
    gFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=0.5/Grid/grid.npz'

    PrepDataSet(fFile, gFile, '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L500-T40-A30/y=0.5/', ['p', 'vorticityY', 'v', 'w'], True)

elif my_rank is 1:

    fFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=1.0/Flow'
    gFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=1.0/Grid/grid.npz'

    PrepDataSet(fFile, gFile, '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L500-T40-A30/y=1.0/', ['p', 'vorticityY', 'v', 'w'], True)

elif my_rank is 3:

    fFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=1.5/Flow'
    gFile = '/media/adhitya/Storage (A:)/TBL/L500-T40-A30/numpyData/y=1.5/Grid/grid.npz'

    PrepDataSet(fFile, gFile, '/media/adhitya/Storage (A:)/TBL/Datasets/Moving/L500-T40-A30/y=1.5/', ['p', 'vorticityY', 'v', 'w'], True)
