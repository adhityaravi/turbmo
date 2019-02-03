import numpy as np
import os
import h5py

def calcGradCa3D(X,U):
	"""
	Calculate gradient - cartesian 3d
	
	X - grid data 
	U - flow data
	"""
	
	nx = X.shape[2]
	ny = X.shape[1]
	nz = X.shape[0]
	
	nVar = X.shape[3]
	
	grad = np.ndarray([nz,ny,nx, 3, nVar])
	
	dx = X[0,0,1:nx,0] - X[0,0,0:nx-1,0]
	dy = X[0,1:ny,0,1] - X[0,0:(ny-1),0,1]
	dz = X[1:nz,0,0,2] - X[0:(nz-1),0,0,2]

	for iVar in range(nVar):
	
		# x axis
		axis=2
		# interior
		for ix in range(1, nx-1):
			grad[:,:,ix,axis,iVar] = (U[:,:,ix+1,iVar] - U[:,:,ix-1,iVar])/(dx[ix-1] + dx[ix])
		# x = 0
		grad[:,:,0,axis,iVar] = (U[:,:,1,iVar] - U[:,:,0,iVar])/dx[0]
		# x = xN
		grad[:,:,nx-1,axis,iVar] = (U[:,:,nx-1,iVar] - U[:,:,nx-2,iVar])/dx[nx-2]
		
		# y axis
		axis = 1
		# interior
		for iy in range(1, ny-1):
			grad[:,iy,:,axis,iVar] = (U[:,iy+1,:,iVar] - U[:,iy-1,:,iVar])/(dy[iy-1] + dy[iy])
		# y = 0
		grad[:,0,:,axis,iVar] = (U[:,1,:,iVar] - U[:,0,:,iVar])/dy[0]
		# y = yN
		grad[:,ny-1,:,axis,iVar] = (U[:,ny-1,:,iVar] - U[:,ny-2,:,iVar])/dy[ny-2]
		
		# z axis
		axis = 0
		# interior
		for iz in range(1, nz-1):
			grad[iz,:,:,axis,iVar] = (U[iz+1,:,:,iVar] - U[iz-1,:,:,iVar]) / (dz[iz-1] + dz[iz])
		# z = 0
		grad[0,:,:,axis,iVar] = (U[1,:,:,iVar] - U[0,:,:,iVar]) / dz[0]
		# z = zN
		grad[nz-1,:,:,axis,iVar] = (U[nz-1,:,:,iVar] - U[nz-2,:,:,iVar])/dz[nz-2]
		
	return grad

# grid file
gridF = 'grid/grid-cellCentered.hdf5'
# flow data directory
flowDir = 'tmp/'

# load and stack grid
grid = h5py.File(gridF)
x = grid['box0/x']
y = grid['box0/y']
z = grid['box0/z']
X = np.stack((x, y, z), axis=3)

snapshots = os.listdir(flowDir)

ctr = 1
ctrEnd = len(snapshots)

for snap in snapshots:

	print('Working on {}. {} files remaining'.format(snap, (ctrEnd-ctr)))
		
	# open hdf5 file
	snap = os.path.join(flowDir, snap)
	f = h5py.File(snap, 'a')
	
	# define paths inside the hdf5 file
	root = str(f.keys()[0])
	uP = os.path.join(root, 'box0/u')
	vP = os.path.join(root, 'box0/v')
	wP = os.path.join(root, 'box0/w')
	voxP = os.path.join(root, 'box0/vorticityX')
	voyP = os.path.join(root, 'box0/vorticityY')
	vozP = os.path.join(root, 'box0/vorticityZ')
	
	# load and stack velocities
	u = f[uP]
	v = f[vP]
	w = f[wP]
	U = np.stack((u, v, w), axis=3)
	
	# calculate gradient
	grad = calcGradCa3D(X, U)

	# calculate vorticity
	vorticityX = grad[:,:,:, 1,2] - grad[:,:,:, 0,1]
	vorticityY = grad[:,:,:, 0,0] - grad[:,:,:, 2,2]
	vorticityZ = grad[:,:,:, 2,1] - grad[:,:,:, 1,0]
	
	# add vorticities and close the hdf5 file
	f.create_dataset(voxP, data=vorticityX)
	f.create_dataset(voyP, data=vorticityY)
	f.create_dataset(vozP, data=vorticityZ)
	
	f.close()
	
	ctr += 1
	
