import h5py
import os

# --> Folder of the HDF5 files
f = 'L1000-T40-A40/Data/boxes'

files = os.listdir(f)

for fi in files:
	print fi
	fi = os.path.join(f, fi)
	fi = h5py.File(fi, 'a')
	root = str(fi.keys()[0])
	vx = os.path.join(root, 'box0/vorticityX')
	vy = os.path.join(root, 'box0/vorticityY')
	vz = os.path.join(root, 'box0/vorticityZ')
	try:
		del fi[vx]
		del fi[vy]
		del fi[vz]
	except KeyError:
		continue
