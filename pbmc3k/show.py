import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap

data1 = np.loadtxt("real_domain1.txt")
data2 = np.loadtxt("real_domain2.txt")
type1 = np.loadtxt("type1.txt")
type2 = np.loadtxt("type2.txt")

umap_data1 = umap.UMAP(n_components=2).fit_transform(data1)
umap_data2 = umap.UMAP(n_components=2).fit_transform(data2)
fig = plt.figure()
styles1 = ['r^', 'g^', 'b^', 'y^'] 
styles2 = ['rx', 'gx', 'bx', 'yx']
for i in range(4):
	index1 = np.where(type1==i)
	plt.plot(umap_data1[index1,0], umap_data1[index1,1], styles1[i], alpha=0.5)
fig = plt.figure()
for i in range(4):
	index2 = np.where(type2==i)
	plt.plot(umap_data2[index2,0], umap_data2[index2,1], styles2[i], alpha=0.5)
plt.show()