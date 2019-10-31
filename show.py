import umap
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# data1 = np.loadtxt("./result/new_data1.txt")
# row1 = np.shape(data1)[0]
# data2 = np.loadtxt("./result/new_data2.txt")
# row2 = np.shape(data2)[0]
# data = np.vstack((data1, data2))
# type1 = np.loadtxt("./simu2/type1.txt")
# type2 = np.loadtxt("./simu2/type2.txt")

# data1_org = np.loadtxt("./simu2/data1.txt")
# data2_org = np.loadtxt("./simu2/data2_0.txt")
# fig = plt.figure()
# styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 
# for i in range(3):
# 	index1 = np.where(type1==i)	
# 	plt.scatter(data1_org[index1,0], data1_org[index1,1],c=styles[i], s=5.)
# fig = plt.figure()	
# ax = Axes3D(fig)
# for i in range(4):
# 	index2 = np.where(type2==i)
# 	ax.scatter(data2_org[index2,0], data2_org[index2,1],data2_org[index2,2], c=styles[i], s=5.)
# plt.show()

# # embedding = umap.UMAP(n_components=2).fit_transform(data)
# embedding = TSNE(n_components=2).fit_transform(data)
# embedding1 = embedding[0:row1,:]
# embedding2 = embedding[row1:row1+row2,:]

# fig = plt.figure()
# plt.scatter(embedding1[:,0], embedding1[:,1],c=[1,0.5,0], s=5.)
# plt.scatter(embedding2[:,0], embedding2[:,1],c=[0.2,0.4,0.1], s=5.)

# fig = plt.figure()
# styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 
# for i in range(3):
# 	index1 = np.where(type1==i)	
# 	plt.scatter(embedding1[index1,0], embedding1[index1,1],c=styles[i], s=5.)
	
# for i in range(4):
# 	index2 = np.where(type2==i)
# 	plt.scatter(embedding2[index2,0], embedding2[index2,1],c=styles[i], s=5.)
# plt.show()

data1 = np.loadtxt("./result/new_data1.txt")
row1 = np.shape(data1)[0]
data2 = np.loadtxt("./result/new_data2.txt")
row2 = np.shape(data2)[0]
data = np.vstack((data1, data2))
# data = np.loadtxt("./result/domain_scAlign.txt")
print(np.shape(data))
type1 = np.loadtxt("./real_data/type1.txt")
type2 = np.loadtxt("./real_data/type2.txt")

data1_org = np.loadtxt("./real_data/real_domain1.txt")
data2_org = np.loadtxt("./real_data/real_domain2.txt")
# data_org = np.vstack((data1_org, data2_org))
# embedding_org = umap.UMAP(n_components=2).fit_transform(data_org)
# embedding_org = TSNE(n_components=2).fit_transform(data_org)
# embedding1_org = embedding_org[0:row1,:]
# embedding2_org = embedding_org[row1:row1+row2,:]
# embedding1_org = umap.UMAP(n_components=2).fit_transform(data1_org)
# embedding2_org = umap.UMAP(n_components=2).fit_transform(data2_org)
# embedding1_org = TSNE(n_components=2).fit_transform(data1_org)
# embedding2_org = TSNE(n_components=2).fit_transform(data2_org)
# fig = plt.figure()
# styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 
# for i in range(7):
# 	index1 = np.where(type1==i)	
# 	plt.scatter(embedding1_org[index1,0], embedding1_org[index1,1],c=styles[i], s=5.)
# fig = plt.figure()	
# for i in range(7):
# 	index2 = np.where(type2==i)
# 	plt.scatter(embedding2_org[index2,0], embedding2_org[index2,1],c=styles[i], s=5.)
# plt.show()

embedding = umap.UMAP(n_components=2).fit_transform(data)
# embedding = TSNE(n_components=2).fit_transform(data)
embedding1 = embedding[0:row1,:]
embedding2 = embedding[row1:row1+row2,:]
# embedding2 = embedding[0:1510,:]
# embedding1 = embedding[1510:1510+509,:]

fig = plt.figure()
plt.scatter(embedding1[:,0], embedding1[:,1],c=[1,0.5,0], s=5.)
plt.scatter(embedding2[:,0], embedding2[:,1],c=[0.2,0.4,0.1], s=5.)

fig = plt.figure()
styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 
for i in range(7):
	index1 = np.where(type1==i)	
	plt.scatter(embedding1[index1,0], embedding1[index1,1],c=styles[i], s=5.)
	
for i in range(7):
	index2 = np.where(type2==i)
	plt.scatter(embedding2[index2,0], embedding2[index2,1],c=styles[i], s=5.)
plt.show()

# data1 = np.loadtxt("data1.txt")
# data2 = np.loadtxt("data2_0.txt")
# type1 = np.loadtxt("type1.txt")
# type2 = np.loadtxt("type2.txt")	
# type1_predict = np.loadtxt("type1_predict.txt")
# print(np.shape(data1))
# print(np.shape(data2))
# fig = plt.figure()
# styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 
# for i in range(1,7):
# 	index1 = np.where(type1==i)	
# 	plt.scatter(data1[index1,0], data1[index1,1],c=styles[i-1], s=5.)
# fig = plt.figure()	
# for i in range(1,7):
# 	index1 = np.where(type1_predict==i)	
# 	plt.scatter(data1[index1,0], data1[index1,1],c=styles[i-1], s=5.)
# fig = plt.figure()	
# for i in range(1,7):
# 	index2 = np.where(type2==i)
# 	plt.scatter(data2[index2,0], data2[index2,1],c=styles[i-1], s=5.)
# plt.show()
