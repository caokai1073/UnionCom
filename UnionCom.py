'''
---------------------
UnionCom fucntions
author: Kai Cao
e-mail:caokai@amss.ac.cn
MIT LICENSE
---------------------
'''
import os
import sys
import time
import numpy as np
from Project import project_tsne, project_barycentric
from Match import match
from visualization import visualize
from utils import *
from test import *
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
# print(os.getcwd())

class params():
	epoch_pd = 20000
	epoch_DNN = 200
	epsilon = 0.001
	lr = 0.001
	batch_size = 100
	rho = 10
	log_DNN = 50
	log_pd = 1000
	manual_seed = 666
	delay = 0
	kmax = 20
	beta = 1
	col = []
	row = []
	output_dim = 32
	
def fit_transform(dataset, epoch_pd=20000, epoch_DNN=200, \
	epsilon=0.001, lr=0.001, batch_size=100, rho=10, beta=1,\
	log_DNN=50, log_pd=1000, manual_seed=666, delay=0, kmax=40,  \
	output_dim=32, distance = 'geodesic', project='tsne'):

	'''
	parameters:
	dataset: list of datasets to be integrated. [dataset1, dataset2, ...].
	epoch_pd: epoch of Prime-dual algorithm.
	epoch_DNN: epoch of training Deep Neural Network.
	epsilon: training rate of data matching matrix F.
	lr: training rate of DNN.
	batch_size: training batch size of DNN.
	beta: trade-off parameter of structure preserving and point matching.
	rho: training damping term.
	log_DNN: log step of training DNN.
	log_pd: log step of prime dual method
	manual_seed: random seed.
	distance: mode of distance, ['geodesic, euclidean'], default is geodesic.
	output_dim: output dimension of integrated data.
	project:　mode of project, ['tsne', 'barycentric'], default is tsne.
	---------------------
	'''
	params.epoch_pd = epoch_pd
	params.epoch_DNN = epoch_DNN
	params.epsilon = epsilon
	params.lr = lr
	params.batch_size = batch_size
	params.rho = rho
	params.log_DNN = log_DNN
	params.log_pd = log_pd
	params.manual_seed = manual_seed
	params.delay = delay
	params.beta = beta
	params.kmax = kmax
	params.output_dim = output_dim

	time1 = time.time()
	init_random_seed(manual_seed)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dataset_num = len(dataset)

	row = []
	col = []
	dist = []
	kmin = []

	#### compute the distance matrix
	print("Shape of Raw data")
	for i in range(dataset_num):
		row.append(np.shape(dataset[i])[0])
		col.append(np.shape(dataset[i])[1])
		print("Dataset {}:".format(i), np.shape(dataset[i]))
		
		dataset[i] = (dataset[i]- np.min(dataset[i])) / (np.max(dataset[i]) - np.min(dataset[i]))

		if distance == 'geodesic':
			dist_tmp, k_tmp = geodesic_distances(dataset[i], params.kmax)
			dist.append(np.array(dist_tmp))
			kmin.append(k_tmp)

		if distance == 'euclidean':
			dist_tmp, k_tmp = euclidean_distances(dataset[i])
			dist.append(np.array(dist_tmp))
			kmin.append(k_tmp)

	params.row = row
	params.col = col

	# find correspondence between cells
	pairs_x = []
	pairs_y = []
	match_result = match(params, dataset, dist, device)
	for i in range(dataset_num-1):
		cost = np.max(match_result[i])-match_result[i]
		row_ind,col_ind = linear_sum_assignment(cost)
		pairs_x.append(row_ind)
		pairs_y.append(col_ind)

	#  projection
	if project == 'tsne':

		P_joint = []
		for i in range(dataset_num):
			P_joint.append(p_joint(dist[i], kmin[i]))
		integrated_data = project_tsne(params, dataset, pairs_x, pairs_y, dist, P_joint, device)

	else:
		integrated_data = project_barycentric(dataset, match_result)	

	print("---------------------------------")
	print("unionCom Done!")
	time2 = time.time()
	print('time:', time2-time1, 'seconds')

	return integrated_data

def Visualize(data, integrated_data, datatype=None, mode='PCA'):

	if datatype is not None:
		visualize(data, integrated_data, datatype, mode=mode)
	else:
		visualize(data, integrated_data, mode=mode)

def test_label_transfer_accuracy(integrated_data, datatype):

	test_UnionCom(integrated_data, datatype)

### UnionCom simulation
# data1 = np.loadtxt("./simu1/domain1.txt")
# data2 = np.loadtxt("./simu1/domain2.txt")
# type1 = np.loadtxt("./simu1/type1.txt")
# type2 = np.loadtxt("./simu1/type2.txt")
#-------------------------------------------------------

### MMD-MA simulation
# data1 = np.loadtxt("./MMD/s3_mapped1.txt")
# data2 = np.loadtxt("./MMD/s3_mapped2.txt")
# type1 = np.loadtxt("./MMD/s3_type1.txt")
# type2 = np.loadtxt("./MMD/s3_type2.txt")
#-------------------------------------------------------

### scGEM data
# data1 = np.loadtxt("./scGEM/GeneExpression.txt")
# data2 = np.loadtxt("./scGEM/DNAmethylation.txt")
# type1 = np.loadtxt("./scGEM/type1.txt")
# type2 = np.loadtxt("./scGEM/type2.txt")
#-------------------------------------------------------

### scNMT data
data1 = np.loadtxt("./scNMT/Paccessibility_300.txt")
data2 = np.loadtxt("./scNMT/Pmethylation_300.txt")
data3 = np.loadtxt("./scNMT/RNA_300.txt")
type1 = np.loadtxt("./scNMT/type1.txt")
type2 = np.loadtxt("./scNMT/type2.txt")
type3 = np.loadtxt("./scNMT/type3.txt")

not_connected, connect_element, index = Maximum_connected_subgraph(data3, params.kmax)

if not_connected:
	data3 = data3[connect_element[index]]
	type3 = type3[connect_element[index]]

min_max_scaler = preprocessing.MinMaxScaler()
data3 = min_max_scaler.fit_transform(data3)
print(np.shape(data3))
#-------------------------------------------------------

# type1 = type1.astype(np.int)
# type2 = type2.astype(np.int)
# datatype = [type1,type2]
# inte = fit_transform([data1,data2])
# test_label_transfer_accuracy(inte, datatype)
# Visualize([data1,data2], inte, datatype, mode='PCA')

type1 = type1.astype(np.int)
type2 = type2.astype(np.int)
type3 = type3.astype(np.int)
datatype = [type1,type2,type3]
inte = fit_transform([data1,data2,data3])
test_label_transfer_accuracy(inte, datatype)
Visualize([data1,data2,data3], inte, datatype, mode='UMAP')
