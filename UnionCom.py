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
import torch
import torch.nn as nn
import torch.optim as optim
import random
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from visualization import visualize
from Model import model
from utils import *
from test import *

class UnionCom(object):

	"""
	UnionCom software for single-cell mulit-omics data integration
	Published at https://academic.oup.com/bioinformatics/article/36/Supplement_1/i48/5870490

	parameters:
	-----------------------------
	dataset: list of datasets to be integrated. [dataset1, dataset2, ...].
	epoch_pd: epoch of Prime-dual algorithm.
	epoch_DNN: epoch of training Deep Neural Network.
	epsilon: training rate of data matching matrix F.
	lr: training rate of DNN.
	batch_size: batch size of DNN.
	beta: trade-off parameter of structure preserving and matching.
	rho: damping term.
	log_DNN: log step of training DNN.
	log_pd: log step of prime dual method
	manual_seed: random seed.
	distance: mode of distance, ['geodesic, euclidean'], default is geodesic.
	output_dim: output dimension of integrated data.
	project:　mode of project, ['tsne', 'barycentric'], default is tsne.
	-----------------------------

	Functions:
	-----------------------------
	fit_transform(dataset)					find correspondence between datasets, 
								align multi-omics data in a common embedded space
	match(data)						find correspondence between datasets
	Prime_Dual(Kx, Ky, dx, dy)				Prime dual algorithm to find the optimal match
	project_barycentric(dataset, match_result)		barycentric projection (from SCOT)
	project_tsne(dataset, pairs_x, pairs_y, P_joint)	tsne-based projection
	Visualize(data, integrated_data, datatype, mode)	Visualization
	test_labelTA(integrated_data, datatype) 		test label transfer accuracy
	-----------------------------

	Examples:
	-----------------------------
	input: numpy arrays with rows corresponding to samples and columns corresponding to features
	output: integrated numpy arrays
	>>> data1 = np.loadtxt("./simu1/domain1.txt")
	>>> data2 = np.loadtxt("./simu1/domain2.txt")
	>>> type1 = np.loadtxt("./simu1/type1.txt")
	>>> type2 = np.loadtxt("./simu1/type2.txt")
	>>> type1 = type1.astype(np.int)
	>>> type2 = type2.astype(np.int)
	>>> uc = UnionCom()
	>>> integrated_data = uc.fit_transform(dataset=[data1,data2])
	>>> uc.test_labelTA(integrated_data, [type1,type2])
	>>> uc.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA')
	-----------------------------
	"""

	def __init__(self, epoch_pd=20000, epoch_DNN=200, \
		epsilon=0.001, lr=0.001, batch_size=100, rho=10, beta=1,\
		log_DNN=50, log_pd=1000, manual_seed=666, delay=0, kmax=40,  \
		output_dim=32, distance_mode ='geodesic', project_mode='tsne'):

		self.epoch_pd = epoch_pd
		self.epoch_DNN = epoch_DNN
		self.epsilon = epsilon
		self.lr = lr
		self.batch_size = batch_size
		self.rho = rho
		self.log_DNN = log_DNN
		self.log_pd = log_pd
		self.manual_seed = manual_seed
		self.delay = delay
		self.beta = beta
		self.kmax = kmax
		self.output_dim = output_dim
		self.distance_mode = 'geodesic'
		self.project_mode = 'tsne'
		self.row = []
		self.col = []
		self.dist = []
		self.kmin = []

	def fit_transform(self, dataset=None):
		"""
		find correspondence between datasets & align multi-omics data in a common embedded space
		"""

		time1 = time.time()
		init_random_seed(self.manual_seed)
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		dataset_num = len(dataset)

		#### compute the distance matrix
		print("Shape of Raw data")
		for i in range(dataset_num):
			self.row.append(np.shape(dataset[i])[0])
			self.col.append(np.shape(dataset[i])[1])
			print("Dataset {}:".format(i), np.shape(dataset[i]))
			
			dataset[i] = (dataset[i]- np.min(dataset[i])) / (np.max(dataset[i]) - np.min(dataset[i]))

			if self.distance_mode == 'geodesic':
				dist_tmp, k_tmp = geodesic_distances(dataset[i], self.kmax)
				self.dist.append(np.array(dist_tmp))
				self.kmin.append(k_tmp)

			if self.distance_mode == 'euclidean':
				dist_tmp, k_tmp = euclidean_distances(dataset[i])
				self.dist.append(np.array(dist_tmp))
				self.kmin.append(k_tmp)

		# find correspondence between samples
		pairs_x = []
		pairs_y = []
		match_result = self.match(dataset=dataset)
		for i in range(dataset_num-1):
			cost = np.max(match_result[i])-match_result[i]
			row_ind,col_ind = linear_sum_assignment(cost)
			pairs_x.append(row_ind)
			pairs_y.append(col_ind)

		#  projection
		if self.project_mode == 'tsne':
			P_joint = []
			for i in range(dataset_num):
				P_joint.append(p_joint(self.dist[i], self.kmin[i]))
			integrated_data = self.project_tsne(dataset, pairs_x, pairs_y, P_joint)
		else:
			integrated_data = self.project_barycentric(dataset, match_result)	

		print("---------------------------------")
		print("unionCom Done!")
		time2 = time.time()
		print('time:', time2-time1, 'seconds')

		return integrated_data

	def match(self, dataset):
		"""
		Find correspondence between multi-omics datasets
		"""

		dataset_num = len(dataset)
		cor_pairs = []
		N = np.int(np.max([len(l) for l in dataset]))
		for i in range(dataset_num-1):
			print("---------------------------------")
			print("Find correspondence between Dataset {} and Dataset {}".format(i+1, \
				len(dataset)))
			cor_pairs.append(self.Prime_Dual(self.dist[i], self.dist[-1], self.col[i], self.col[-1]))

		print("Finished Matching!")
		return cor_pairs

	def Prime_Dual(self, Kx, Ky, dx, dy):
		"""
		prime dual combined with Adam algorithm to find the local optimal soluation
		"""

		N = np.int(np.maximum(len(Kx), len(Ky)))
		print("use device:", self.device)
		Kx = Kx / N
		Ky = Ky / N
		Kx = torch.from_numpy(Kx).float().to(self.device)
		Ky = torch.from_numpy(Ky).float().to(self.device)
		m = np.shape(Kx)[0]
		n = np.shape(Ky)[0]
		F = np.zeros((m,n))
		F = torch.from_numpy(F).float().to(self.device)
		Im = torch.ones((m,1)).float().to(self.device)
		In = torch.ones((n,1)).float().to(self.device)
		Lambda = torch.zeros((n,1)).float().to(self.device)
		Mu = torch.zeros((m,1)).float().to(self.device)
		S = torch.zeros((n,1)).float().to(self.device)
		a = np.sqrt(dy/dx)
		pho1 = 0.9
		pho2 = 0.999
		delta = 10e-8
		Fst_moment = torch.zeros((m,n)).float().to(self.device)
		Snd_moment = torch.zeros((m,n)).float().to(self.device)
		i=0
		while(i<self.epoch_pd):

			### compute gradient
			grad = 4*torch.mm(F, torch.mm(Ky, torch.mm(torch.t(F), torch.mm(F, Ky)))) \
			- 4*a*torch.mm(Kx, torch.mm(F,Ky)) + torch.mm(Mu, torch.t(In)) \
			+ torch.mm(Im, torch.t(Lambda)) + self.rho*(torch.mm(F, torch.mm(In, torch.t(In))) - torch.mm(Im, torch.t(In)) \
			+ torch.mm(Im, torch.mm(torch.t(Im), F)) + torch.mm(Im, torch.t(S-In)))
			
			### adam momentum
			i += 1
			Fst_moment = pho1*Fst_moment + (1-pho1)*grad
			Snd_moment = pho2*Snd_moment + (1-pho2)*grad*grad
			hat_Fst_moment = Fst_moment/(1-np.power(pho1,i))
			hat_Snd_moment = Snd_moment/(1-np.power(pho2,i))
			grad = hat_Fst_moment/(torch.sqrt(hat_Snd_moment)+delta)
			F_tmp = F - grad
			F_tmp[F_tmp<0]=0
			
			### update 
			F = (1-self.epsilon)*F + self.epsilon*F_tmp

			### update slack variable
			grad_s = Lambda + self.rho*(torch.mm(torch.t(F), Im) - In + S)
			s_tmp = S - grad_s
			s_tmp[s_tmp<0]=0
			S = (1-self.epsilon)*S + self.epsilon*s_tmp

			### update dual variables
			Mu = Mu + self.epsilon*(torch.mm(F,In) - Im)
			Lambda = Lambda + self.epsilon*(torch.mm(torch.t(F), Im) - In + S)

			#### if scaling factor changes too fast, we can delay the update
			if i>=self.delay:
				a = torch.trace(torch.mm(Kx, torch.mm(torch.mm(F, Ky), torch.t(F)))) / \
				torch.trace(torch.mm(Kx, Kx))

			if (i+1) % self.log_pd == 0:
				norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Ky), torch.t(F)))
				print("epoch:[{:d}/{:d}] err:{:.4f} alpha:{:.4f}".format(i+1, self.epoch_pd, norm2.data.item(), a))

		F = F.cpu().numpy()
		# pairs = np.zeros(m)
		# for i in range(m):
		# 	pairs[i] = np.argsort(F[i])[-1]
		return F

	def project_barycentric(self, dataset, match_result):
		print("---------------------------------")
		print("Begin finding the embedded space")
		integrated_data = []
		for i in range(len(dataset)-1):
			integrated_data.append(np.matmul(match_result[i], dataset[-1]))
		integrated_data.append(dataset[-1])
		print("Done")
		return integrated_data

	def project_tsne(self, dataset, pairs_x, pairs_y, P_joint):
		"""
		tsne-based projection (nonlinear method) to match and preserve structures of different modalities.
		Here we provide a way using neural network to find the embbeded space. 
		However, traditional gradient descent method can also be used.
		"""

		print("---------------------------------")
		print("Begin finding the embedded space")

		net = model(self.col, self.output_dim)
		Project_DNN = init_model(net, self.device, restore=None)

		optimizer = optim.RMSprop(Project_DNN.parameters(), lr=self.lr)
		c_mse = nn.MSELoss()
		Project_DNN.train()

		dataset_num = len(dataset)

		for i in range(dataset_num):
			P_joint[i] = torch.from_numpy(P_joint[i]).float().to(self.device)
			dataset[i] = torch.from_numpy(dataset[i]).float().to(self.device)

		for epoch in range(self.epoch_DNN):
			len_dataloader = np.int(np.max(self.row)/self.batch_size)
			if len_dataloader == 0:
				len_dataloader = 1
				self.batch_size = np.max(self.row)
			for step in range(len_dataloader):
				KL_loss = []
				for i in range(dataset_num):
					random_batch = np.random.randint(0, self.row[i], self.batch_size)
					data = dataset[i][random_batch]
					P_tmp = torch.zeros([self.batch_size, self.batch_size]).to(self.device)
					for j in range(self.batch_size):
						P_tmp[j] = P_joint[i][random_batch[j], random_batch]
					P_tmp = P_tmp / torch.sum(P_tmp)
					low_dim_data = Project_DNN(data, i)
					Q_joint = Q_tsne(low_dim_data)

					## loss of structure preserving 
					KL_loss.append(torch.sum(P_tmp * torch.log(P_tmp / Q_joint)))

				## loss of structure matching 
				feature_loss = np.array(0)
				feature_loss = torch.from_numpy(feature_loss).to(self.device).float()
				for i in range(dataset_num-1):
					low_dim = Project_DNN(dataset[i][pairs_x[i]], i)
					low_dim_biggest_dataset = Project_DNN(dataset[dataset_num-1][pairs_y[i]], len(dataset)-1)
					feature_loss += c_mse(low_dim, low_dim_biggest_dataset)
					# min_norm = torch.min(torch.norm(low_dim), torch.norm(low_dim_biggest_dataset))
					# feature_loss += torch.abs(torch.norm(low_dim) - torch.norm(low_dim_biggest_dataset))/min_norm

				loss = self.beta * feature_loss
				for i in range(dataset_num):
					loss += KL_loss[i]

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			if (epoch+1) % self.log_DNN == 0:
				print("epoch:[{:d}/{}]: loss:{:4f}, align_loss:{:4f}".format(epoch+1, \
					self.epoch_DNN, loss.data.item(), feature_loss.data.item()))

		integrated_data = []
		for i in range(dataset_num):
			integrated_data.append(Project_DNN(dataset[i], i))
			integrated_data[i] = integrated_data[i].detach().cpu().numpy()
		print("Done")
		return integrated_data

	def Visualize(self, data, integrated_data, datatype=None, mode='PCA'):
		if datatype is not None:
			visualize(data, integrated_data, datatype, mode=mode)
		else:
			visualize(data, integrated_data, mode=mode)

	def test_LabelTA(self, integrated_data, datatype):

		test_UnionCom(integrated_data, datatype)


# if __name__ == '__main__':

	### batch correction for HSC data
	# data1 = np.loadtxt("./hsc/domain1.txt")
	# data2 = np.loadtxt("./hsc/domain2.txt")
	# type1 = np.loadtxt("./hsc/type1.txt")
	# type2 = np.loadtxt("./hsc/type2.txt")

	### UnionCom simulation
	# data1 = np.loadtxt("./simu2/domain1.txt")
	# data2 = np.loadtxt("./simu2/domain2.txt")
	# type1 = np.loadtxt("./simu2/type1.txt")
	# type2 = np.loadtxt("./simu2/type2.txt")
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
	# data1 = np.loadtxt("./scNMT/Paccessibility_300.txt")
	# data2 = np.loadtxt("./scNMT/Pmethylation_300.txt")
	# data3 = np.loadtxt("./scNMT/RNA_300.txt")
	# type1 = np.loadtxt("./scNMT/type1.txt")
	# type2 = np.loadtxt("./scNMT/type2.txt")
	# type3 = np.loadtxt("./scNMT/type3.txt")
	# not_connected, connect_element, index = Maximum_connected_subgraph(data3, params.kmax)
	# if not_connected:
	# 	data3 = data3[connect_element[index]]
	# 	type3 = type3[connect_element[index]]
	# min_max_scaler = preprocessing.MinMaxScaler()
	# data3 = min_max_scaler.fit_transform(data3)
	# print(np.shape(data3))
	#-------------------------------------------------------

	### integrate two datasets
	# type1 = type1.astype(np.int)
	# type2 = type2.astype(np.int)
	# uc = UnionCom()
	# integrated_data = uc.fit_transform(dataset=[data1,data2])
	# uc.test_LabelTA(integrated_data, [type1,type2])
	# uc.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA')

	### integrate three datasets
	# type1 = type1.astype(np.int)
	# type2 = type2.astype(np.int)
	# type3 = type3.astype(np.int)
	# datatype = [type1,type2,type3]
	# inte = fit_transform([data1,data2,data3])
	# test_label_transfer_accuracy(inte, datatype)
	# Visualize([data1,data2,data3], inte, datatype, mode='UMAP')
