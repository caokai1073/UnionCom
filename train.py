import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import random
import sys
from unioncom.PrimeDual import *
from unioncom.utils import save_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def train(Project, params, dataset, dist, P_joint, device):

	optimizer = optim.RMSprop(Project.parameters(), lr=params.lr)
	c_mse = nn.MSELoss()
	c_domain = nn.NLLLoss()
	cirterion_CE = nn.CrossEntropyLoss()

	Project.train()

	dataset_num = len(dataset)

	for i in range(dataset_num):
		P_joint[i] = torch.from_numpy(P_joint[i]).float().to(device)

	row = []
	col = []
	for i in range(dataset_num):
		row.append(np.shape(dataset[i])[0])
		col.append(np.shape(dataset[i])[1])

	N = np.int(np.max([len(l) for l in dataset]))

	dataset_anchor = []
	dist_anchor = []
	cor_pairs = []
	for i in range(dataset_num):
		random_anchor = random.sample(range(0,row[i]), int(row[i]))

		dataset_anchor.append(dataset[i][random_anchor])
		dataset_anchor[i] = torch.from_numpy(dataset_anchor[i]).to(device).float()
		
		anchor_num = np.int(row[i])
		dist_anchor.append(np.zeros([anchor_num, anchor_num]))

		for j in range(anchor_num):
			dist_anchor[i][j] = dist[i][random_anchor[j], random_anchor]

	for i in range(dataset_num-1):
		print("Match corresponding points between Dataset {} and Dataset {}".format(i+1, \
			len(dataset)))
		
		cor_pairs.append(cor_pairs_match_Adam(dist_anchor[i], dist_anchor[-1], N, \
				params, col[i], col[-1], device))
	
	print("Finished Matching!")
	print("Begin finding the embedded space")
	for epoch in range(params.epoch_DNN):
		len_dataloader = np.int(np.max(row)/params.batch_size)
		if len_dataloader == 0:
			print("Please set batch_size smaller!")
			sys.exit()
		for step in range(len_dataloader):
			KL_loss = []
			for i in range(dataset_num):
				random_batch = np.random.randint(0, row[i], params.batch_size)
				data = dataset[i][random_batch]
				data = torch.from_numpy(data).to(device).float()
				P_tmp = torch.zeros([params.batch_size, params.batch_size]).to(device)
				for j in range(params.batch_size):
					P_tmp[j] = P_joint[i][random_batch[j], random_batch]
				P_tmp = P_tmp / torch.sum(P_tmp)
				low_dim_data = Project(data, i)
				Q_joint = Q_tsne(low_dim_data)

				KL_loss.append(torch.sum(P_tmp * torch.log(P_tmp / Q_joint)))

			feature_loss = np.array(0)
			feature_loss = torch.from_numpy(feature_loss).to(device).float()
			for i in range(dataset_num-1):
				low_dim_anchor = Project(dataset_anchor[i], i)
				low_dim_anchor_biggest_dataset = Project(dataset_anchor[dataset_num-1][cor_pairs[i]], len(dataset)-1)
				feature_loss += c_mse(low_dim_anchor, low_dim_anchor_biggest_dataset)
				min_norm = torch.min(torch.norm(low_dim_anchor), torch.norm(low_dim_anchor_biggest_dataset))
				feature_loss += torch.abs(torch.norm(low_dim_anchor) - torch.norm(low_dim_anchor_biggest_dataset))/min_norm

			loss = params.beta * feature_loss
			for i in range(dataset_num):
				loss += KL_loss[i]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if (epoch+1) % params.log_DNN == 0:
			print("epoch:[{:d}/{}]: loss:{:4f}, align_loss:{:4f}".format(epoch+1, \
				params.epoch_DNN, loss.data.item(), feature_loss.data.item()))

	return Project

def neg_square_dists(X):
	sum_X = torch.sum(X*X, 1)
	tmp = torch.add(-2 * X.mm(torch.transpose(X,1,0)), sum_X)
	D = torch.add(torch.transpose(tmp,1,0), sum_X)

	return -D

def Q_tsne(Y):
	distances = neg_square_dists(Y)
	inv_distances = torch.pow(1. - distances, -1)
	inv_distances = inv_distances - torch.diag(inv_distances.diag(0))
	inv_distances = inv_distances + 1e-15
	return inv_distances / torch.sum(inv_distances)
























