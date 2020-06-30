import sys
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import random
from Model import model
from utils import init_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def project_tsne(params, dataset, pairs, dist, P_joint, device):
	print("---------------------------------")
	print("Begin finding the embedded space")

	net = model(params.col, params.output_dim)
	Project_DNN = init_model(net, device, restore=None)

	optimizer = optim.RMSprop(Project_DNN.parameters(), lr=params.lr)
	c_mse = nn.MSELoss()
	Project_DNN.train()

	dataset_num = len(dataset)

	for i in range(dataset_num):
		P_joint[i] = torch.from_numpy(P_joint[i]).float().to(device)
		dataset[i] = torch.from_numpy(dataset[i]).float().to(device)

	for epoch in range(params.epoch_DNN):
		len_dataloader = np.int(np.max(params.row)/params.batch_size)
		if len_dataloader == 0:
			len_dataloader = 1
			params.batch_size = np.max(params.row)
		for step in range(len_dataloader):
			KL_loss = []
			for i in range(dataset_num):
				random_batch = np.random.randint(0, params.row[i], params.batch_size)
				data = dataset[i][random_batch]
				P_tmp = torch.zeros([params.batch_size, params.batch_size]).to(device)
				for j in range(params.batch_size):
					P_tmp[j] = P_joint[i][random_batch[j], random_batch]
				P_tmp = P_tmp / torch.sum(P_tmp)
				low_dim_data = Project_DNN(data, i)
				Q_joint = Q_tsne(low_dim_data)

				KL_loss.append(torch.sum(P_tmp * torch.log(P_tmp / Q_joint)))

			feature_loss = np.array(0)
			feature_loss = torch.from_numpy(feature_loss).to(device).float()
			for i in range(dataset_num-1):
				low_dim = Project_DNN(dataset[i], i)
				low_dim_biggest_dataset = Project_DNN(dataset[dataset_num-1][pairs[i]], len(dataset)-1)
				feature_loss += c_mse(low_dim, low_dim_biggest_dataset)
				# min_norm = torch.min(torch.norm(low_dim), torch.norm(low_dim_biggest_dataset))
				# feature_loss += torch.abs(torch.norm(low_dim) - torch.norm(low_dim_biggest_dataset))/min_norm

			loss = params.beta * feature_loss
			for i in range(dataset_num):
				loss += KL_loss[i]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if (epoch+1) % params.log_DNN == 0:
			print("epoch:[{:d}/{}]: loss:{:4f}, align_loss:{:4f}".format(epoch+1, \
				params.epoch_DNN, loss.data.item(), feature_loss.data.item()))

	integrated_data = []
	for i in range(dataset_num):
		integrated_data.append(Project_DNN(dataset[i], i))
		integrated_data[i] = integrated_data[i].detach().cpu().numpy()
	print("Done")
	return integrated_data

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

def project_barycentric(dataset, match):
	print("---------------------------------")
	print("Begin finding the embedded space")
	integrated_data = []

	for i in range(len(dataset)-1):
		integrated_data.append(np.matmul(match[i], dataset[-1]))
	integrated_data.append(dataset[-1])
	print("Done")
	return integrated_data




