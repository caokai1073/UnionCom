import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import matplotlib
from matplotlib import pyplot as plt


from utils import save_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def train(Project, params, domain1, domain2, P_joint1, P_joint2, cor_pairs, device):

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	optimizer = optim.RMSprop(Project.parameters(), lr=params.lr)
	c_mse = nn.MSELoss()
	c_domain = nn.NLLLoss()
	cirterion_CE = nn.CrossEntropyLoss()

	Project.train()
	num1, _ = np.shape(domain1)
	num2, _ = np.shape(domain2)
	P_joint1 = torch.from_numpy(P_joint1).float().to(device)
	P_joint2 = torch.from_numpy(P_joint2).float().to(device)
	
	for epoch in range(params.epoch_DNN):
		len_dataloader = np.int(num1/params.batch_size)
		for step in range(len_dataloader):  

			random_choice1 = np.random.randint(0, num1, params.batch_size)
			random_choice2 = np.random.randint(0, num2, params.batch_size)
			
			data_1 = domain1[random_choice1]
			data_2 = domain2[random_choice2]
			data_2_align = domain2[cor_pairs[random_choice1]]
			data_1 = torch.from_numpy(data_1).to(device).float()
			data_2 = torch.from_numpy(data_2).to(device).float()
			data_2_align = torch.from_numpy(data_2_align).to(device).float()

			P_tmp1 = torch.zeros([params.batch_size, params.batch_size]).to(device)
			P_tmp2 = torch.zeros([params.batch_size, params.batch_size]).to(device)
			for i in range(params.batch_size):
				P_tmp1[i] = P_joint1[random_choice1[i], random_choice1]
				P_tmp2[i] = P_joint2[random_choice2[i], random_choice2]
			P_tmp1 = P_tmp1 / torch.sum(P_tmp1)
			P_tmp2 = P_tmp2 / torch.sum(P_tmp2)

			low_dim_data1 = Project(data_1, 1)
			low_dim_data2 = Project(data_2, 2)
			low_dim_data2_align = Project(data_2_align, 2)
			feature_loss = c_mse(low_dim_data1, low_dim_data2_align)

			Q_joint1 = Q_tsne(low_dim_data1)
			Q_joint2 = Q_tsne(low_dim_data2)

			KL_loss1 = torch.sum(P_tmp1 * torch.log(P_tmp1 / Q_joint1))
			KL_loss2 = torch.sum(P_tmp2 * torch.log(P_tmp2 / Q_joint2))
			KL_loss = KL_loss1 + params.beta1 * KL_loss2
			
			loss = KL_loss + params.beta2 * feature_loss


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (step+1) % params.log_step == 0:
				print("[{:4d}/{}] [{:2d}/{}]: KL_loss={:4f}, feature_loss={:4f}".format(epoch+1, \
					params.epoch_DNN, step+1, len_dataloader, KL_loss.data.item(), feature_loss.data.item()))
	
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

def Energy_function(X, W_pos, W_neg, Lambda):
	distances = neg_square_dists(X)
	attractive = torch.sum(-W_pos*distances)
	repulsive =  torch.sum(W_neg*torch.exp(distances))
	# print(attractive, repulsive)
	return attractive + Lambda*repulsive
























