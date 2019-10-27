import os
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp 
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

##### use CPU
def cor_pairs_CPU(Kx, Kz, N, params):

	Kx = Kx / N
	Kz = Kz / N
	m = np.shape(Kx)[0]
	n = np.shape(Kz)[0]
	F = np.zeros((m,n))
	Im = np.ones((m,1))
	In = np.ones((n,1))
	Lambda = np.zeros((n,1))
	Mu = np.zeros((m,1))
	S = np.zeros((n,1))
	a = np.sqrt(params.p2/params.p1)
	for i in range(params.epoch_pd):
		grad = 2*np.dot(F, np.dot(Kz, np.dot(F.T, np.dot(F, Kz.T)))) \
		+ 2*np.dot(F, np.dot(Kz.T, np.dot(F.T, np.dot(F, Kz)))) \
		- 2*a*np.dot(np.dot(Kx.T, F), Kz) - 2*a*np.dot(np.dot(Kx, F), Kz.T) + np.dot(Mu, In.T) \
		+ np.dot(Im, Lambda.T) + params.rho*(np.dot(np.dot(F, In), In.T) - np.dot(Im, In.T) \
		+ np.dot(np.dot(Im, Im.T), F) + np.dot(Im, (S-In).T))
		F_tmp = F - grad
		# for j in range(m):
		# 	for k in range(n):
		# 		if F_tmp[j,k]<0:
		# 			F_tmp[j,k]=0
		F_tmp[F_tmp<0]=0
		F = (1-params.epsilon)*F + params.epsilon*F_tmp

		grad_s = Lambda + params.rho*(np.dot(F.T, Im) - In + S)
		s_tmp = S - grad_s
		# for j in  range(n):
		# 	if s_tmp[j,0]<0:
		# 		s_tmp[j,0]=0
		s_tmp[s_tmp<0]=0
		S = (1-params.epsilon)*S + params.epsilon*s_tmp

		Mu = Mu + params.epsilon*(np.dot(F,In) - Im)
		Lambda = Lambda + params.epsilon*(np.dot(F.T, Im) - In + S)

		#### if scaling factor a changes too fast, we can delay the update of speed.
		# if i>=10000:
		# 	grad_a = 2*a*np.dot(Kx.T, Kx) - np.dot(Kx.T, np.dot(F, np.dot(Kz, F.T))) - np.dot(F, np.dot(Kz.T, np.dot(F.T, Kx)))
		# 	a = a - params.epsilon*grad_a
		# 	a = np.mean(a)

		a = torch.trace(torch.mm(torch.t(Kx), torch.mm(torch.mm(F, Kz), torch.t(F)))) / \
			torch.trace(torch.mm(torch.t(Kx), Kx))
		# if a > 2*np.sqrt(params.p2/params.p1):
		# 	a = 2*np.sqrt(params.p2/params.p1)
		# if a < 0.5*np.sqrt(params.p2/params.p1):
		# 	a = 0.5*np.sqrt(params.p2/params.p1)
		    
		print("[{:d}/{:d}]".format(i,params.epoch_pd), np.linalg.norm(a*Kx - np.dot(np.dot(F, Kz), F.T)), a)
	pairs = np.zeros(m)
	for i in range(m):
		pairs[i] = np.argsort(F[i])[-1]
	return F, pairs

##### use GPU
def cor_pairs_GPU(Kx, Kz, N, params, p1, p2, device):
	print(device)
	Kx = Kx / N 
	Kz = Kz / N
	Kx = torch.from_numpy(Kx).float().to(device)
	Kz = torch.from_numpy(Kz).float().to(device)
	m = np.shape(Kx)[0]
	n = np.shape(Kz)[0]
	F = np.zeros((m,n))
	F = torch.from_numpy(F).float().to(device)
	Im = np.ones((m,1))
	Im = torch.from_numpy(Im).float().to(device)
	In = np.ones((n,1))
	In = torch.from_numpy(In).float().to(device)
	Lambda = np.zeros((n,1))
	Lambda = torch.from_numpy(Lambda).float().to(device)
	Mu = np.zeros((m,1))
	Mu = torch.from_numpy(Mu).float().to(device)
	S = np.zeros((n,1))
	S = torch.from_numpy(S).float().to(device)
	a = np.sqrt(p2/p1)
	# a = torch.from_numpy(a).float().to(device)

	# print("1111111111111111111")
	for i in range(params.epoch_pd):
		grad = 2*torch.mm(F, torch.mm(Kz, torch.mm(torch.t(F), torch.mm(F, torch.t(Kz))))) \
		+ 2*torch.mm(F, torch.mm(torch.t(Kz), torch.mm(torch.t(F), torch.mm(F, Kz)))) \
		- 2*a*torch.mm(torch.t(Kx), torch.mm(F, Kz)) - 2*a*torch.mm(Kx, torch.mm(F,torch.t(Kz))) + torch.mm(Mu, torch.t(In)) \
		+ torch.mm(Im, torch.t(Lambda)) + params.rho*(torch.mm(F, torch.mm(In, torch.t(In))) - torch.mm(Im, torch.t(In)) \
		+ torch.mm(Im, torch.mm(torch.t(Im), F)) + torch.mm(Im, torch.t(S-In)))
		F_tmp = F - grad
		F_tmp[F_tmp<0]=0
		F = (1-params.epsilon)*F + params.epsilon*F_tmp
		# F = F - params.epsilon*grad
		# F[F<0]=0

		grad_s = Lambda + params.rho*(torch.mm(torch.t(F), Im) - In + S)
		s_tmp = S - grad_s
		s_tmp[s_tmp<0]=0
		S = (1-params.epsilon)*S + params.epsilon*s_tmp
		# S = S - params.epsilon*grad_s
		# S[S<0]=0

		Mu = Mu + params.epsilon*(torch.mm(F,In) - Im)
		Lambda = Lambda + params.epsilon*(torch.mm(torch.t(F), Im) - In + S)

		#### if scaling factor a changes too fast, we can delay the update of speed.
		if i>=params.delay:
			grad_a = 2*a*torch.mm(torch.t(Kx), Kx) - torch.mm(torch.t(Kx), torch.mm(F, torch.mm(Kz, torch.t(F)))) - \
			torch.mm(F, torch.mm(torch.t(Kz), torch.mm(torch.t(F), Kx)))
			a = a - params.epsilon_a*grad_a
			a = torch.mean(a).to(device)
		# if i>=200000:
		# 	a = torch.trace(torch.mm(torch.t(Kx), torch.mm(torch.mm(F, Kz), torch.t(F)))) / \
		# 		torch.trace(torch.mm(torch.t(Kx), Kx))

		norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Kz), torch.t(F)))
		# if i<20000:
		if (i+1) % 100 == 0:
			print("[{:d}/{:d}]".format(i,params.epoch_pd), norm2.data.item(), a)
		# else:
		# 	print("[{:d}/{:d}]".format(i,params.epoch_pd), norm2.data.item(), a.data.item())

	F = F.cpu().numpy()
	pairs = np.zeros(m)
	for i in range(m):
		pairs[i] = np.argsort(F[i])[-1]
	return F, pairs


# def cor_pairs(Kx, Kz, N):

# 	Kx = Kx / N
# 	Kz = Kz / N
# 	m = np.shape(Kx)[0]
# 	n = np.shape(Kz)[0]
# 	rate = 0.0005
# 	epoch = 10000
# 	F = np.zeros((m,n))
# 	Im = np.ones((m,1))
# 	In = np.ones((n,1))
# 	Lambda = np.zeros((n,1))
# 	Mu = np.zeros((m,1))
# 	S = np.zeros((n,1))
# 	p = 10
# 	for i in range(epoch):
# 		grad = 2*np.dot(F, np.dot(Kz, np.dot(np.dot(F.T, F), Kz.T))) \
# 		+ 2*np.dot(F, np.dot(Kz.T, np.dot(np.dot(F.T, F), Kz))) \
# 		- 2*np.dot(np.dot(Kx.T, F), Kz) - 2*np.dot(np.dot(Kx, F), Kz.T) + np.dot(Mu, In.T) \
# 		+ np.dot(Im, Lambda.T) + p*(np.dot(np.dot(F, In), In.T) - np.dot(Im, In.T) \
# 		+ np.dot(np.dot(Im, Im.T), F) + np.dot(Im, (S-In).T))
# 		F_tmp = F - grad
# 		for j in range(m):
# 			for k in range(n):
# 				if F_tmp[j,k]<0:
# 					F_tmp[j,k]=0
# 		F = (1-rate)*F + rate*F_tmp

# 		grad_s = Lambda + p*(np.dot(F.T, Im) - In + S)
# 		s_tmp = S - grad_s
# 		for j in  range(n):
# 			if s_tmp[j,0]<0:
# 				s_tmp[j,0]=0
# 		S = (1-rate)*S + rate*s_tmp

# 		Mu = Mu + rate*(np.dot(F,In) - Im)
# 		Lambda = Lambda + rate*(np.dot(F.T, Im) - In + S)
# 		print("[{:d}/{:d}]".format(i,epoch), np.linalg.norm(Kx - np.dot(np.dot(F, Kz), F.T)))
# 	pairs = np.zeros(m)
# 	for i in range(m):
# 		pairs[i] = np.argsort(F[i])[-1]
# 	print(S)
# 	return F, pairs
