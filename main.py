import os
import sys
import numpy as np
import torch 
import torch.nn as nn
from simulate_data import *
from model import Project
from train import train
from utils import *
from test import *
from PrimeDual import *
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch_DNN',    type=int,     default='600',      help="epoches of DNN")
parser.add_argument('--epoch_pd',     type=int,     default='60000',   help="epoches of prime-dual algorithm")
parser.add_argument('--epsilon',      type=float,   default='0.0003',   help="learning rate of F")
parser.add_argument('--epsilon_a',    type=float,   default='0.0003',   help="learning rate of alpha")
parser.add_argument('--lr',           type=float,   default='0.0005',   help="learning rate of DNN")
parser.add_argument('--batch_size',   type=int,     default='50',       help="batch size of DNN")
parser.add_argument('--beta1',        type=float,   default='1',      help="beta_1 in loss function of DNN")
parser.add_argument('--beta2',        type=float,   default='1',        help="beta_2 in loss function of DNN")
parser.add_argument('--rho',          type=float,   default='10',       help="parameter of augumented larangian function")
parser.add_argument('--log_step',     type=int,     default='1',        help="log step of DNN")
parser.add_argument('--manual_seed',  type=int,     default='8888',     help="random seed")
parser.add_argument('--simu',         type=int,     default='1',        help="which simulation you want, you can choose {1,2,3}")
parser.add_argument('--isRealData',   type=int,     default='0',        help="choose simulated data or real data")
parser.add_argument('--delay',        type=int,     default='10000', help="delay the update of alpha for numerical stability")
params = parser.parse_args()

init_random_seed(params.manual_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if params.isRealData == 0:
	######simu data
	if params.simu == 1:
		####simu1
		n0 = 100
		n1 = 100
		n2 = 50
		N = n0+n1+n2
		type1 = np.loadtxt("./simu1/type1.txt") 
		type2 = np.loadtxt("./simu1/type2.txt")
		domain1 = np.loadtxt("./simu1/domain1.txt")
		domain2 = np.loadtxt("./simu1/domain2.txt")
		domain2_0 = np.loadtxt("./simu1/domain2_0.txt")
		# data1, data2, data2_0, permutation, type1, type2 = two_branch_points(n0=n0, n1=n1, n2=n2)
		# domain1 = project_high_dim_dropout(data1, 2, 1000, fraction=0)
		# domain2, domain2_0 = project_high_dim_dropout(data2, 2, 500, data2_0, flag=1, fraction=0)
	elif params.simu == 2:
		#####simu2
		n0 = 100
		n1 = 50
		n2 = 50
		N = n0+n1+n2
		type1 = np.loadtxt("./simu2/type1.txt") 
		type2 = np.loadtxt("./simu2/type2.txt")
		domain1 = np.loadtxt("./simu2/domain1.txt")
		domain2 = np.loadtxt("./simu2/domain2.txt")
		domain2_0 = np.loadtxt("./simu2/domain2_0.txt")
		# data1, data2, data2_0, permutation, type1, type2 = one_branch_point2(n0=n0, n1=n1, n2=n2)
		# domain1 = project_high_dim_dropout(data1, 2, 1000, fraction=0)
		# domain2, domain2_0 = project_high_dim_dropout(data2, 3, 500, data2_0, flag=1, fraction=0)
	elif params.simu == 3:
		#####simu3
		n1 = 250
		n2 = 100
		n3 = 100
		n4 = 160
		n5 = 60
		n6 = 60
		N = n1+n2+n3+n4+n5+n6
		type1 = np.loadtxt("./simu3/type1.txt") 
		type2 = np.loadtxt("./simu3/type2.txt")
		domain1 = np.loadtxt("./simu3/domain1.txt")
		domain2 = np.loadtxt("./simu3/domain2.txt")
		domain2_0 = np.loadtxt("./simu3/domain2_0.txt")
		# data1, data2, data2_0, permutation, type1, type2 = cell_circle(n1,n2,n3,n4,n5,n6)
		# domain1 = project_high_dim_dropout(data1, 2, 1000, fraction=0)
		# domain2, domain2_0 = project_high_dim_dropout(data2, 2, 500, data2_0, flag=1, fraction=0)
	else:
		print("error: input correct simu number!")

	# np.savetxt("permutation.txt", permutation)
	# np.savetxt("data1.txt", data1)
	# np.savetxt("data2.txt", data2)
	# np.savetxt("data2_0.txt", data2_0)
	# np.savetxt("domain1.txt", domain1)
	# np.savetxt("domain2.txt", domain2)
	# np.savetxt("domain2_0.txt", domain2_0)
	# np.savetxt("type1.txt", type1)
	# np.savetxt("type2.txt", type2)
	
else:
	#######real data
	domain1 = np.loadtxt("./real_data/real_domain1.txt")
	# domain1 = np.transpose(domain1)
	domain2_0 = np.loadtxt("./real_data/real_domain2.txt")
	# domain2_0 = np.transpose(domain2_0)
	type1 = np.loadtxt("./real_data/type1.txt")
	type2 = np.loadtxt("./real_data/type2.txt")
	domain2 = domain2_0

type1 = type1.astype(np.int)
type2 = type2.astype(np.int)
row1, col1 = np.shape(domain1)
row2, col2 = np.shape(domain2_0)
N = np.maximum(np.shape(domain1)[0], np.shape(domain2_0)[0])
print(np.shape(domain1))
print(np.shape(domain2))
if row1 > row2:
	print("error! N_x must less or equall to N_y")

# #z-score normalization
domain1 = (domain1 - np.mean(domain1)) / np.std(domain1)
domain2 = (domain2 - np.mean(domain2)) / np.std(domain2)
domain2_0 = (domain2_0 - np.mean(domain2_0))/ np.std(domain2_0)

geo_dis1, kmin1 = geodesic_distances(domain1)
geo_dis2, kmin2 = geodesic_distances(domain2)
# geo_dis1, kmin1 = euclidean_distances(domain1)
# geo_dis2, kmin2 = euclidean_distances(domain2)
print(kmin1, kmin2)
np.savetxt("./result/geo_dis1.txt", geo_dis1)
np.savetxt("./result/geo_dis2.txt", geo_dis2)

P_joint1 = p_joint(domain1, kmin1)
P_joint2 = p_joint(domain2, kmin2)

# F, cor_pairs = cor_pairs_CPU(geo_dis1, geo_dis2, N, params, col1, col2,)
F, cor_pairs = cor_pairs_GPU(geo_dis1, geo_dis2, N, params, col1, col2, device)
cor_pairs = cor_pairs.astype(np.int)

np.savetxt("./result/correspondence_F.txt", F)
np.savetxt("./result/cor_pairs.txt", cor_pairs)

# cor_pairs = np.loadtxt("./result/cor_pairs.txt")
# cor_pairs = cor_pairs.astype(np.int)
print(cor_pairs)

net = Project()
Project = init_model(net, device, restore = None)

print("Training Project and Discriminator")

Project = train(Project, params, domain1, domain2, P_joint1, P_joint2, cor_pairs, device)

########## test
domain1_test = torch.from_numpy(domain1)
domain2_test = torch.from_numpy(domain2_0)
domain1_test = domain1_test.float().to(device)
domain2_test = domain2_test.float().to(device)
data_1 = Project(domain1_test, 1)
data_1 = data_1.detach().cpu().numpy()
data_2 = Project(domain2_test, 2)
data_2 = data_2.detach().cpu().numpy()

row1, col1 = np.shape(domain1_test)
row2, col2 = np.shape(domain2_test)

np.savetxt('./result/new_data1.txt',data_1)
np.savetxt('./result/new_data2.txt',data_2)

# data_1 = np.loadtxt("./result/new_data1.txt")
# data_2 = np.loadtxt("./result/new_data2.txt")

if params.isRealData == 0:
	fraction = align_fraction(data_1, data_2, params)
	print("average fraction:")
	print(fraction)

acc = transfer_accuracy(data_1, data_2, type1, type2)
print("label transfer accuracy:")
print(acc)
