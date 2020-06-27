import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.sparse as sp 
from itertools import chain
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def init_random_seed(manual_seed):
	seed = None
	if manual_seed is None:
		seed = random.randint(1,10000)
	else:
		seed = manual_seed
	print("use random seed: {}".format(seed))
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def init_model(net, device, restore):
	if restore is not None and os.path.exits(restore):
		net.load_state_dict(torch.load(restore))
		net.restored = True
		print("Restore model from: {}".format(os.path.abspath(restore)))
	else:
		print("No trained model, train UnionCom from scratch.")

	if torch.cuda.is_available():
		cudnn.benchmark =True
		net.to(device)

	return net

def save_model(net, model_root, filename):

	if not os.path.exists(model_root):
		os.makedirs(model_root)
	torch.save(net.state_dict(), os.path.join(model_root, filename))
	print("save pretrained model to: {}".format(os.path.join(model_root, filename)))

#-||x_i-x_j||^2
# def neg_squared_euc_dists(X):
#     sum_X = np.sum(np.square(X), 1)
#     D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

#     return -D

#input -||x_i-x_j||^2/2*sigma^2, compute softmax
def softmax(D, diag_zero=True):
	# e_x = np.exp(D)
	e_x = np.exp(D - np.max(D, axis=1).reshape([-1, 1]))
	if diag_zero:
		np.fill_diagonal(e_x, 0)
	e_x = e_x + 1e-15
	return e_x / e_x.sum(axis=1).reshape([-1,1])


#input -||x_i-x_j||^2, compute P_ji = exp(-||x_i-x_j||^2/2*sigma^2)/sum(exp(-||x_i-x_j||^2/2*sigma^2))
def calc_P(distances, sigmas=None):
	if sigmas is not None:
		two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
		return softmax(distances / two_sig_sq)
	else:
		return softmax(distances)

#a binary search algorithm for target
def binary_search(eval_fn, target ,tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
	for i in range(max_iter):
		guess = (lower + upper) /2.
		val = eval_fn(guess)
		if val > target:
			upper = guess
		else:
			lower = guess
		if np.abs(val - target) <= tol:
			break
	return guess

#input matrix P, compute perp(P_i)=2^H(P_i), where H(P_i)=-sum(p_ij * log2 P_ij)
def calc_perplexity(prob_matrix):
	entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
	perplexity = 2 ** entropy
	return perplexity

#input -||x_i-x_j||^2 and sigma, out put perplexity
def perplexity(distances, sigmas):
	return calc_perplexity(calc_P(distances, sigmas))

def find_optimal_sigmas(distances, target_perplexity):
	sigmas = []
	for i in range(distances.shape[0]):
		eval_fn = lambda sigma: perplexity(distances[i:i+1, :], np.array(sigma))
		correct_sigma = binary_search(eval_fn, target_perplexity)
		sigmas.append(correct_sigma)
	return np.array(sigmas)

def p_conditional_to_joint(P):
	return (P + P.T) / (2. * P.shape[0])

def p_joint(X, target_perplexity):
    # distances = neg_squared_euc_dists(X)
    distances = -X
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    p_conditional = calc_P(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)
    return P

def q_tsne(Y):	
	distances = neg_squared_euc_dists(Y)
	inv_distances = np.power(1. - distances, -1)
	np.fill_diagonal(inv_distances, 0)	
	return inv_distances / np.sum(inv_distances)

def geodesic_distances(X, kmax):
	kmin = 5
	nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
	knn = nbrs.kneighbors_graph(X, mode='distance')
	connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
	# not_connected False
	# index = 0
	while connected_components is not 1:
		if kmin > np.max((kmax, 0.01*len(X))):
			# not_connected = True
			break
		kmin += 2
		nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
		knn = nbrs.kneighbors_graph(X, mode='distance')
		connected_components = sp.csgraph.connected_components(knn, directed=False)[0]

	dist = sp.csgraph.floyd_warshall(knn, directed=False)

	dist_max = np.nanmax(dist[dist != np.inf])
	dist[dist > dist_max] = 2*dist_max

	# connected_element = []
	# if not_connected:
	# 	inf_matrix = []
		
	# 	for i in range(len(X)):
	# 		inf_matrix.append(list(chain.from_iterable(np.argwhere(np.isinf(dist[i])))))

		
	# 	for i in range(len(X)):
	# 		if i==0:
	# 			connected_element.append([])
	# 			connected_element[0].append(i)
	# 		else:
	# 			for j in range(len(connected_element)+1):
	# 				if j == len(connected_element):
	# 					connected_element.append([])
	# 					connected_element[j].append(i)
	# 					break
	# 				if inf_matrix[i] == inf_matrix[connected_element[j][0]]:
	# 					connected_element[j].append(i)
	# 					break
	# 	for i in range(len(connected_element)):
	# 		if i==0:
	# 			mx = len(connected_element[0])
	# 			index = 0
	# 		if len(connected_element[i])>mx:
	# 			mx = len(connected_element[0])
	# 			index = i

	# 	X = X[connected_element[index]]
	# 	kmin = 5
	# 	nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
	# 	knn = nbrs.kneighbors_graph(X, mode='distance')
	# 	connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
	# 	while connected_components is not 1:
	# 		kmin += 2
	# 		nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
	# 		knn = nbrs.kneighbors_graph(X, mode='distance')
	# 		connected_components = sp.csgraph.connected_components(knn, directed=False)[0]

	# 	dist = sp.csgraph.floyd_warshall(knn, directed=False)


	return dist, kmin

def euclidean_distances(data):
	row, col = np.shape(data)
	dist = np.zeros((row, row))
	for i in range(row):
		diffMat = np.tile(data[i], (row,1)) - data
		sqDiffMat = diffMat**2
		sqDistances = sqDiffMat.sum(axis=1)
		dist[i]=sqDistances
	return dist, 5








	