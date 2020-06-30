import numpy as np 
from PrimeDual import cor_pairs_match
from utils import save_model

def match(params, dataset, dist, device):

	dataset_num = len(dataset)

	N = np.int(np.max([len(l) for l in dataset]))

	cor_pairs = []

	for i in range(dataset_num-1):
		print("---------------------------------")
		print("Find matching matrix between Dataset {} and Dataset {}".format(i+1, \
			len(dataset)))

		cor_pairs.append(cor_pairs_match(dist[i], dist[-1], N, \
			params, params.col[i], params.col[-1], device))

	print("Finished Matching!")

	return cor_pairs

























