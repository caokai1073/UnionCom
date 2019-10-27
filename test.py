import os
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp 
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def align_fraction(data1, data2, params):
	row1, col1 = np.shape(data1)
	row2, col2 = np.shape(data2)
	fraction = 0
	if params.isRealData == 0 and params.simu == 3:
		for i in range(row1):
			count = 0
			diffMat = np.tile(data1[i], (row2,1)) - data2
			sqDiffMat = diffMat**2
			sqDistances = sqDiffMat.sum(axis=1)
			for j in range(row2):
				if i<=125:
					if sqDistances[j] < sqDistances[i]:
						count += 1
				elif i>125 and i<=175:
					if sqDistances[j] < sqDistances[i+125]:
						count += 1
				elif i>175 and i<=225:
					if sqDistances[j] < sqDistances[i+175]:
						count += 1
				elif i>225 and i<=305:
					if sqDistances[j] < sqDistances[i+225]:
						count += 1
				elif i>305 and i<=335:
					if sqDistances[j] < sqDistances[i+305]:
						count += 1
				else:
					if sqDistances[j] < sqDistances[i+335]:
						count += 1
			fraction += count / row2
	else:	
		for i in range(row1):
			count = 0
			diffMat = np.tile(data1[i], (row2,1)) - data2
			sqDiffMat = diffMat**2
			sqDistances = sqDiffMat.sum(axis=1)
			for j in range(row2):
				if sqDistances[j] < sqDistances[i]:
					count += 1
			fraction += count / row2

	return fraction / row1


# def label_transfer_accuracy(row1, row2, type1, type2, cor_pairs):
# 	acc = 0.
# 	type1_predict = np.zeros(row1)
# 	for i in range(row1):
# 		type1_predict[i] = type2[cor_pairs[i]]
# 	for i in range(row1):
# 		if type1[i] == type1_predict[i]:
# 			acc += 1
# 	acc /= row1

# 	return acc


def transfer_accuracy(domain1, domain2, type1, type2):
	knn = KNeighborsClassifier()
	knn.fit(domain2, type2)
	type1_predict = knn.predict(domain1)
	np.savetxt("./result/type1_predict.txt", type1_predict)
	count = 0
	for label1, label2 in zip(type1_predict, type1):
		if label1 == label2:
			count += 1
	return count / len(type1)


