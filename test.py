import os
import random
import numpy as np
import scipy.sparse as sp 
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def align_fraction(data1, data2):
	row1, col1 = np.shape(data1)
	row2, col2 = np.shape(data2)
	fraction = 0
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

def transfer_accuracy(domain1, domain2, type1, type2):
	knn = KNeighborsClassifier()
	knn.fit(domain2, type2)
	type1_predict = knn.predict(domain1)
	np.savetxt("type1_predict.txt", type1_predict)
	count = 0
	for label1, label2 in zip(type1_predict, type1):
		if label1 == label2:
			count += 1
	return count / len(type1)

def test_UnionCom(integrated_data, datatype):

	for i in range(len(integrated_data)-1):
		# fraction = align_fraction(data[i], data[-1])
		# print("average fraction:")
		# print(fraction)

		acc = transfer_accuracy(integrated_data[i], integrated_data[-1], datatype[i], datatype[-1])
		print("label transfer accuracy of data{:d}:".format(i+1))
		print(acc)
