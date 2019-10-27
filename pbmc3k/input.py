import numpy as np
import pandas as pd
from pandas import DataFrame
import random

label = pd.read_csv("pbmc_celltype.txt")
label = label.values
# print(label)
# print(np.shape(label))
domain = np.loadtxt("pbmc.txt")
domain = np.transpose(domain)
# print(np.shape(domain))
domain_1234 = []
type_1234 = []

for i in range(2638):
	if label[i]=="Naive CD4 T":
		domain_1234.append(domain[i])
		type_1234.append(0)
	if label[i]=="CD8 T":
		domain_1234.append(domain[i])
		type_1234.append(1)
	if label[i]=="CD14+ Mono":
		domain_1234.append(domain[i])
		type_1234.append(2)
	if label[i]=="B":
		domain_1234.append(domain[i])
		type_1234.append(3)
	if label[i]=="Memory CD4 T":
		domain_1234.append(domain[i])
		type_1234.append(4)

domain1 = []
domain2 = []
type1 =[]
type2 =[]
row = np.shape(domain_1234)[0]
p = 0.35
for i in range(row):
	if random.random() <= p:
		if type_1234[i] is not 1 and type_1234[i] is not 4:
			domain1.append(domain_1234[i])
			type1.append(type_1234[i])
	else:
		domain2.append(domain_1234[i])
		type2.append(type_1234[i])

# domain1 = domain[0:879]
# domain2 = domain[879:2638]
# label1 = label[0:879]
# label2 = label[879:2638]
np.savetxt("real_domain1.txt", domain1)
np.savetxt("real_domain2.txt", domain2)
# row1 = len(label1)
# type1 = np.zeros(row1)
# row2 = len(label2)
# type2 = np.zeros(row2)
# count = np.zeros(9)
# for i in range(row1):
# 	# print(label1[i])
# 	if label1[i]=="Naive CD4 T":
# 		count[0] += 1
# 		type1[i]=0
# 	if label1[i]=="Memory CD4 T":
# 		count[1] += 1
# 		type1[i]=1
# 	if label1[i]=="CD14+ Mono":
# 		count[2] += 1
# 		type1[i]=2
# 	if label1[i]=="B":
# 		count[3] += 1
# 		type1[i]=3
# 	if label1[i]=="CD8 T":
# 		count[4] += 1
# 		type1[i]=4
# 	if label1[i]=="FCGR3A+ Mono":
# 		count[5] += 1
# 		type1[i]=5
# 	if label1[i]=="NK":
# 		count[6] += 1
# 		type1[i]=6
# 	if label1[i]=="DC":
# 		count[7] += 1
# 		type1[i]=7
# 	if label1[i]=="Platelet":
# 		count[8] += 1
# 		type1[i]=8

# type1 = type1.astype(np.int)
np.savetxt("type1.txt",type1)

# for i in range(row2):
# 	if label2[i]=="Naive CD4 T":
# 		count[0] += 1
# 		type2[i]=0
# 	if label2[i]=="Memory CD4 T":
# 		count[1] += 1
# 		type2[i]=1
# 	if label2[i]=="CD14+ Mono":
# 		count[2] += 1
# 		type2[i]=2
# 	if label2[i]=="B":
# 		count[3] += 1
# 		type2[i]=3
# 	if label2[i]=="CD8 T":
# 		count[4] += 1
# 		type2[i]=4
# 	if label2[i]=="FCGR3A+ Mono":
# 		count[5] += 1
# 		type2[i]=5
# 	if label2[i]=="NK":
# 		count[6] += 1
# 		type2[i]=6
# 	if label2[i]=="DC":
# 		count[7] += 1
# 		type2[i]=7
# 	if label2[i]=="Platelet":
# 		count[8] += 1
# 		type2[i]=8
# print(count)
# type2 = type2.astype(np.int)
np.savetxt("type2.txt",type2)
# print(type1)
# print(domain1)