import numpy as np
import pandas as pd
from pandas import DataFrame
import random

label1 = pd.read_csv("young_type.txt")
label1 = label1.values
label2 = pd.read_csv("old_type.txt")
label2 = label2.values
row1 = np.shape(label1)[0]
row2 = np.shape(label2)[0]

type1 =[]
type2 =[]
for i in range(row1):
	if label1[i]=="LT":
		type1.append(0)
	if label1[i]=="ST":
		type1.append(1)
	if label1[i]=="MPP":
		type1.append(2)

for i in range(row2):
	if label2[i]=="LT":
		type2.append(0)
	if label2[i]=="ST":
		type2.append(1)
	if label2[i]=="MPP":
		type2.append(2)

# np.savetxt("real_domain1.txt", domain1)
# np.savetxt("real_domain2.txt", domain2)

np.savetxt("type1.txt",type1)

np.savetxt("type2.txt",type2)
