import numpy as np

data2 = np.loadtxt("data2.txt")
data2_0 = np.loadtxt("data2_0.txt")
count = 0
permutation = np.zeros(300)
for i in range(300):
	for j in range(300):
		judge = (data2_0[i]==data2[j])
		if judge[0] == True and judge[1] == True:
			permutation[i]=j
			count += 1

print(permutation)
np.savetxt("permutation.txt", permutation)