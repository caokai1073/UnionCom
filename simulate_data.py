import torch
import numpy as np
import math
import random
import matplotlib
from matplotlib import pyplot as plt
import scipy.sparse as sp 
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

def one_branch_point2(length=3, branch1=1.5, branch2=0.8, n0=100, n1=50, n2=50, noise=0.05):

    x0 = np.random.uniform(-length, 0, n0)
    x0 = np.sort(x0)
    y0 = np.random.normal(scale=noise, size=n0)

    x1 = np.random.uniform(0, branch1, n1)
    x1 = np.sort(x1)
    y1 = np.random.normal(scale=1.5*noise, size=n1)

    x2 = np.random.uniform(0, branch2, n2)
    x2 = np.sort(x2)
    y2 = np.random.normal(scale=1.2*noise, size=n2)

    x3 = np.random.uniform(0, 1.3, 50)
    x3 = np.sort(x3)
    y3 = np.random.normal(scale=noise, size=50)

    theta = math.pi / 4

    x12 = x1 * math.cos(theta) + y1 * math.sin(theta)
    y12 = y1 * math.cos(theta) - x1 * math.sin(theta)

    x22 = x2 * math.cos(-theta) + y2 * math.sin(-theta)
    y22 = y2 * math.cos(-theta) - x2 * math.sin(-theta)

    x32 = x3 * math.cos(-3*theta) + y3 * math.sin(-3*theta)
    y32 = y3 * math.cos(-3*theta) - x3 * math.sin(-3*theta)

    x = np.append(np.append(x0, x12), x22)
    y = np.append(np.append(y0, y12), y22)
    xx = np.append(np.append(np.append(x0, x12), x22), x32)
    yy = np.append(np.append(np.append(y0, y12), y22), y32)
    data1 = np.array([x, y])
    data1 = np.transpose(data1)
    data1 = data1 + np.random.normal(0,0.03,np.shape(data1))

    beta = math.pi / 2
    x3 = xx * math.cos(beta) - yy * math.sin(beta)
    y3 = xx * math.sin(beta) + yy * math.cos(beta) 
    data2 = np.array([x3, np.cos(y3), np.sin(y3)])
    data2 = np.transpose(data2)
    data2 = data2 + np.random.normal(0,0.03,np.shape(data2))

    data2_0 = data2
    permutation = np.random.permutation(data2.shape[0])
    data2 = data2[permutation,:]

    row1 = np.shape(data1)[0]
    row2 = np.shape(data2)[0]
    type1 = np.zeros(row1)
    type2 = np.zeros(row2)
    type1[0:100]=1
    type1[100:150]=2
    type1[150:200]=3

    type2[0:200]=type1
#     fig = plt.figure()
#     ax = Axes3D(fig)
# #    plt.axis([-32,15,-12,8])
#     plt.plot(data1[0:n0,0],data1[0:n0,1],'b^',alpha=0.5)
#     plt.plot(data1[n0:n0+n1,0],data1[n0:n0+n1,1], 'g^',alpha=0.5)
#     plt.plot(data1[n0+n1:n0+n1+n2,0],data1[n0+n1:n0+n1+n2,1],'r^',alpha=0.5)
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     # plt.axis([-1,1.5,-1,1.5])
#     ax.plot(data2[0:n0,0],data2[0:n0,1],data2[0:n0,2],'bx',alpha=0.9)
#     ax.plot(data2[n0:n0+n1,0],data2[n0:n0+n1,1],data2[n0:n0+n1,2],'gx',alpha=0.9)
#     ax.plot(data2[n0+n1:n0+n1+n2,0],data2[n0+n1:n0+n1+n2,1],data2[n0+n1:n0+n1+n2,2],'rx',alpha=0.9)
#     ax.plot(data2[n0+n1+n2:250,0],data2[n0+n1+n2:250,1],data2[n0+n1+n2:250,2],'kx',alpha=0.9)
#     plt.show()
    # np.savetxt('data1.txt', data1)
    # np.savetxt('data2.txt', data2)
    return data1, data2, data2_0, permutation, type1, type2

def two_branch_points(length=3, branch1=2, branch2=1, n0=100, n1=100, n2=50, noise=0.05):
    x0 = np.random.uniform(-length, 0, n0)
    x0 = np.sort(x0)
    y0 = np.random.normal(scale=noise, size=n0)

    x1 = np.random.uniform(0, branch1, n1)
    x1 = np.sort(x1)
    y1 = np.random.normal(scale=1.5*noise, size=n1)

    x2 = np.random.uniform(0, branch2, n2)
    x2 = np.sort(x2)
    y2 = np.random.normal(scale=1.2*noise, size=n2)

    theta = math.pi / 4
    x12 = x1 * math.cos(theta) + y1 * math.sin(theta)
    y12 = y1 * math.cos(theta) - x1 * math.sin(theta)

    x22 = x2 * math.cos(-theta) + y2 * math.sin(-theta)
    y22 = y2 * math.cos(-theta) - x2 * math.sin(-theta)

    x_hat = length + branch1 * math.cos(theta)/2
    y_hat = -branch1 * math.cos(theta)/2

    x1_2 = (x12[int(n1/2) : n1]- x_hat) * math.cos(2 * theta) - (y12[int(n1/2) : n1] - y_hat) * math.sin(2 * theta) + x_hat
    y1_2 = (y12[int(n1/2) : n1]- y_hat)  * math.cos(2 * theta)+ (x12[int(n1/2) : n1]- x_hat) * math.sin(2 * theta) + y_hat

    x1_2 = (x12[int(n1/2) : n1] - x12[int(n1/2)]) * math.cos(2 * theta) - (y12[int(n1/2) : n1] - y12[int(n1/2)])* math.sin(2*theta) + x12[int(n1/2)]
    y1_2 = (x12[int(n1/2) : n1] - x12[int(n1/2)]) * math.sin(2 * theta) + (y12[int(n1/2) : n1] - y12[int(n1/2)])* math.cos(2*theta) + y12[int(n1/2)]

    x1 = np.append(x12, x1_2)
    y1 = np.append(y12, y1_2)

    x = np.append(np.append(x0, x22), x1)
    y = np.append(np.append(y0, y22), y1)
    data1 = np.array([x, y])
    data1 = np.transpose(data1)
    data1 = data1 + np.random.normal(0,0.03,np.shape(data1))

    data2 = np.array([y, x])
    data2 = np.transpose(data2)
    data2 = data2 + np.random.normal(0,0.03,np.shape(data2))
    data2_0 = data2
    permutation = np.random.permutation(data2.shape[0])
    data2 = data2[permutation,:]

    row1 = np.shape(data1)[0]
    type1 = np.zeros(row1)
    type1[0:100]=1
    type1[100:150]=2
    type1[150:200]=3
    type1[200:250]=4
    type1[250:300]=5
    type2 = type1
    # fig = plt.figure()
    # plt.plot(data1[0:100,0], data1[0:100,1], 'b^',alpha=0.5)
    # plt.plot(data1[100:150,0], data1[100:150,1], 'g^',alpha=0.5)
    # plt.plot(data1[150:200,0], data1[150:200,1], 'r^',alpha=0.5)
    # plt.plot(data1[200:250,0], data1[200:250,1], 'y^',alpha=0.5)
    # plt.plot(data1[250:300,0], data1[250:300,1], 'k^',alpha=0.5)
    # fig = plt.figure()
    # plt.plot(data2[0:100,0], data2[0:100,1], 'bx',alpha=0.9)
    # plt.plot(data2[100:150,0], data2[100:150,1], 'gx',alpha=0.9)
    # plt.plot(data2[150:200,0], data2[150:200,1], 'rx',alpha=0.9)
    # plt.plot(data2[200:250,0], data2[200:250,1], 'yx',alpha=0.9)
    # plt.plot(data2[250:300,0], data2[250:300,1], 'kx',alpha=0.9)
    # plt.show()

    return data1, data2, data2_0, permutation, type1, type2


def cell_circle(n1=250,n2=100,n3=100,n4=160, n5=60, n6=60):
    N1 = n1
    N2 = N1 + n2
    N3 = N2 + n3
    N4 = N3 + n4
    N5 = N4 + n5
    N6 = N5 + n6
    N = n1 + n2 + n3 + n4 + n5 + n6

    N12 = int(N1/2)
    N22 = int(N2/2)
    N32 = int(N3/2)
    N42 = int(N4/2)
    N52 = int(N5/2)
    N62 = int(N6/2)
    print(N1,N2,N3,N4,N5,N6)
    print(N12,N22,N32,N42,N52,N62)

    radius = np.random.uniform(8, 16, N)
    radius.sort()
    theta = np.random.uniform(0, 2*math.pi, N)
    theta.sort()
    # theta2 = np.random.uniform(math.pi/3, 2*math.pi/3, n2)
    # theta3 = np.random.uniform(2*math.pi/3, math.pi, n3)
    # theta4 = np.random.uniform(math.pi, 4*math.pi/3, n4)
    # theta5 = np.random.uniform(4*math.pi/3, 5*math.pi/3, n5)
    # theta6 = np.random.uniform(5*math.pi/3, 2*math.pi, n6)

    x1 = radius[0:N1]*np.sin(theta[0:N1])
    y1 = radius[0:N1]*np.cos(theta[0:N1])
    permutation = np.random.permutation(x1.shape[0])
    x1 = x1[permutation]
    y1 = y1[permutation]
    x2 = radius[N1:N2]*np.sin(theta[N1:N2])
    y2 = radius[N1:N2]*np.cos(theta[N1:N2])
    permutation = np.random.permutation(x2.shape[0])
    x2 = x2[permutation]
    y2 = y2[permutation]
    x3 = radius[N2:N3]*np.sin(theta[N2:N3])
    y3 = radius[N2:N3]*np.cos(theta[N2:N3])
    permutation = np.random.permutation(x3.shape[0])
    x3 = x3[permutation]
    y3 = y3[permutation]
    x4 = radius[N3:N4]*np.sin(theta[N3:N4])
    y4 = radius[N3:N4]*np.cos(theta[N3:N4])
    permutation = np.random.permutation(x4.shape[0])
    x4 = x4[permutation]
    y4 = y4[permutation]
    x5 = radius[N4:N5]*np.sin(theta[N4:N5])
    y5 = radius[N4:N5]*np.cos(theta[N4:N5])
    permutation = np.random.permutation(x5.shape[0])
    x5 = x5[permutation]
    y5 = y5[permutation]
    x6 = radius[N5:N6]*np.sin(theta[N5:N6])
    y6 = radius[N5:N6]*np.cos(theta[N5:N6])
    permutation = np.random.permutation(x6.shape[0])
    x6 = x6[permutation]
    y6 = y6[permutation]
    

    x = np.append(np.append(np.append(np.append(np.append(x1, x2), x3), x4), x5), x6)
    y = np.append(np.append(np.append(np.append(np.append(y1, y2), y3), y4), y5), y6)

    x += np.random.normal(0, 0.1*radius,N)

    data2 = np.array([x, y])
    data2 = np.transpose(data2)
    data2 = data2 + np.random.normal(0,0.03,np.shape(data2))
    data2_0 = data2
    # permutation = np.random.permutation(data2.shape[0])
    # data2 = data2[permutation,:]

    alpha = math.pi / 2
    x_rotation = x * math.cos(alpha) + y * math.sin(alpha)
    y_rotation = y * math.cos(alpha) - x * math.sin(alpha)

    data1 = np.array([x_rotation, y_rotation])
    data1 = np.transpose(data1) 
    # random_choice = random.sample(range(0,N), int(N/2))
    # random_choice.sort()
    # print(random_choice)
    tmp = np.vstack((data1[0:N12], data1[N1:N12+N22]))
    tmp = np.vstack((tmp, data1[N2:N22+N32]))
    tmp = np.vstack((tmp, data1[N3:N32+N42]))
    tmp = np.vstack((tmp, data1[N4:N42+N52]))
    data1 = np.vstack((tmp, data1[N5:N52+N62]))
    data1 = data1 + np.random.normal(0,0.03,np.shape(data1))

    row1 = np.shape(data1)[0]
    row2 = np.shape(data2)[0]
    type1 = np.zeros(row1)
    type2 = np.zeros(row2)
    type1[0:N12]=1
    type1[N12:N22]=2
    type1[N22:N32]=3
    type1[N32:N42]=4
    type1[N42:N52]=5
    type1[N52:N62]=6

    type2[0:N1]=1
    type2[N1:N2]=2
    type2[N2:N3]=3
    type2[N3:N4]=4
    type2[N4:N5]=5
    type2[N5:N6]=6

    fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.axis([-8,8,-8,8])
    plt.plot(data1[0:N12,0],data1[0:N12,1],'bx',alpha=0.9)
    plt.plot(data1[N12:N22,0],data1[N12:N22,1],'gx',alpha=0.9)
    plt.plot(data1[N22:N32,0],data1[N22:N32,1],'rx',alpha=0.9)
    plt.plot(data1[N32:N42,0],data1[N32:N42,1],'yx',alpha=0.9)
    plt.plot(data1[N42:N52,0],data1[N42:N52,1],'kx',alpha=0.9)
    plt.plot(data1[N52:N62,0],data1[N52:N62,1],'mx',alpha=0.9)

    fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.axis([-8,8,-8,8])
    
    plt.plot(data2[0:N1,0],data2[0:N1,1],'b^',alpha=0.5)
    plt.plot(data2[N1:N2,0],data2[N1:N2,1],'g^',alpha=0.5)
    plt.plot(data2[N2:N3,0],data2[N2:N3,1],'r^',alpha=0.5)
    plt.plot(data2[N3:N4,0],data2[N3:N4,1],'y^',alpha=0.5)
    plt.plot(data2[N4:N5,0],data2[N4:N5,1],'k^',alpha=0.5)
    plt.plot(data2[N5:N6,0],data2[N5:N6,1],'m^',alpha=0.5)
    plt.show()

    return data1, data2, data2_0, permutation, type1, type2

def project_high_dim(data, d, p, data2_0=None, flag=0):
    T = np.random.normal(0,0.5,(d,p))

    if flag:
        return np.matmul(data, T), np.matmul(data2_0, T)
    return np.matmul(data, T)


def project_high_dim_dropout(data, d, p, data2_0=None, flag=0, fraction=0.5):
    T = np.random.normal(0,0.5,(d,p))
    data_highDim = np.matmul(data, T)
    dropout = np.random.uniform(0,1,np.shape(data_highDim))
    dropout[dropout<=fraction] = 0
    dropout[dropout>fraction] = 1
    if flag:
        data_highDim_0 = np.matmul(data2_0, T)
        return dropout*data_highDim, dropout*data_highDim_0
    return dropout*data_highDim
