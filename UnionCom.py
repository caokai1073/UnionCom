'''
---------------------
UnionCom fucntions
author: Kai Cao
e-mail:caokai@amss.ac.cn
MIT LICENSE
---------------------
'''
import os
import sys
import numpy as np
from sklearn import preprocessing
from unioncom.model import *
from unioncom.train import train
from unioncom.utils import *
from unioncom.test import *

# print(os.getcwd())

class params():
    epoch_pd = 20000
    epoch_DNN = 200
    epsilon = 0.001
    lr = 0.001
    batch_size = 100
    rho = 10
    log_DNN = 10
    log_pd = 500
    manual_seed = 8888
    delay = 0
    kmax = 20
    beta = 1

def fit_transform(dataset, datatype=None, epoch_pd=20000, epoch_DNN=200, \
    epsilon=0.001, lr=0.001, batch_size=100, rho=10, beta=1,\
    log_DNN=10, log_pd=500, manual_seed=666, delay=0, kmax=20, distance = 'geodesic', \
    output_dim=32, test=False):

    '''
    parameters:
    dataset: list of datasets to be integrated. [dataset1, dataset2, ...].
    datatype: list of data type. [datatype1, datatype2, ...].
    epoch_pd: epoch of Prime-dual algorithm.
    epoch_DNN: epoch of training Deep Neural Network.
    epsilon: training rate of data matching matrix F.
    lr: training rate of DNN.
    batch_size: training batch size of DNN.
    beta: trade-off parameter of structure preserving and point matching.
    rho: training damping term.
    log_DNN: log step of training DNN.
    log_pd: log step of prime dual method
    manual_seed: random seed.
    distance: mode of distance.
    output_dim: output dimension of integrated data.
    test: test the match fraction and label transfer accuracy, need datatype.
    ---------------------
    '''
    params.epoch_pd = epoch_pd
    params.epoch_DNN = epoch_DNN
    params.epsilon = epsilon
    params.lr = lr
    params.batch_size = batch_size
    params.rho = rho
    params.log_DNN = log_DNN
    params.log_pd = log_pd
    params.manual_seed = manual_seed
    params.delay = delay
    params.beta = beta
    params.kmax = kmax

    init_random_seed(manual_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if test: 
        for i in range(len(datatype)):
            datatype[i] = datatype[i].astype(np.int)
        datatype = np.array(datatype)

    row = []
    col = []
    for i in range(len(dataset)):
        row.append(np.shape(dataset[i])[0])
        col.append(np.shape(dataset[i])[1])

    print("Shape of Raw data")
    for i in range(len(dataset)):
        print("Dataset {}:".format(i), np.shape(dataset[i]))

    input_dim = col

    #### compute the distance matrix and the largest connected component
    dist = []
    kmin = []

    for i in range(len(dataset)):
    
        dataset[i] = (dataset[i]- np.min(dataset[i])) / (np.max(dataset[i]) - np.min(dataset[i]))

        if distance == 'geodesic':
            dist_tmp, k_tmp = geodesic_distances(dataset[i], params.kmax)
            dist.append(dist_tmp)
            kmin.append(k_tmp)

        if distance == 'euclidean':
            dist_tmp, k_tmp = euclidean_distances(dataset[i])
            dist.append(dist_tmp)
            kmin.append(k_tmp)
        
    dist = np.array(dist)

    P_joint = []
    for i in range(len(dataset)):
        P_joint.append(p_joint(dist[i], kmin[i]))

    net = Project(input_dim, output_dim)
    Project_net = init_model(net, device, restore=None)

    result = train(Project_net, params, dataset, dist, P_joint, device)

    return test_UnionCom(result, dataset, datatype, params, device, test)
