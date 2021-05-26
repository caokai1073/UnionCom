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
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from unioncom.visualization import visualize
from unioncom.Model import model
from unioncom.utils import *
from unioncom.test import *

class UnionCom(object):

    """
    UnionCom software for single-cell mulit-omics data integration
    Published at https://academic.oup.com/bioinformatics/article/36/Supplement_1/i48/5870490

    parameters:
    -----------------------------
    dataset: list of datasets to be integrated. [dataset1, dataset2, ...].
    integration_type: "MultiOmics" or "BatchCorrect", default is "MultiOmics". "BatchCorrect" needs aligned features.
    epoch_pd: epoch of Prime-dual algorithm.
    epoch_DNN: epoch of training Deep Neural Network.
    epsilon: training rate of data matching matrix F.
    lr: training rate of DNN.
    batch_size: batch size of DNN.
    beta: trade-off parameter of structure preserving and matching.
    perplexity: perplexity of tsne projection
    rho: damping term.
    log_DNN: log step of training DNN.
    log_pd: log step of prime dual method
    manual_seed: random seed.
    delay: delay updata of alpha
    kmax: largest number of neighbors in geodesic distance 
    output_dim: output dimension of integrated data.
    distance_mode: mode of distance, 'geodesic' 
                                        or distances in sklearn.metrics.pairwise.pairwise_distances, 
                                        default is 'geodesic'.
    project_mode:　mode of project, ['tsne', 'barycentric'], default is tsne.
    -----------------------------

    Functions:
    -----------------------------
    fit_transform(dataset)              find correspondence between datasets, 
                                        align multi-omics data in a common embedded space
    match(data)                         find correspondence between datasets
    Prime_Dual(Kx, Ky, dx, dy)             Prime dual algorithm to find the optimal match
    project_barycentric(dataset, match_result)          barycentric projection (from SCOT)
    project_tsne(dataset, pairs_x, pairs_y, P_joint)    tsne-based projection
    Visualize(data, integrated_data, datatype, mode)    Visualization
    test_labelTA(integrated_data, datatype)             test label transfer accuracy
    -----------------------------

    Examples:
    -----------------------------
    input: numpy arrays with rows corresponding to samples and columns corresponding to features
    output: integrated numpy arrays
    >>> from unioncom import UnionCom
    >>> import numpy as np
    >>> data1 = np.loadtxt("./simu1/domain1.txt")
    >>> data2 = np.loadtxt("./simu1/domain2.txt")
    >>> type1 = np.loadtxt("./simu1/type1.txt")
    >>> type2 = np.loadtxt("./simu1/type2.txt")
    >>> type1 = type1.astype(np.int)
    >>> type2 = type2.astype(np.int)
    >>> uc = UnionCom.UnionCom()
    >>> integrated_data = uc.fit_transform(dataset=[data1,data2])
    >>> uc.test_labelTA(integrated_data, [type1,type2])
    >>> uc.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA')
    -----------------------------
    """

    def __init__(self, integration_type='MultiOmics', epoch_pd=2000, epoch_DNN=100, \
        epsilon=0.01, lr=0.001, batch_size=100, rho=10, beta=1, perplexity=30, \
        log_DNN=10, log_pd=100, manual_seed=666, delay=0, kmax=40,  \
        output_dim=32, distance_mode ='geodesic', project_mode='tsne'):

        self.integration_type = integration_type
        self.epoch_pd = epoch_pd
        self.epoch_DNN = epoch_DNN
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.rho = rho
        self.log_DNN = log_DNN
        self.log_pd = log_pd
        self.manual_seed = manual_seed
        self.delay = delay
        self.beta = beta
        self.perplexity = perplexity
        self.kmax = kmax
        self.output_dim = output_dim
        self.distance_mode = distance_mode
        self.project_mode = project_mode
        self.row = []
        self.col = []
        self.dist = []
        self.cor_dist = []

    def fit_transform(self, dataset=None):
        """
        find correspondence between datasets & align multi-omics data in a common embedded space
        """

        distance_modes =  ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 
            'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 
            'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 
            'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']

        if self.integration_type not in ['BatchCorrect','MultiOmics']:
                raise Exception("integration_type error! Enter MultiOmics or BatchCorrect.")

        if self.distance_mode is not 'geodesic' and self.distance_mode not in distance_modes:
                raise Exception("distance_mode error! Enter a correct distance_mode.")

        time1 = time.time()
        init_random_seed(self.manual_seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_num = len(dataset)
        for i in range(dataset_num):
            self.row.append(np.shape(dataset[i])[0])
            self.col.append(np.shape(dataset[i])[1])

        #### compute the distance matrix
        print("Shape of Raw data")
        for i in range(dataset_num):
            print("Dataset {}:".format(i), np.shape(dataset[i]))

            dataset[i] = (dataset[i]- np.min(dataset[i])) / (np.max(dataset[i]) - np.min(dataset[i]))

            if self.distance_mode == 'geodesic':
                distances = geodesic_distances(dataset[i], self.kmax)
                self.dist.append(np.array(distances))
            else:
                distances = pairwise_distances(dataset[i], metric=self.distance_mode)
                self.dist.append(distances)
            

            if self.integration_type == 'BatchCorrect':
                if self.distance_mode not in distance_modes:
                    raise Exception("Note that BatchCorrect needs aligned features.")
                else:
                    if self.col[i] != self.col[-1]:
                        raise Exception("BatchCorrect needs aligned features.")
                    cor_distances = pairwise_distances(dataset[i], dataset[-1], metric=self.distance_mode)
                    self.cor_dist.append(cor_distances)     

        # find correspondence between samples
        pairs_x = []
        pairs_y = []
        match_result = self.match(dataset=dataset)
        for i in range(dataset_num-1):
        	cost = np.max(match_result[i])-match_result[i]
        	row_ind,col_ind = linear_sum_assignment(cost)
        	pairs_x.append(row_ind)
        	pairs_y.append(col_ind)

        #  projection
        if self.project_mode == 'tsne':
            P_joint = []
            time1 = time.time()
            for i in range(dataset_num):
                P_joint.append(joint_probabilities(self.dist[i], self.perplexity))

            for i in range(dataset_num):
                if self.col[i] > 50:
                    dataset[i] = PCA(n_components=50).fit_transform(dataset[i])
                    self.col[i] = 50

            integrated_data = self.project_tsne(dataset, pairs_x, pairs_y, P_joint)

        elif self.project_mode == 'barycentric':
            integrated_data = self.project_barycentric(dataset, match_result)	

        else:
            raise Exception("Choose correct project_mode: 'tsne or barycentric'")

        print("---------------------------------")
        print("unionCom Done!")
        time2 = time.time()
        print('time:', time2-time1, 'seconds')

        return integrated_data

    def match(self, dataset):
        """
        Find correspondence between multi-omics datasets
        """

        dataset_num = len(dataset)
        cor_pairs = []
        N = np.int(np.max([len(l) for l in dataset]))
        for i in range(dataset_num-1):
            print("---------------------------------")
            print("Find correspondence between Dataset {} and Dataset {}".format(i+1, \
                len(dataset)))
            if self.integration_type == "MultiOmics":
                cor_pairs.append(self.Prime_Dual([self.dist[i], self.dist[-1]], dx=self.col[i], dy=self.col[-1]))
            else:
                cor_pairs.append(self.Prime_Dual(self.cor_dist[i]))

        print("Finished Matching!")
        return cor_pairs

    def Prime_Dual(self, dist, dx=None, dy=None):
        """
        prime dual combined with Adam algorithm to find the local optimal soluation
        """

        print("use device:", self.device)

        if self.integration_type == "MultiOmics":
            Kx = dist[0]
            Ky = dist[1]
            N = np.int(np.maximum(len(Kx), len(Ky)))
            Kx = Kx / N
            Ky = Ky / N
            Kx = torch.from_numpy(Kx).float().to(self.device)
            Ky = torch.from_numpy(Ky).float().to(self.device)
            a = np.sqrt(dy/dx)
            m = np.shape(Kx)[0]
            n = np.shape(Ky)[0]

        else:
            m = np.shape(dist)[0]
            n = np.shape(dist)[1]
            a=1
            dist = torch.from_numpy(dist).float().to(self.device)

        F = np.zeros((m,n))
        F = torch.from_numpy(F).float().to(self.device)
        Im = torch.ones((m,1)).float().to(self.device)
        In = torch.ones((n,1)).float().to(self.device)
        Lambda = torch.zeros((n,1)).float().to(self.device)
        Mu = torch.zeros((m,1)).float().to(self.device)
        S = torch.zeros((n,1)).float().to(self.device)
        
        pho1 = 0.9
        pho2 = 0.999
        delta = 10e-8
        Fst_moment = torch.zeros((m,n)).float().to(self.device)
        Snd_moment = torch.zeros((m,n)).float().to(self.device)

        i=0
        while(i<self.epoch_pd):

            ### compute gradient

            # tmp = Kx - torch.mm(F, torch.mm(Ky, torch.t(F)))
            # w_tmp = -4*torch.abs(tmp) * torch.sign(tmp)
            # grad1 = torch.mm(w_tmp, torch.mm(F, torch.t(Ky)))

            # tmp = torch.mm(torch.t(F), torch.mm(a*Kx, F)) - Ky
            # w_tmp = 4*torch.abs(tmp) * torch.sign(tmp)
            # grad2 = torch.mm(Kx, torch.mm(F, torch.t(w_tmp)))

            if self.integration_type == "MultiOmics":
                grad = 4*torch.mm(F, torch.mm(Ky, torch.mm(torch.t(F), torch.mm(F, Ky)))) \
                - 4*a*torch.mm(Kx, torch.mm(F,Ky)) + torch.mm(Mu, torch.t(In)) \
                + torch.mm(Im, torch.t(Lambda)) + self.rho*(torch.mm(F, torch.mm(In, torch.t(In))) - torch.mm(Im, torch.t(In)) \
                + torch.mm(Im, torch.mm(torch.t(Im), F)) + torch.mm(Im, torch.t(S-In)))
            else:
                grad = dist + torch.mm(Im, torch.t(Lambda)) + self.rho*(torch.mm(F, torch.mm(In, torch.t(In))) - torch.mm(Im, torch.t(In)) \
                + torch.mm(Im, torch.mm(torch.t(Im), F)) + torch.mm(Im, torch.t(S-In)))
            # print(dist)
            ### adam momentum
            i += 1
            Fst_moment = pho1*Fst_moment + (1-pho1)*grad
            Snd_moment = pho2*Snd_moment + (1-pho2)*grad*grad
            hat_Fst_moment = Fst_moment/(1-np.power(pho1,i))
            hat_Snd_moment = Snd_moment/(1-np.power(pho2,i))
            grad = hat_Fst_moment/(torch.sqrt(hat_Snd_moment)+delta)
            F_tmp = F - grad
            F_tmp[F_tmp<0]=0

            ### update 
            F = (1-self.epsilon)*F + self.epsilon*F_tmp

            ### update slack variable
            grad_s = Lambda + self.rho*(torch.mm(torch.t(F), Im) - In + S)
            s_tmp = S - grad_s
            s_tmp[s_tmp<0]=0
            S = (1-self.epsilon)*S + self.epsilon*s_tmp

            ### update dual variables
            Mu = Mu + self.epsilon*(torch.mm(F,In) - Im)
            Lambda = Lambda + self.epsilon*(torch.mm(torch.t(F), Im) - In + S)

            #### if scaling factor changes too fast, we can delay the update
            if self.integration_type == "MultiOmics":
                if i>=self.delay:
                    a = torch.trace(torch.mm(Kx, torch.mm(torch.mm(F, Ky), torch.t(F)))) / \
                    torch.trace(torch.mm(Kx, Kx))

            if (i+1) % self.log_pd == 0:
                if self.integration_type == "MultiOmics":
                    norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Ky), torch.t(F)))
                    print("epoch:[{:d}/{:d}] err:{:.4f} alpha:{:.4f}".format(i+1, self.epoch_pd, norm2.data.item(), a))
                else:
                    norm2 = torch.norm(dist*F)
                    print("epoch:[{:d}/{:d}] err:{:.4f}".format(i+1, self.epoch_pd, norm2.data.item()))

        F = F.cpu().numpy()
        return F

    def project_barycentric(self, dataset, match_result):
        print("---------------------------------")
        print("Begin finding the embedded space")
        integrated_data = []
        for i in range(len(dataset)-1):
            integrated_data.append(np.matmul(match_result[i], dataset[-1]))
        integrated_data.append(dataset[-1])
        print("Done")
        return integrated_data

    def project_tsne(self, dataset, pairs_x, pairs_y, P_joint):
        """
        tsne-based projection (nonlinear method) to match and preserve structures of different modalities.
        Here we provide a way using neural network to find the embbeded space. 
        However, traditional gradient descent method can also be used.
        """
        print("---------------------------------")
        print("Begin finding the embedded space")

        net = model(self.col, self.output_dim)
        Project_DNN = init_model(net, self.device, restore=None)

        optimizer = optim.RMSprop(Project_DNN.parameters(), lr=self.lr)
        c_mse = nn.MSELoss()
        Project_DNN.train()

        dataset_num = len(dataset)

        for i in range(dataset_num):
            P_joint[i] = torch.from_numpy(P_joint[i]).float().to(self.device)
            dataset[i] = torch.from_numpy(dataset[i]).float().to(self.device)

        for epoch in range(self.epoch_DNN):
            len_dataloader = np.int(np.max(self.row)/self.batch_size)
            if len_dataloader == 0:
                len_dataloader = 1
                self.batch_size = np.max(self.row)
            for step in range(len_dataloader):
                KL_loss = []
                for i in range(dataset_num):
                    random_batch = np.random.randint(0, self.row[i], self.batch_size)
                    data = dataset[i][random_batch]
                    P_tmp = torch.zeros([self.batch_size, self.batch_size]).to(self.device)
                    for j in range(self.batch_size):
                        P_tmp[j] = P_joint[i][random_batch[j], random_batch]
                    P_tmp = P_tmp / torch.sum(P_tmp)
                    low_dim_data = Project_DNN(data, i)
                    Q_joint = Q_tsne(low_dim_data)

                    ## loss of structure preserving 
                    KL_loss.append(torch.sum(P_tmp * torch.log(P_tmp / Q_joint)))

        		## loss of structure matching 
                feature_loss = np.array(0)
                feature_loss = torch.from_numpy(feature_loss).to(self.device).float()
                for i in range(dataset_num-1):

                    low_dim = Project_DNN(dataset[i][pairs_x[i]], i)
                    low_dim_biggest_dataset = Project_DNN(dataset[dataset_num-1][pairs_y[i]], len(dataset)-1)
                    feature_loss += c_mse(low_dim, low_dim_biggest_dataset)
                    # min_norm = torch.min(torch.norm(low_dim), torch.norm(low_dim_biggest_dataset))
                    # feature_loss += torch.abs(torch.norm(low_dim) - torch.norm(low_dim_biggest_dataset))/min_norm

                loss = self.beta * feature_loss
                for i in range(dataset_num):
                    loss += KL_loss[i]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % self.log_DNN == 0:
                print("epoch:[{:d}/{}]: loss:{:4f}, align_loss:{:4f}".format(epoch+1, \
                    self.epoch_DNN, loss.data.item(), feature_loss.data.item()))

        integrated_data = []
        for i in range(dataset_num):
            integrated_data.append(Project_DNN(dataset[i], i))
            integrated_data[i] = integrated_data[i].detach().cpu().numpy()
        print("Done")
        return integrated_data

    def Visualize(self, data, integrated_data, datatype=None, mode='PCA'):
        if datatype is not None:
            visualize(data, integrated_data, datatype, mode=mode)
        else:
            visualize(data, integrated_data, mode=mode)

    def test_LabelTA(self, integrated_data, datatype):

        test_UnionCom(integrated_data, datatype)


# if __name__ == '__main__':

    # data1 = np.loadtxt("./Seurat_scRNA/CTRL_PCA.txt")
    # data2 = np.loadtxt("./Seurat_scRNA/STIM_PCA.txt")
    # type1 = np.loadtxt("./Seurat_scRNA/CTRL_type.txt")
    # type2 = np.loadtxt("./Seurat_scRNA/STIM_type.txt")

    ### batch correction for HSC data
    # data1 = np.loadtxt("./hsc/domain1.txt")
    # data2 = np.loadtxt("./hsc/domain2.txt")
    # type1 = np.loadtxt("./hsc/type1.txt")
    # type2 = np.loadtxt("./hsc/type2.txt")

    ### UnionCom simulation
    # data1 = np.loadtxt("./simu1/domain1.txt")
    # data2 = np.loadtxt("./simu1/domain2.txt")
    # type1 = np.loadtxt("./simu1/type1.txt")
    # type2 = np.loadtxt("./simu1/type2.txt")
    #-------------------------------------------------------

    ### MMD-MA simulation
    # data1 = np.loadtxt("./MMD/s1_mapped1.txt")
    # data2 = np.loadtxt("./MMD/s1_mapped2.txt")
    # type1 = np.loadtxt("./MMD/s1_type1.txt")
    # type2 = np.loadtxt("./MMD/s1_type2.txt")
    #-------------------------------------------------------

    ### scGEM data
    # data1 = np.loadtxt("./scGEM/GeneExpression.txt")
    # data2 = np.loadtxt("./scGEM/DNAmethylation.txt")
    # type1 = np.loadtxt("./scGEM/type1.txt")
    # type2 = np.loadtxt("./scGEM/type2.txt")
    #-------------------------------------------------------

    ### scNMT data
    # data1 = np.loadtxt("./scNMT/Paccessibility_300.txt")
    # data2 = np.loadtxt("./scNMT/Pmethylation_300.txt")
    # data3 = np.loadtxt("./scNMT/RNA_300.txt")
    # type1 = np.loadtxt("./scNMT/type1.txt")
    # type2 = np.loadtxt("./scNMT/type2.txt")
    # type3 = np.loadtxt("./scNMT/type3.txt")
    # not_connected, connect_element, index = Maximum_connected_subgraph(data3, 40)
    # if not_connected:
    # 	data3 = data3[connect_element[index]]
    # 	type3 = type3[connect_element[index]]
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data3 = min_max_scaler.fit_transform(data3)
    # print(np.shape(data3))
    #-------------------------------------------------------

    # print(np.shape(data1))
    # print(np.shape(data2))

    ### integrate two datasets
    # type1 = type1.astype(np.int)
    # type2 = type2.astype(np.int)
    # uc = UnionCom(distance_mode='geodesic', project_mode='tsne', integration_type="MultiOmics", batch_size=100)
    # integrated_data = uc.fit_transform(dataset=[data1,data2])
    # uc.test_LabelTA(integrated_data, [type1,type2])
    # uc.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA')

    ## integrate three datasets
    # type1 = type1.astype(np.int)
    # type2 = type2.astype(np.int)
    # type3 = type3.astype(np.int)
    # datatype = [type1,type2,type3]
    # uc = UnionCom()

    # inte = uc.fit_transform([data1,data2,data3])
    # uc.test_LabelTA(inte, [type1,type2,type3])
    # uc.Visualize([data1,data2,data3], inte, datatype, mode='UMAP')
