import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def visualize(data, data_integrated, datatype=None):

    dataset_num = len(data)
    font_label = {
             'weight': 'normal',
             'size': 22,
         }

    font_title = {
             'weight': 'normal',
             'size': 25,
         }

    styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c'] 

    embedding = []
    dataset_xyz = []
    for i in range(dataset_num):
        dataset_xyz.append("dataset{:d}".format(i+1))
        embedding.append(PCA(n_components=2).fit_transform(data[i]))
   
    if datatype is not None:
        for i in range(dataset_num):
            fig = plt.figure()
            for j in set(datatype[i]):
                index = np.where(datatype[i]==j) 
                plt.scatter(embedding[i][index,0], embedding[i][index,1], c=styles[j], s=5.)
            plt.title(dataset_xyz[i], font_title)
            plt.xlabel('Dimension-1', font_label)
            plt.ylabel('Dimension-2', font_label)
            plt.legend()
    else:
        for i in range(dataset_num):
            fig = plt.figure()
            plt.scatter(embedding[i][:,0], embedding[i][:,1],c=styles[i], s=5.)
            plt.title(dataset_xyz[i], font_title)
            plt.xlabel('Dimension-1', font_label)
            plt.ylabel('Dimension-2', font_label)
            plt.legend()



    data_all = np.vstack((data_integrated[0], data_integrated[1]))
    for i in range(2, dataset_num):
        data_all = np.vstack((data_all, data_integrated[i]))

    embedding_all = PCA(n_components=2).fit_transform(data_all)

    tmp = 0
    num = [0]
    for i in range(dataset_num):
        num.append(tmp+np.shape(data_integrated[i])[0])
        tmp += np.shape(data_integrated[i])[0]

    embedding = []
    for i in range(dataset_num):
        embedding.append(embedding_all[num[i]:num[i+1]])

    color = [[1,0.5,0], [0.2,0.4,0.1], [0.1,0.2,0.8], [0.5, 1, 0.5], [0.1, 0.8, 0.2]]
    # marker=['x','^','o','*','v']
    
    if datatype is not None:
        fig = plt.figure()
        for i in range(dataset_num):
            plt.scatter(embedding[i][:,0], embedding[i][:,1], c=color[i], s=5., alpha=0.8)
        plt.title('Embeddings', font_title)
        plt.xlabel('Dimension-1', font_label)
        plt.ylabel('Dimension-2', font_label)
        plt.legend()

        fig = plt.figure()
        for i in range(dataset_num):  
            for j in set(datatype[i]):
                index = np.where(datatype[i]==j) 
                plt.scatter(embedding[i][index,0], embedding[i][index,1], c=styles[j], s=5., alpha=0.8)
        plt.title('Cell Types', font_title)
        plt.xlabel('Dimension-1', font_label)
        plt.ylabel('Dimension-2', font_label)
        plt.legend()
    else:
        fig = plt.figure()
        for i in range(dataset_num):
            plt.scatter(embedding[i][:,0], embedding[i][:,1], c=color[i], s=5., alpha=0.8)
        plt.title('Embeddings', font_title)
        plt.xlabel('Dimension-1', font_label)
        plt.ylabel('Dimension-2', font_label)
        plt.legend()

    plt.show()
