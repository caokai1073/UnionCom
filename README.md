# UnionCom

## Paper
[Unsupervised Topological Alignment for Single-Cell Multi-Omics Integration](https://www.biorxiv.org/content/10.1101/2020.02.02.931394v1)

## Enviroment
Ubuntu 18.04.3 LTS  
python >= 3.6

numpy 1.17.3  
torch 1.3.0  
torchvision 0.4.1  
scikit-learn 0.21.3  

## Install
UnionCom software is available on the Python package index (PyPI). To install it using pip, simply type:
```
pip3 install unioncom
```

## Parameters
```
UnionCom.fit_transform(dataset, datatype=None, epoch_pd=30000, epoch_DNN=100, epsilon=0.001, 
epsilon_a=0.001, lr=0.001, batch_size=100, rho=10, log_DNN=10, manual_seed=8888, delay=0, 
beta=1, kmax=20, distance = 'geodesic', output_dim=32, test=False)
```
```
dataset: list of datasets to be integrated. [dataset1, dataset2, ...].
datatype: list of data type. [datatype1, datatype2, ...].
epoch_pd: epoch of Prime-dual algorithm.
epoch_DNN: epoch of training Deep Neural Network.
epsilon: training rate of data matching matrix F.
epsilon_a: training rate of scaling factor alpha.
lr: training rate of DNN.
batch_size: training batch size of DNN.
rho: training damping term.
log_DNN: log step of training DNN.
manual_seed: random seed.
delay: delay steps of alpha. (from 0 to epoch_pd)
beta: trade-off parameter of structure preserving and point matching.
kmax: maximum value of knn when constructing geodesic distance matrix
distance: mode of distance. [geodesic, euclidean]
output_dim: output dimension of integrated data.
test: test the match fraction and label transfer accuracy, need datatype.
```

## Integrate data
```data_0.txt, ... ,data_N.txt``` to be integrated, use
```
from unioncom import UnionCom
import numpy as np

data_0 = np.loadtxt("data_0.txt")
...
data_N = np.loadtxt("data_N.txt")

data = [data_0, ..., data_N]

integrated_data = UnionCom.fit_transform(data)

matched_data_0 = integrated_data[0]
...
matched_data_N = integrated_data[N]
```

## Test label transfer accuracy
```
from unioncom import UnionCom
import numpy as np

data_0 = np.loadtxt("data_0.txt")
label_0 = np.loadtxt("label_0.txt")
...
data_N = np.loadtxt("data_N.txt")
label_N = np.loadtxt("label_N.txt")

data = [data_0, ..., data_N]
label = [label_0,...,label_N]

integrated_data = UnionCom.fit_transform(data, label, test=True)
```

## Example
We obtained **Cheow_expression.csv** and **Cheow_methylation.csv** from https://github.com/jw156605/MATCHER

[Result of scGEM data by UnionCom](https://github.com/caokai1073/UnionCom/blob/master/result.pdf)








