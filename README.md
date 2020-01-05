# UnionCom

## Enviroment:
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
UnionCom.fit_transform(dataset, datatype=None, epoch_total=1, epoch_pd=30000, epoch_DNN=100, epsilon=0.001, 
epsilon_a=0.001, lr=0.001, batch_size=100, rho=10, log_step=10, manual_seed=8888, delay=0, beta=0, 
usePercent=1.0, kmax=20, distance = 'geodesic', output_dim=32, test=False)
```
```
dataset: list of datasets to be integrated. [dataset1, dataset2, ...].
datatype: list of data type. [datatype1, datatype2, ...].
epoch_total: total epoch of training, used when data subsampling is used.
epoch_pd: epoch of Prime-dual algorithm.
epoch_DNN: epoch of training Deep Neural Network.
epsilon: training rate of data matching matrix F.
epsilon_a: training rate of scaling factor alpha.
lr: training rate of DNN.
batch_size: training batch size of DNN.
rho: training damping term.
log_step: log step of training DNN.
manual_seed: random seed.
delay: delay steps of alpha.
beta: trade-off parameter of structure preserving and point matching.
usePercent: data subsampling percentage. (from 0.0 to 1.0)
kmax: maximum value of knn when constructing geodesic distance matrix
distance: mode of distance. [geodesic, Euclidean]
output_dim: output dimension of integrated data.
test: test the match fraction and label transfer accuracy, need datatype.
```

## Usage for integrate data
if ```data0.txt, ... ,dataN.txt``` to be integrated, then use
```
from unioncom import UnionCom
import  numpy as np

data0 = np.loadtxt("data0.txt")
...
dataN = np.loadtxt("dataN.txt")

data = [data0, ..., dataN]

integrated_data = UnionCom.fit_transform(data)

new_data0 = integrated_data[0]
...
new_dataN = integrated_data[N]
```

## Usage for test label transfer accuracy
```
from unioncom import UnionCom
import  numpy as np

data0 = np.loadtxt("data0.txt")
label0 = np.loadtxt("label0.txt")
...
dataN = np.loadtxt("dataN.txt")
labelN = np.loadtxt("labelN.txt")

data = [data0, ..., dataN]
label = [label0,...,labelN]

integrated_data = UnionCom.fit_transform(data, label, test=True)
```









