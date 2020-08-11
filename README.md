# UnionCom

## Paper
[Unsupervised Topological Alignment for Single-Cell Multi-Omics Integration](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i48/5870490)

+ In the above paper, <a href="https://www.codecogs.com/eqnedit.php?latex=1_{n_x\times&space;n_y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1_{n_x\times&space;n_y}" title="1_{n_x\times n_y}" /></a> on Pages i50 and i51 should be corrected as <a href="https://www.codecogs.com/eqnedit.php?latex=1_{n_x\times&space;n_x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1_{n_x\times&space;n_x}" title="1_{n_x\times n_x}" /></a>. The authors thank Dr. Chanwoo park from Seoul National University for pointing out this typo.

## Enviroment
Ubuntu 18.04.3 LTS  
python >= 3.6

numpy 1.17.3  
torch 1.3.0  
torchvision 0.4.1  
scikit-learn 0.21.3  

## Install
UnionCom software is available on the Python package index (PyPI), latest version 0.2.1. To install it using pip, simply type:
```
pip3 install unioncom
```
## v0.2.1
+ Software optimization
+ Split function "train" into functions "Match" and "Project"
+ Use Kuhn-Munkres algorithm to find optimal pairs between datasets instead of parbabilistic matrix matching
+ Add a new parameter "project" to provide options for barycentric projection
+ Separate "test_label_transfer_accuracy" function from "fit_transform" function
+ fix some bugs

## Examples (jupyter notebook)

+ [Integration of simulations in UnionCom paper](https://github.com/caokai1073/UnionCom/blob/master/Examples/Simulation_example.ipynb)

+ [Integration of simulations in MMD-MA paper](https://github.com/caokai1073/UnionCom/blob/master/Examples/Simulation_data_from_MMD-MA.ipynb)

+ [Batch correction](https://github.com/caokai1073/UnionCom/blob/master/Examples/Batch_correction_example.ipynb)

+ [Integration of multi-omics data](https://github.com/caokai1073/UnionCom/blob/master/Examples/scGEM_and_scNMT_example.ipynb)

+ [Integration of datasets with specific cells](https://github.com/caokai1073/UnionCom/blob/master/Examples/dataset-specific_example.ipynb)


## Integrate data
Each row should contain the measured values for a single cell, and each column should contain the values of a feature across cells.
```data_0.txt, ... ,data_N.txt``` to be integrated, use

```python
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
To test the label transfer accuracy, you need to input cell types of ```data_0.txt, ... ,data_N.txt``` as ```type_0.txt, ... ,type_N.txt```
```python
type_0 = np.loadtxt("type_0.txt")
...
type_N = np.loadtxt("type_N.txt")
datatype = [type_0,...,type_N]

UnionCom.test_label_transfer_accuracy(data, datatype)
```

## Visualization by PCA
```python
type_0 = type_0.astype(np.int)
...
type_N = type_N.astype(np.int)
datatype = [type_0,...,type_N]

UnionCom.PCA_visualize(data, integrated_data) # without datatype
UnionCom.PCA_visualize(data, integrated_data, datatype) # with datatype
```

## Parameters of ```UnionCom.fit_transform```

The list of parameters is given below:
> + ```epoch_pd```: epoch of Prime-dual algorithm (default=20000).
> + ```epoch_DNN```: epoch of training Deep Neural Network (default=200).
> + ```epsilon```: training rate of data matching matrix F (default=0.001).
> + ```lr```: training rate of DNN (default=0.001).
> + ```batch_size```: training batch size of DNN (default=100).
> + ```rho```: training damping term (default=10).
> + ```delay```: delay steps of alpha (default=0).
> + ```beta```: trade-off parameter of structure preserving and point matching (default=1).
> + ```kmax```: maximum value of knn when constructing geodesic distance matrix (default=20).
> + ```output_dim```: output dimension of integrated data (default=32).

The other parameters include:
> + ```dataset```: list of datasets to be integrated. [dataset1, dataset2, ...].
> + ```datatype```: list of data type. [datatype1, datatype2, ...].
> + ```log_pd```: log step of Prime Dual (default=1000).
> + ```log_DNN```: log step of training DNN (default=10).
> + ```manual_seed```: random seed (default=666).
> + ```distance```: mode of distance. ['geodesic' (suggested for multimodal integration), 'euclidean'(suggested for batch correction)] (default='geodesic').
> + ```project```: mode of project, ['tsne', 'barycentric'] (default='tsne').



