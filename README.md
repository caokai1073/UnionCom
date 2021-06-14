# UnionCom

## Paper
[Unsupervised Topological Alignment for Single-Cell Multi-Omics Integration](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i48/5870490)

+ Corrigendum: In the above paper, $1_{n_x\times n_y}$ on Pages i50 and i51 should be corrected as $1_{n_x\times n_x}$. The authors thank Dr. Chanwoo park from Seoul National University for pointing out this typo.

## Enviroment
python >= 3.6

numpy 1.19.5  
torch 1.7.0  
scipy 1.4.1
torchvision 0.4.1  
scikit-learn 0.23.2  
umap-learn 0.3.10

## Install
UnionCom software is available on the Python package index (PyPI), latest version 0.4.0. To install it using pip, simply type:
```
pip3 install unioncom
```

## Change Log
### v0.4.0
+ Add batch effect correction method by setting integration_type="BatchCorrect";
+ Add more distances (e.g. cosine, cityblock, see sklearn.metrics.pairwise) to formulate distance matrices. 
+ Fix some bugs;

## Examples (jupyter notebook)

+ [Integration of simulation 1 in UnionCom paper](https://github.com/caokai1073/UnionCom/blob/master/Examples/Simulation1.ipynb)

+ [Integration of simulation 2 in UnionCom paper](https://github.com/caokai1073/UnionCom/blob/master/Examples/Simulation2.ipynb)

+ [Integration of simulations in MMD-MA paper](https://github.com/caokai1073/UnionCom/blob/master/Examples/MMD-MA-simulations.ipynb)

+ [Batch correction of HSC data](https://github.com/caokai1073/UnionCom/blob/master/Examples/HSC.ipynb)

+ [Integration of scGEM data](https://github.com/caokai1073/UnionCom/blob/master/Examples/scGEM.ipynb)

+ [Integration of scNMT data](https://github.com/caokai1073/UnionCom/blob/master/Examples/scNMT.ipynb)

Each row should contain the measured values for a single cell, and each column should contain the values of a feature across cells.

```python
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
>>> uc.test_LabelTA(integrated_data, [type1,type2])
>>> uc.Visualize([data1,data2], integrated_data, mode='PCA') # without datatype
>>> uc.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA') # with datatype
```

## Parameters of ```class UnionCom```

The list of parameters is given below:
> + ```epoch_pd```: epoch of Prime-dual algorithm (default=2000).
> + ```epoch_DNN```: epoch of training Deep Neural Network (default=100).
> + ```epsilon```: training rate of data matching matrix F (default=0.01).
> + ```lr```: training rate of DNN (default=0.001).
> + ```batch_size```: training batch size of DNN (default=100).
> + ```rho```: training damping term (default=10).
> + ```delay```: delay steps of alpha (default=0).
> + ```beta```: trade-off parameter of structure preserving and point matching (default=1).
> + ```perplexity```: perplexity of tsne projection (default=30)
> + ```kmax```: maximum value of knn when constructing geodesic distance matrix (default=40).
> + ```output_dim```: output dimension of integrated data (default=32).

The other parameters include:
> + ```log_pd```: log step of Prime Dual (default=1000).
> + ```log_DNN```: log step of training DNN (default=10).
> + ```manual_seed```: random seed (default=666).
> + ```distance_mode```: mode of distance. ['geodesic' (suggested for multimodal integration), 'euclidean'(suggested for batch correction)] (default='geodesic').
> + ```project_mode```: mode of project, ['tsne', 'barycentric'] (default='tsne').
> + ```integration_type```: "MultiOmics" or "BatchCorrect". "BatchCorrect" needs aligned features. (default='MultiOmics')

### Contact via caokai@amss.ac.cn

