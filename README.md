# UnionCom

## Enviroment:
Ubuntu 18.04.3 LTS  
python 3.6.8  
numpy 1.17.3  
torch 1.3.0  
torchvision 0.4.1  
scikit-learn 0.21.3  

## Install
UnionCom software is available by
```
pip3 install UnionCom
```

## Usage for integrate data
if ```data0.txt, ... ,dataN.txt``` to be integrated, then use
```
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
data0 = np.loadtxt("data0.txt")
label0 = np.loadtxt("label0.txt")
...
dataN = np.loadtxt("dataN.txt")
labelN = np.loadtxt("labelN.txt")

data = [data0, ..., dataN]
label = [label0,...,labelN]

integrated_data = UnionCom.fit_transform(data, label, test=True)
```









