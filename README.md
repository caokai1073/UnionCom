# UnionCom

## Enviroment:
Ubuntu 18.04.3 LTS  
python 3.6.8  
numpy 1.17.3  
torch 1.3.0  
torchvision 0.4.1  
scikit-learn 0.21.3  

## simu1
`python3 main.py --isRealData 0 --simu 1 --epoch_pd 60000 --delay 10000`

The input number of neural netowrk of `self.feature1` and `self.feature2` in  model.py should be 1000 and 500.


## simu2
`python3 main.py --isRealData 0 --simu 2 --epoch_pd 60000 --delay 10000`

The input number of neural netowrk of `self.feature1` and `self.feature2` in  model.py should be 1000 and 500.


## simu3
`python3 main.py --isRealData 0 --simu 3 --epoch_pd 60000 --delay 10000`

The input number of neural netowrk of `self.feature1` and `self.feature2` in  model.py should be 1000 and 500.


## HSCs
`python3 main.py --isRealData 1 --epoch_pd 60000 --delay 0`

The input number of neural netowrk of `self.feature1` and `self.feature2` in  model.py should be 1446 and 1446.


## sc-GEM
`python3 main.py --isRealData 1 --epoch_pd 100000 --delay 10000`

The input number of neural netowrk of `self.feature1` and `self.feature2` in  model.py should be 34 and 27.


## AP&NF
`python3 main.py --isRealData 1 --epoch_pd 150000 --delay 100000`
Te input number of neural netowrk of `self.feature1` and `self.feature2` in  model.py should be 5081 and 5308.
