from torchvision import models
import torch.nn as nn

class Project(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(Project, self).__init__()
		self.restored = False
		self.input_dim = input_dim
		self.output_dim = output_dim

		num = len(input_dim)
		feature = []

		for i in range(num):
			feature.append(
			nn.Sequential(
			nn.Linear(self.input_dim[i],2*self.input_dim[i]),
			nn.BatchNorm1d(2*self.input_dim[i]),
			nn.LeakyReLU(0.1, True),
			nn.Linear(2*self.input_dim[i],2*self.input_dim[i]),
			nn.BatchNorm1d(2*self.input_dim[i]),
			nn.LeakyReLU(0.1, True),
			nn.Linear(2*self.input_dim[i],self.input_dim[i]),
			nn.BatchNorm1d(self.input_dim[i]),
			nn.LeakyReLU(0.1, True),
			nn.Linear(self.input_dim[i],self.output_dim),
			nn.BatchNorm1d(self.output_dim),
			nn.LeakyReLU(0.1, True),
		))

		self.feature = nn.ModuleList(feature)

		self.feature_show = nn.Sequential(
			nn.Linear(self.output_dim,self.output_dim),
			nn.BatchNorm1d(self.output_dim),
			nn.LeakyReLU(0.1, True),
			nn.Linear(self.output_dim,self.output_dim),
			nn.BatchNorm1d(self.output_dim),
			nn.LeakyReLU(0.1, True),
			nn.Linear(self.output_dim,self.output_dim),
		)

	def forward(self, input_data, domain):
		feature = self.feature[domain](input_data)
		feature = self.feature_show(feature)

		return feature





