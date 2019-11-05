from torchvision import models
import torch.nn as nn

class Project(nn.Module):
	def __init__(self):
		super(Project, self).__init__()
		self.restored = False

		# self.feature1 = nn.Sequential(
		# 	nn.Linear(1446,5000),
		# 	nn.BatchNorm1d(5000),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.Linear(5000,3000),
		# 	nn.BatchNorm1d(3000),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.Linear(3000,500),
		# 	nn.BatchNorm1d(500),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.Linear(500,200),
		# 	nn.BatchNorm1d(200),
		# 	nn.LeakyReLU(0.1, True),
		# )

		# self.feature2 = nn.Sequential(
		# 	nn.Linear(1446,5000),
		# 	nn.BatchNorm1d(5000),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.Linear(5000,3000),
		# 	nn.BatchNorm1d(3000),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.Linear(3000,500),
		# 	nn.BatchNorm1d(500),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.Linear(500,200),
		# 	nn.BatchNorm1d(200),
		# 	nn.LeakyReLU(0.1, True),
		# )
		self.feature1 = nn.Sequential(
			nn.Linear(1000,10000),
			nn.BatchNorm1d(10000),
			nn.LeakyReLU(0.1, True),
			nn.Linear(10000,5000),
			nn.BatchNorm1d(5000),
			nn.LeakyReLU(0.1, True),
			nn.Linear(5000,3000),
			nn.BatchNorm1d(3000),
			nn.LeakyReLU(0.1, True),
			nn.Linear(3000,2000),
			nn.BatchNorm1d(2000),
			nn.LeakyReLU(0.1, True),
		)

		self.feature2 = nn.Sequential(
			nn.Linear(500,10000),
			nn.BatchNorm1d(10000),
			nn.LeakyReLU(0.1, True),
			nn.Linear(10000,5000),
			nn.BatchNorm1d(5000),
			nn.LeakyReLU(0.1, True),
			nn.Linear(5000,3000),
			nn.BatchNorm1d(3000),
			nn.LeakyReLU(0.1, True),
			nn.Linear(3000,2000),
			nn.BatchNorm1d(2000),
			nn.LeakyReLU(0.1, True),
		)

		self.feature3 = nn.Sequential(
			nn.Linear(2000,1000),
			nn.BatchNorm1d(1000),
			nn.LeakyReLU(0.1, True),
			nn.Linear(1000,500),
			nn.BatchNorm1d(500),
			nn.LeakyReLU(0.1, True),
			nn.Linear(500,32),
		)

	def forward(self, input_data, domain):
		if domain == 1:
			feature = self.feature1(input_data)
		else:
			feature = self.feature2(input_data)

		feature = self.feature3(feature)

		return feature





