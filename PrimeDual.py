# import os
import random
import torch
import numpy as np

def cor_pairs_match_Adam(Kx, Kz, N, params, p1, p2, device):
	print("use device:", device)
	Kx = Kx / N 
	Kz = Kz / N
	Kx = torch.from_numpy(Kx).float().to(device)
	Kz = torch.from_numpy(Kz).float().to(device)
	m = np.shape(Kx)[0]
	n = np.shape(Kz)[0]
	F = np.zeros((m,n))
	F = torch.from_numpy(F).float().to(device)
	Im = torch.ones((m,1)).float().to(device)
	In = torch.ones((n,1)).float().to(device)
	Lambda = torch.zeros((n,1)).float().to(device)
	Mu = torch.zeros((m,1)).float().to(device)
	S = torch.zeros((n,1)).float().to(device)
	a = np.sqrt(p2/p1)
	pho1 = 0.9
	pho2 = 0.999
	delta = 10e-8
	Fst_moment = torch.zeros((m,n)).float().to(device)
	Snd_moment = torch.zeros((m,n)).float().to(device)
	i=0
	while(i<params.epoch_pd):
		grad = 2*torch.mm(F, torch.mm(Kz, torch.mm(torch.t(F), torch.mm(F, torch.t(Kz))))) \
		+ 2*torch.mm(F, torch.mm(torch.t(Kz), torch.mm(torch.t(F), torch.mm(F, Kz)))) \
		- 2*a*torch.mm(torch.t(Kx), torch.mm(F, Kz)) - 2*a*torch.mm(Kx, torch.mm(F,torch.t(Kz))) + torch.mm(Mu, torch.t(In)) \
		+ torch.mm(Im, torch.t(Lambda)) + params.rho*(torch.mm(F, torch.mm(In, torch.t(In))) - torch.mm(Im, torch.t(In)) \
		+ torch.mm(Im, torch.mm(torch.t(Im), F)) + torch.mm(Im, torch.t(S-In)))
		i += 1
		Fst_moment = pho1*Fst_moment + (1-pho1)*grad
		Snd_moment = pho2*Snd_moment + (1-pho2)*grad*grad
		hat_Fst_moment = Fst_moment/(1-np.power(pho1,i))
		hat_Snd_moment = Snd_moment/(1-np.power(pho2,i))
		grad = hat_Fst_moment/(torch.sqrt(hat_Snd_moment)+delta)
		F_tmp = F - grad
		F_tmp[F_tmp<0]=0
		F = (1-params.epsilon)*F + params.epsilon*F_tmp

		grad_s = Lambda + params.rho*(torch.mm(torch.t(F), Im) - In + S)
		s_tmp = S - grad_s
		s_tmp[s_tmp<0]=0
		S = (1-params.epsilon)*S + params.epsilon*s_tmp
		Mu = Mu + params.epsilon*(torch.mm(F,In) - Im)
		Lambda = Lambda + params.epsilon*(torch.mm(torch.t(F), Im) - In + S)

		#### if scaling factor a changes too fast, we can delay the update of speed.
		if i>=params.delay:

			a = torch.trace(torch.mm(torch.t(Kx), torch.mm(torch.mm(F, Kz), torch.t(F)))) / \
			torch.trace(torch.mm(torch.t(Kx), Kx))

		if (i+1) % params.log_pd == 0:
			norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Kz), torch.t(F)))
			print("epoch:[{:d}/{:d}] err:{:.4f} alpha:{:.4f}".format(i+1, params.epoch_pd, norm2.data.item(), a))

	F = F.cpu().numpy()
	pairs = np.zeros(m)
	for i in range(m):
		pairs[i] = np.argsort(F[i])[-1]
	return pairs