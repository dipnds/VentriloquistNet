import matplotlib.pyplot as plt
import torch

mean_kp = torch.load('kp_general.pt').flatten().unsqueeze(0).unsqueeze(0)
wt = torch.zeros(136); wt[-40:] = 1; wt[:34] = 1
wt = torch.diag(wt)
mean_kp = torch.matmul(mean_kp,wt)
mean_kp = mean_kp.reshape((-1,2))
plt.figure(); plt.scatter(mean_kp[:,0],-mean_kp[:,1],s=2); plt.savefig('meankp.png')
