import torch
import matplotlib.pyplot as plt

pred = torch.load('trsample.pt')['faceP0'][0,:,:,:]
pred = pred*255; pred = pred.permute(1,2,0).cpu()
gt = torch.load('trsample.pt')['faceT'][0,:,:,:]
gt = gt*255; gt = gt.permute(1,2,0).cpu()

norm = torch.load('../norm.pt')
std_face = norm['std_face']
mean_face = norm['mean_face']
mean_face = torch.unsqueeze(torch.unsqueeze(mean_face, 0), 0)
std_face = torch.unsqueeze(torch.unsqueeze(std_face, 0), 0)

pred = pred*std_face + mean_face
gt = gt*std_face + mean_face
plt.imshow(pred.int()); plt.show()
plt.imshow(gt.int()); plt.show()