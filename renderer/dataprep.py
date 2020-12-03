import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision.transforms import Resize

class prep(Dataset):
    
    def __init__(self, path, split):
        
        self.path = path+split+'/'
        self.datalist = os.listdir(self.path)
        self.datalist.sort()
        self.datalist = self.datalist
        
        norm = torch.load(path+'norm.pt')
        
        self.mean_face = norm['mean_face']
        self.mean_face = torch.unsqueeze(torch.unsqueeze(self.mean_face, 0), 0)
        self.std_face = norm['std_face']
        self.std_face = torch.unsqueeze(torch.unsqueeze(self.std_face, 0), 0)
        
        self.resize = Resize([227,227]) # required by pre trained encoder

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        filename = self.datalist[idx]
        triplet = torch.load(self.path+filename)
        del triplet['mel']
        
        face0 = triplet['face0']
        faceT = triplet['faceT']
        face0 -= self.mean_face; face0 /= self.std_face; face0 /= 255
        faceT -= self.mean_face; faceT /= self.std_face; faceT /= 255
        
        temp = triplet['key0'].int().numpy()
        temp[:,1] = np.clip(temp[:,1],0,face0.shape[0]-1)
        temp[:,0] = np.clip(temp[:,0],0,face0.shape[1]-1)
        key0 = torch.zeros((face0.shape[0],face0.shape[1],1))
        key0[temp[:,1], temp[:,0]] = 1 # create mask
        temp = triplet['keyT'].int().numpy()
        temp[:,1] = np.clip(temp[:,1],0,faceT.shape[0]-1)
        temp[:,0] = np.clip(temp[:,0],0,faceT.shape[1]-1)
        keyT = torch.zeros((faceT.shape[0],faceT.shape[1],1))
        keyT[temp[:,1], temp[:,0]] = 1 # create mask
        
        face0 = self.resize(face0.permute(-1,0,1))
        faceT = self.resize(faceT.permute(-1,0,1))
        key0 = self.resize(key0.permute(-1,0,1))
        keyT = self.resize(keyT.permute(-1,0,1))
        
        triplet['key0'] = key0
        triplet['keyT'] = keyT
        triplet['face0'] = face0
        triplet['faceT'] = faceT
        
        return triplet