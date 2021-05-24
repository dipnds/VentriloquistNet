import torch
from torch.utils.data import Dataset
import os

class prep(Dataset):
    
    def __init__(self, path, split):
        
        self.path = path+split+'/'
        self.datalist = os.listdir(self.path)
        self.datalist.sort()
        self.datalist = self.datalist
        
        norm = torch.load(path+'norm.pt')
        
        self.mean_mel = norm['mean_mel']
        self.mean_mel = torch.unsqueeze(self.mean_mel, 1)
        self.std_mel = norm['std_mel']
        self.std_mel = torch.unsqueeze(self.std_mel, 1)
        
        self.mean_kp = norm['mean_kp']
        self.mean_kp = torch.unsqueeze(self.mean_kp, 0)
        self.std_kp = norm['std_kp']
        self.std_kp = torch.unsqueeze(self.std_kp, 0)
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        filename = self.datalist[idx]
        triplet = torch.load(self.path+filename)
        del triplet['face0'], triplet['faceT'], triplet['sketch0'], triplet['sketchT']
        
        mel = triplet['mel']
        mel -= self.mean_mel; mel /= self.std_mel
        
        key0 = triplet['key0']
        keyT = triplet['keyT']
        
        key0 = key0 - (key0.max(dim=0).values+key0.min(dim=0).values)/2
        key0 = key0 / ((key0.max(dim=0).values-key0.min(dim=0).values)/2)
        keyT = keyT - (keyT.max(dim=0).values+keyT.min(dim=0).values)/2
        keyT = keyT / ((keyT.max(dim=0).values-keyT.min(dim=0).values)/2)
        
        keyT -= self.mean_kp; keyT /= self.std_kp  
        key0 -= self.mean_kp; key0 /= self.std_kp
        
        orienT = keyT.clone().detach()
        orienT[0:27,:] = 0; orienT[31:32,:] = 0; orienT[35:,:] = 0
        # everything but nose is 0
        
        triplet['mel'] = mel[:,0:50]
        triplet['key0'] = torch.flatten(key0)
        triplet['keyT'] = torch.flatten(keyT)
        triplet['orienT'] = torch.flatten(orienT)
        
        print(filename)
        
        return triplet