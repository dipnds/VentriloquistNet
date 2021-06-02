import torch
from torch.utils.data import Dataset
import pickle as pkl
import numpy as np

class prep(Dataset):
    
    def __init__(self, path, split):
        
        datadict = pkl.load(open(path+'emo_data.pkl','rb'))
        if split == 'eval':
            self.feat_list = datadict['feat_test']
            self.lab_list = datadict['lab_test']
        else:
            self.feat_list = datadict['feat_train']
            self.lab_list = datadict['lab_train']
        
        self.lab_list = self.lab_list[:,:7] + self.lab_list[:,7:]
        self.lab_list = np.argmax(self.lab_list,axis=1)
        
        self.mean = np.mean(datadict['feat_train'], axis=(0,2))
        self.mean = torch.tensor(self.mean).float().unsqueeze(-1)
        self.std = np.std(datadict['feat_train'], axis=(0,2))
        self.std = torch.tensor(self.std).float().unsqueeze(-1)
                
    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, idx):
        
        start = np.random.randint(0,216-90)
        ip = torch.tensor(self.feat_list[idx,:,start:start+90,:]).float()
        ip = (ip - self.mean) / self.std
        ip = ip.permute((2,0,1))
        target = torch.tensor(self.lab_list[idx])
        
        return (ip, target)
