import torch
from torch.utils.data import Dataset
import os
import pickle as pkl

class prep_old(Dataset):
    def __init__(self, path, split):
        
        # person_list = pkl.load(open('../split.pkl','rb'))[split]
        
        # debug dataset
        path = '../processed'
        person_list = ['id00012','id00015','id00016']
        
        self.datalist = []
        for person in person_list:
            try:
                for vid in os.listdir(path+'/'+person):
                    if len(os.listdir(path+'/'+person+'/'+vid)) > 0:
                        fname = os.listdir(path+'/'+person+'/'+vid)
                        fname.sort()
                        fname = fname[0]
                        fname = fname.split('_')[-1]
                        self.datalist.append((path+'/'+person+'/'+vid+'/',fname))
            except: pass
        
        self.meta = torch.tensor([131.0912, 103.8827, 91.4953]).unsqueeze(-1).unsqueeze(-1)
                                
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        file = self.datalist[idx]
        face = torch.load(file[0] + 'face_' + file[1])
        sketch = torch.load(file[0] + 'sketch_' + file[1])['sketch']        
        
        i = torch.randint(low = 0, high = 8, size = (1,1))[0][0]
        j = torch.randint(low = 8, high = face.shape[0], size = (1,1))[0][0]
                
        face_source = face[i,:,:,:].type(torch.float)
        face_source = (face_source - self.meta) / 255
        sketch_source = sketch[i,:,:,:].type(torch.float)
        
        face_target = face[j,:,:,:].type(torch.float)
        face_target = (face_target - self.meta) / 255
        sketch_target = sketch[j,:,:,:].type(torch.float)
        
        return face_source, sketch_source, face_target, sketch_target

class prep_new(Dataset):
    def __init__(self, path, split):
        
        # person_list = pkl.load(open('../split.pkl','rb'))[split]
        
        # debug dataset
        path = '../processed'
        person_list = ['id00012','id00015','id00016']
        
        self.datalist = []
        for person in person_list:
            try:
                for vid in os.listdir(path+'/'+person):
                    l = len(os.listdir(path+'/'+person+'/'+vid))
                    if l > 0:
                        self.datalist.append((path+'/'+person+'/'+vid+'/',l-3))
            except: pass
        
        self.meta = torch.tensor([131.0912, 103.8827, 91.4953]).unsqueeze(-1).unsqueeze(-1)
                                
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        file = self.datalist[idx]
        l = file[1]
        
        i = int(torch.randint(low = 0, high = 8, size = (1,1))[0][0])
        j = int(torch.randint(low = 8, high = l, size = (1,1))[0][0])
        
        fatch0 = torch.load(file[0] + 'fatch_' + str(i) + '.pt')
        fatchT = torch.load(file[0] + 'fatch_' + str(j) + '.pt')
                
        face_source = fatch0[0:3,:,:].type(torch.float)
        face_source = (face_source - self.meta) / 255
        sketch_source = fatch0[-1:,:,:].type(torch.float)
        
        face_target = fatchT[0:3,:,:].type(torch.float)
        face_target = (face_target - self.meta) / 255
        sketch_target = fatchT[-1:,:,:].type(torch.float)
        
        return face_source, sketch_source, face_target, sketch_target
