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
        
        self.resize = Resize([224,224]) # required by pre trained encoder

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        filename = self.datalist[idx]
        triplet = torch.load(self.path+filename)
        del triplet['mel']
        
        # face
        face0 = triplet['face0'].type(torch.float32)
        faceT = triplet['faceT'].type(torch.float32)
        face0 -= self.mean_face; face0 /= self.std_face; face0 /= 255
        faceT -= self.mean_face; faceT /= self.std_face; faceT /= 255
        face0 = self.resize(face0.permute(-1,0,1))
        faceT = self.resize(faceT.permute(-1,0,1))
        
        # sketch
        sketch0 = torch.unsqueeze(triplet['sketch0'].type(torch.float32), -1)
        sketchT = torch.unsqueeze(triplet['sketchT'].type(torch.float32), -1)
        sketch0 = self.resize(sketch0.permute(-1,0,1))
        sketchT = self.resize(sketchT.permute(-1,0,1))
        
        # temp = triplet['key0'].int().numpy()
        # temp[:,1] = np.clip(temp[:,1],0,face0.shape[0]-1)
        # temp[:,0] = np.clip(temp[:,0],0,face0.shape[1]-1)
        # key0 = torch.zeros((face0.shape[0],face0.shape[1],1))
        # key0[temp[:,1], temp[:,0]] = 1 # create mask
        # temp = triplet['keyT'].int().numpy()
        # temp[:,1] = np.clip(temp[:,1],0,faceT.shape[0]-1)
        # temp[:,0] = np.clip(temp[:,0],0,faceT.shape[1]-1)
        # keyT = torch.zeros((faceT.shape[0],faceT.shape[1],1))
        # keyT[temp[:,1], temp[:,0]] = 1 # create mask
        
        triplet['sketch0'] = sketch0
        triplet['sketchT'] = sketchT
        triplet['face0'] = face0
        triplet['faceT'] = faceT
        
        return triplet
    
    
class gan_prep(Dataset):
    def __init__(self, path):
        
        self.datalist = []
        for person in os.listdir(path):
            for vid in os.listdir(path+'/'+person):
                if len(os.listdir(path+'/'+person+'/'+vid)) == 3:
                    fname = os.listdir(path+'/'+person+'/'+vid)[0]
                    fname = fname.split('.')[0]
                    fname = fname.split('_')[-1]
                    self.datalist.append((path+'/'+person+'/'+vid+'/',fname))
                    
        self.meta = torch.tensor([129.186279296875, 104.76238250732422, 93.59396362304688]) # from vgg
        self.meta = self.meta.unsqueeze(-1).unsqueeze(-1)
                    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        file = self.datalist[idx]
        
        i = torch.randint(low = 1, high = 31, size = (1,2))
        
        face = torch.load(file[0] + 'face_' + file[1] + '.mp4.pt')
        sketch = torch.load(file[0] + 'sketch_' + file[1] + '.mp4.pt')['sketch']
        W_i = torch.load(file[0] + 'W_' + file[1] + '.pt',map_location='cpu').requires_grad_(False)
        
        face_source = (face[i[0,0],:,:,:] - self.meta)/255
        face_source = face_source.type(torch.float)
        face_target = (face[i[0,1],:,:,:] - self.meta)/255
        face_target = face_target.type(torch.float)
        
        sketch_source = sketch[i[0,0],:,:,:]
        sketch_source = sketch_source.type(torch.float)
        sketch_target = sketch[i[0,1],:,:,:]
        sketch_target = sketch_target.type(torch.float)
        
        f_lm = torch.cat((face_source,sketch_source),dim=-3)
        
        return f_lm, face_target, sketch_target, W_i, idx
        
