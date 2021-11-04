import torch
from torch.utils.data import Dataset
import os
import pickle as pkl
import numpy as np
import random
import torch.nn.functional as F

class prep(Dataset):
    
    def __init__(self, path, split):
                
        # mel = torch.load('mel_norm.pt',map_location=(torch.device('cpu')))
        # self.mel_mean = mel['m']
        # self.mel_mean = self.mel_mean.unsqueeze(0).unsqueeze(-1)
        # self.mel_std = mel['s']
        # self.mel_std = self.mel_std.unsqueeze(0).unsqueeze(-1)
        
        # self.kp_init = torch.load('kp_general.pt').flatten().unsqueeze(0)
        self.datalist = pkl.load(open('../datalist_vox2_'+split+'.pkl','rb'))
        
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        fps = 30
        
        path = self.datalist[idx]
        mel = torch.load(path + 'mel.pt')
        mfcc = torch.load(path + 'mfcc.pt')
        kp_seq = torch.load(path + 'kp_seq.pt')
                
        # temp = kp_seq.shape[0]-fps-1
        # if temp < 2: kp_start = 0
        # else: kp_start = np.random.randint(1,temp)
        kp_start = 0
        mfcc_start = int(kp_start * (mfcc.shape[1] / kp_seq.shape[0]))
        mel_start = int(kp_start * (mel.shape[1] / kp_seq.shape[0]))
                
        kp_seq = kp_seq[kp_start:kp_start+fps].float()
        m = kp_seq.mean(dim=(0,1),keepdims=True)
        s = kp_seq.std(dim=(0,1),keepdims=True)
        kp_seq = (kp_seq - m) / s
        kp_seq = kp_seq.flatten(start_dim=1)
        kp_seq -= self.kp_init
                
        mfcc = mfcc[:,mfcc_start:mfcc_start+fps*3+2,:]
        mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
        mfcc = mfcc.permute((2,0,1))
        
        mel = mel[:,mel_start:mel_start+fps*3+2]
        mel = mel.permute((2,0,1))
        mel = (mel - self.mel_mean) / self.mel_std
        
        # negative example of kp for D_lipsync training
        L = list(np.arange(0,len(self.datalist)))
        L.pop(idx)
        neg_idx = random.choice(L)
        
        path = self.datalist[neg_idx]
        neg_kp_seq = torch.load(path + 'kp_seq.pt')[0:fps].float()
        m = neg_kp_seq.mean(dim=(0,1),keepdims=True)
        s = neg_kp_seq.std(dim=(0,1),keepdims=True)
        neg_kp_seq = (neg_kp_seq - m) / s
        neg_kp_seq = neg_kp_seq.flatten(start_dim=1)
        neg_kp_seq -= self.kp_init
        
        if kp_seq.shape[0] < fps: kp_seq = F.pad(kp_seq,(0,0,0,fps-kp_seq.shape[0]))
        if neg_kp_seq.shape[0] < fps: neg_kp_seq = F.pad(neg_kp_seq,(0,0,0,fps-neg_kp_seq.shape[0]))
                
        if mel.shape[2] < fps*3+2: mel = F.pad(mel,(0,fps*3+2-mel.shape[2],0,0,0,0))
        if mfcc.shape[2] < fps*3+2: mfcc = F.pad(mfcc,(0,fps*3+2-mfcc.shape[2],0,0,0,0))
        
        return (mel.float(), mfcc.float(), kp_seq, neg_kp_seq)
