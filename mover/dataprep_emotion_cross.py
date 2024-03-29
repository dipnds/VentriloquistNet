import torch
from torch.utils.data import Dataset
import os
import pickle as pkl
import numpy as np
import random
import torch.nn.functional as F

class prep(Dataset):
    
    def __init__(self, path, split):
        
        # mfcc = pkl.load(open('/media/deepan/Backup/thesis/'+'emo_data.pkl','rb'))['feat_train']
        mfcc = pkl.load(open('/usr/stud/dasd/workspace/'+'emo_data.pkl','rb'))['feat_train']
        self.mfcc_mean = np.mean(mfcc, axis=(0,2))
        self.mfcc_mean = torch.tensor(self.mfcc_mean).float().unsqueeze(-1)
        self.mfcc_std = np.std(mfcc, axis=(0,2))
        self.mfcc_std = torch.tensor(self.mfcc_std).float().unsqueeze(-1)
        
        self.kp_init = torch.load('kp_general.pt').flatten().unsqueeze(0)
        
        if not os.path.isfile('../datalist_mead_'+split+'.pkl'):

            person_list = pkl.load(open('../split_mead.pkl','rb'))[split]
            datalist = []
            
            # dummy subset
            #if split == 'train': person_list = ['M005']
            #else: person_list = ['M003']
            
            for person in person_list:
                for emo in os.listdir(path+person):
                    for level in os.listdir(path+person+'/'+emo):
                        for utter in os.listdir(path+person+'/'+emo+'/'+level):
                            
                            temp = os.listdir(path+person+'/'+emo+'/'+level+'/'+utter)
                            if 'kp_seq.pt' in temp and 'mel.pt' in temp and 'mfcc.pt' in temp:
                                datalist.append(path+person+'/'+emo+'/'+level+'/'+utter+'/')
                            
            self.datalist = datalist
            pkl.dump(datalist,open('../datalist_mead_'+split+'.pkl','wb'))

        else: self.datalist = pkl.load(open('../datalist_mead_'+split+'.pkl','rb'))
        self.map = {'angry':0, 'contempt':1, 'disgusted':2, 'fear':3, 'happy':4, 'neutral':5, 'sad':6, 'surprised':7}
        
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        fps = 30
        
        path = self.datalist[idx]
        mfcc = torch.load(path + 'mfcc.pt')
        kp_seq = torch.load(path + 'kp_seq.pt')
                
        temp = kp_seq.shape[0]-fps-1
        if temp < 2: kp_start = 0
        else: kp_start = np.random.randint(1,temp)
        mfcc_start = int(kp_start * (mfcc.shape[1] / kp_seq.shape[0]))
                
        kp_seq = kp_seq[kp_start:kp_start+fps].float()
        m = kp_seq.mean(dim=(0,1),keepdims=True)
        s = kp_seq.std(dim=(0,1),keepdims=True)
        kp_seq = (kp_seq - m) / s
        kp_seq = kp_seq.flatten(start_dim=1)
        kp_seq -= self.kp_init
        
        mfcc = mfcc[:,mfcc_start:mfcc_start+fps*3+2,:]
        mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
        mfcc = mfcc.permute((2,0,1))
        
        # snip = path.split('/')[-4]
        # if snip != 'contempt':
        #     gt = self.map[snip]
        # else:
        #     if path.split('/')[-3] != 'level_3':
        #         gt = 4
        #     else:
        #         gt = 1
        gt = self.map[path.split('/')[-4]]
        
        if kp_seq.shape[0] < fps: kp_seq = F.pad(kp_seq,(0,0,0,fps-kp_seq.shape[0]))
        if mfcc.shape[2] < fps*3+2: mfcc = F.pad(mfcc,(0,fps*3+2-mfcc.shape[2],0,0,0,0))
        
        return (mfcc.float(), kp_seq, gt)
