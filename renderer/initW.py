import torch
import os

path = '/storage/user/dasd/vox2/dev/processed'

for person in os.listdir(path):
    for vid in os.listdir(path+'/'+person):
        if len(os.listdir(path+'/'+person+'/'+vid)) == 2:
            fname = os.listdir(path+'/'+person+'/'+vid)[0]
            fname = fname.split('.')[0]
            fname = fname.split('_')[-1]
            W = torch.rand(512)
            torch.save(W, path+'/'+person+'/'+vid+'/'+'W_'+fname+'.pt')