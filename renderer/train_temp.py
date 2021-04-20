from torch.utils.data import DataLoader
import torch
import tqdm
from dataprep_temp import prep_old, prep_new

batch_size = 8
epochs = 5
device = torch.device('cuda:0')
datapath = '/media/deepan/Backup/thesis/dataset_voxceleb/triplets/'

ev_set = prep_new(datapath,'eval')
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=True,num_workers=4)        
def eval(epoch):
    tr_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
    with torch.no_grad():
        for batch, (face_source, sketch_source, face_gt, sketch_target) in tr_batch:
            face_source = face_source.to(device); face_gt = face_gt.to(device)
            sketch_source = sketch_source.to(device); sketch_target = sketch_target.to(device)         
        return 0
for epoch in range(epochs):
    bestEv_loss = eval(epoch)     

ev_set = prep_old(datapath,'eval')
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=True,num_workers=4)        
def eval(epoch):
    tr_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
    with torch.no_grad():
        for batch, (face_source, sketch_source, face_gt, sketch_target) in tr_batch:
            face_source = face_source.to(device); face_gt = face_gt.to(device)
            sketch_source = sketch_source.to(device); sketch_target = sketch_target.to(device)         
        return 0
for epoch in range(epochs):
    bestEv_loss = eval(epoch)        
        