from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np

from dataprep import prep
import networks.network3 as network

batch_size = 64
epochs = 225
log_nth = 10; plot_nth = 50
del_mode = False
kp_mode='norm'

device = torch.device('cuda:0')

modelpath = 'models/'
datapath = '/media/deepan/Backup/thesis/dataset_voxceleb/triplets/'
tr_set = prep(datapath,'train', del_mode=del_mode, kp_mode=kp_mode)
ev_set = prep(datapath,'eval', del_mode=del_mode, kp_mode=kp_mode)
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=4)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=2)

name = network.__name__.split('.')[1]
writer = SummaryWriter(comment=name)

def train(model, epoch, best_loss):
    
    model.train()
    tr_loss = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
    
    for batch, triplet in tr_batch:
        key0 = triplet['key0']; keyT = triplet['keyT']; mel = triplet['mel']; orienT = triplet['orienT']
        key0 = key0.to(device); keyT = keyT.to(device); mel = mel.to(device); orienT = orienT.to(device)
        
        optimizer.zero_grad()
        keyP = model(key0,mel,orienT)
        loss = criterion(keyT,keyP); loss.backward()
        optimizer.step()
        
        # print('GT', keyT.mean().cpu().numpy(), keyT.std().cpu().numpy())
        # print('P', keyP.mean().detach().cpu().numpy(), keyP.std().detach().cpu().numpy())
        
        tr_loss.append(loss.detach().item())
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr E:{epoch+1}, B:{batch+1}, L:{np.mean(tr_loss):.2E}')
        if (batch+1)%plot_nth == 0:
            writer.add_scalar('Loss/tr', np.mean(tr_loss), epoch+batch/len(tr_loader))
    
    tr_loss = np.mean(tr_loss)
    if best_loss is None or tr_loss < best_loss:
        best_loss = tr_loss
        torch.save(model, modelpath+'bestTr_'+name+'.model')

def eval(model, epoch, best_loss, scheduler):
    
    model.eval()
    ev_loss = []
    with torch.no_grad():
        ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
        
        for batch, triplet in ev_batch:
            key0 = triplet['key0']; keyT = triplet['keyT']; mel = triplet['mel']; orienT = triplet['orienT']
            key0 = key0.to(device); keyT = keyT.to(device); mel = mel.to(device); orienT = orienT.to(device)
            
            keyP = model(key0,mel,orienT)
            loss = criterion(keyT,keyP)
            
            ev_loss.append(loss.detach().item())
            if (batch+1)%(log_nth/2) == 0:
                ev_batch.set_description(f'Ev E:{epoch+1}, B:{batch+1}, L:{np.mean(ev_loss):.2E}')
        
        loss = np.mean(ev_loss)
        writer.add_scalar('Loss/ev', loss, epoch)
        
        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(model, modelpath+'bestEv_'+name+'.model')
            
        scheduler.step()
        return best_loss

model = network.Net().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

bestEv_loss = None; bestTr_loss = None
for epoch in range(epochs):
    bestTr_loss = train(model,epoch,bestTr_loss)
    bestEv_loss = eval(model,epoch,bestEv_loss,scheduler)        
        
