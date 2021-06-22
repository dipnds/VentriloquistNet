from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np

from dataprep_emotion import prep
import networks.sp2emo as network

batch_size = 16
epochs = 20
log_nth = 100; plot_nth = 500

device = torch.device('cuda:0')

modelpath = 'models/'
# datapath = '/media/deepan/Backup/thesis/'
datapath = '/usr/stud/dasd/workspace/'
tr_set = prep(datapath,'train')
ev_set = prep(datapath,'eval')
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=3,drop_last=True)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=3,drop_last=True)

name = network.__name__.split('.')[1]
writer = SummaryWriter(comment=name)

def train(model, epoch):
    
    model.train()
    tr_loss = []; tr_acc = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
    
    for batch, (ip, target) in tr_batch:
        ip = ip.to(device); target = target.to(device)

        optimizer.zero_grad()
        pred,_ = model(ip)
                
        loss = criterion(pred,target)
        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(pred.detach(),dim=1)
        acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]
        
        tr_acc.append(acc)
        tr_loss.append(loss.detach().item())
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr E:{epoch+1}, B:{batch+1}, L:{np.mean(tr_loss):.2E}, A:{np.mean(tr_acc):.2E}')
        if (batch+1)%plot_nth == 0:
            writer.add_scalar('Loss/tr', np.mean(tr_loss), epoch+batch/len(tr_loader))
        
    torch.save(model, modelpath+'bestTr_'+name+'.model')

def eval(model, epoch, best_loss, scheduler):
    
    model.eval()
    ev_loss = []; ev_acc = []
    with torch.no_grad():
        ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
        
        for batch, (ip, target) in ev_batch:
            ip = ip.to(device); target = target.to(device)
            
            pred,_ = model(ip)
            loss = criterion(pred,target)
            
            pred = torch.argmax(pred.detach(),dim=1)
            acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]
        
            ev_acc.append(acc)
            ev_loss.append(loss.detach().item())
            if (batch+1)%(log_nth) == 0:
                ev_batch.set_description(f'Ev E:{epoch+1}, B:{batch+1}, L:{np.mean(ev_loss):.2E}, A:{np.mean(ev_acc):.2E}')
        
        loss = np.mean(ev_loss)
        writer.add_scalar('Loss/ev', loss, epoch)
        
        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(model, modelpath+'bestEv_'+name+'.model')
            
        scheduler.step()
        return best_loss

model = network.sp2emo().to(device)
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)# 10, 0.1

bestEv_loss = None; bestTr_loss = None
for epoch in range(epochs):
    train(model,epoch)
    bestEv_loss = eval(model,epoch,bestEv_loss,scheduler)        
