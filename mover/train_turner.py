from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image

from dataprep import prep
import networks.network3 as network

batch_size = 4
epochs = 10
log_nth = 10; plot_nth = 50

device = torch.device('cuda:0')

modelpath = 'models/'
datapath = '/media/deepan/Backup/thesis/mead/processed/'
tr_set = prep(datapath,'train')
ev_set = prep(datapath,'eval')
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=4)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=2)

name = network.__name__.split('.')[1]
writer = SummaryWriter(comment=name)

def train(model, epoch):
    
    model.train()
    tr_loss = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
    
    for batch, (ip, target) in tr_batch:
        ip = ip.to(device); target = target.to(device)
        
        optimizer.zero_grad()
        pred = model(ip)
        loss = criterion(target,pred)
        loss.backward()
        optimizer.step()
        
        tr_loss.append(loss.detach().item())
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr E:{epoch+1}, B:{batch+1}, L:{np.mean(tr_loss):.2E}')
        if (batch+1)%plot_nth == 0:
            writer.add_scalar('Loss/tr', np.mean(tr_loss), epoch+batch/len(tr_loader))
            
        pred = pred[0].view((-1,2)); target = target[0].view((-1,2))
        f = plt.figure()
        f = plt.scatter(-pred[:,0],-pred[:,1],2,'r'); f = plt.scatter(-target[:,0],-target[:,1],2,'b')
        writer.add_figure('KP/tr', f, epoch)
        
    torch.save(model, modelpath+'bestTr_'+name+'.model')

def eval(model, epoch, best_loss, scheduler):
    
    model.eval()
    ev_loss = []
    with torch.no_grad():
        ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
        
        for batch, (ip, target) in ev_batch:
            ip = ip.to(device); target = target.to(device)
            
            pred = model(ip)
            loss = criterion(target,pred)
            
            ev_loss.append(loss.detach().item())
            if (batch+1)%(log_nth) == 0:
                ev_batch.set_description(f'Ev E:{epoch+1}, B:{batch+1}, L:{np.mean(ev_loss):.2E}')
        
        loss = np.mean(ev_loss)
        writer.add_scalar('Loss/ev', loss, epoch)
        
        pred = pred[0].view((-1,2)); target = target[0].view((-1,2))
        f = plt.figure()
        f = plt.scatter(-pred[:,0],-pred[:,1],2,'r'); f = plt.scatter(-target[:,0],-target[:,1],2,'b')
        writer.add_figure('KP/ev', f, epoch)
        
        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(model, modelpath+'bestEv_'+name+'.model')
            
        # scheduler.step()
        return best_loss

model = network.Net().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

bestEv_loss = None; bestTr_loss = None
for epoch in range(epochs):
    train(model,epoch)
    bestEv_loss = eval(model,epoch,bestEv_loss,scheduler)        
        
