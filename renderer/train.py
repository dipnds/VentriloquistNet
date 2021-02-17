from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np

from dataprep import prep
import networks.network3 as network

batch_size = 16
epochs = 20
log_nth = 50; plot_nth = 100

# device = torch.device('cpu')
device = torch.device('cuda:0')

modelpath = 'models/'
datapath = '/media/deepan/Backup/thesis/dataset_voxceleb/triplets/'
tr_set = prep(datapath,'eval')
ev_set = prep(datapath,'eval')
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=4)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=True,num_workers=4)

name = network.__name__.split('.')[1]
writer = SummaryWriter(comment=name)

def train(model, epoch, best_loss):
    
    model.train()
    tr_loss = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
    
    for batch, triplet in tr_batch:
        sketch0 = triplet['sketch0']; sketchT = triplet['sketchT']
        face0 = triplet['face0']; faceT = triplet['faceT']
        sketch0 = sketch0.to(device); sketchT = sketchT.to(device)
        face0 = face0.to(device); faceT = faceT.to(device)
        
        optimizer.zero_grad()
        faceP0, facePT = model(sketch0,sketchT,face0,faceT)
        loss1 = criterion1(faceT,faceP0) + criterion1(face0,facePT)
        # loss2 = criterion2(faceT,keyT) + criterion2(facePT,key0)
        # loss3 = criterion1(faceT,facePT) + criterion1(face0,faceP0)
        loss = loss1# + loss2 - loss3
        loss.backward()
        optimizer.step()
        
        # print('GT', keyT.mean().cpu().numpy(), keyT.std().cpu().numpy())
        # print('P', keyP.mean().detach().cpu().numpy(), keyP.std().detach().cpu().numpy())
        
        tr_loss.append(loss.detach().item())
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr E:{epoch+1}, B:{batch+1}, L:{np.mean(tr_loss):.2E}')
        if (batch+1)%plot_nth == 0:
            writer.add_scalar('Loss/tr', np.mean(tr_loss), epoch+batch/len(tr_loader))
    
    norm = torch.load('../norm.pt')
    mean_face = torch.unsqueeze(torch.unsqueeze(norm['mean_face'], 1), 2)
    std_face = torch.unsqueeze(torch.unsqueeze(norm['std_face'], 1), 2)
    vis_img = torch.zeros(3,sketch0.shape[2],sketch0.shape[3])
    vis_img[0,:,:] = vis_img[0,:,:] + sketch0[0].detach().cpu()
    vis_img[2,:,:] = vis_img[2,:,:] + sketchT[0].detach().cpu()
    vis_img = torch.cat((vis_img,
                         (face0[0].detach().cpu() * 255*std_face + mean_face)/255,
                         (faceT[0].detach().cpu() * 255*std_face + mean_face)/255,
                         (faceP0[0].detach().cpu() * 255*std_face + mean_face)/255),
                         axis=1)
    writer.add_image('Face/tr', vis_img, epoch)
    
    tr_loss = np.mean(tr_loss)
    if best_loss is None or tr_loss < best_loss:
        best_loss = tr_loss
        torch.save(model, modelpath+'bestTr_'+name+'.model')
        
    # if epoch == epochs:
    #     sample = {'face0':face0, 'faceT':faceT, 'faceP0':faceP0, 'facePT':facePT}
    #     torch.save(sample, 'trsample.pt')

def eval(model, epoch, best_loss, scheduler):
    
    model.eval()
    ev_loss = []
    with torch.no_grad():
        ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
        
        for batch, triplet in ev_batch:
            sketch0 = triplet['sketch0']; sketchT = triplet['sketchT']
            face0 = triplet['face0']; faceT = triplet['faceT']
            sketch0 = sketch0.to(device); sketchT = sketchT.to(device)
            face0 = face0.to(device); faceT = faceT.to(device)
                
            faceP0, facePT = model(sketch0,sketchT,face0,faceT)
            loss1 = criterion1(faceT,faceP0) + criterion1(face0,facePT)
            # loss2 = criterion2(faceT,faceP0) + criterion2(face0,facePT)
            # loss3 = criterion1(faceT,facePT) + criterion1(face0,faceP0)
            loss = loss1# + loss2 - loss3
            
            ev_loss.append(loss.detach().item())
            if (batch+1)%(log_nth/2) == 0:
                ev_batch.set_description(f'Ev E:{epoch+1}, B:{batch+1}, L:{np.mean(ev_loss):.2E}')
        
        loss = np.mean(ev_loss)
        writer.add_scalar('Loss/ev', loss, epoch)
        
        norm = torch.load('../norm.pt')
        mean_face = torch.unsqueeze(torch.unsqueeze(norm['mean_face'], 1), 2)
        std_face = torch.unsqueeze(torch.unsqueeze(norm['std_face'], 1), 2)
        vis_img = torch.zeros(3,sketch0.shape[2],sketch0.shape[3])
        vis_img[0,:,:] = vis_img[0,:,:] + sketch0[0].detach().cpu()
        vis_img[2,:,:] = vis_img[2,:,:] + sketchT[0].detach().cpu()
        vis_img = torch.cat((vis_img,
                             (face0[0].detach().cpu() * 255*std_face + mean_face)/255,
                             (faceT[0].detach().cpu() * 255*std_face + mean_face)/255,
                             (faceP0[0].detach().cpu() * 255*std_face + mean_face)/255),
                            axis=1)
        writer.add_image('Face/ev', vis_img, epoch)
        
        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(model, modelpath+'bestEv_'+name+'.model')
        
        # if epoch == epochs:
        #     sample = {'face0':face0, 'faceT':faceT, 'faceP0':faceP0, 'facePT':facePT}
        #     torch.save(sample, 'evsample.pt')
        
        # scheduler.step()
        return best_loss

model = network.Net().to(device)
criterion1 = nn.MSELoss(reduction='mean')
# criterion1 = nn.MSELoss(reduction='mean')
# criterion2 = network.faceKP(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

bestEv_loss = None; bestTr_loss = None
for epoch in range(epochs):
    bestTr_loss = train(model,epoch,bestTr_loss)
    bestEv_loss = eval(model,epoch,bestEv_loss,scheduler)        
        
