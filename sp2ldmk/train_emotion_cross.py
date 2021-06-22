from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np

from dataprep_emotion_cross import prep
from networks.kp2emo import kp2emo_cross

batch_size = 32
epochs = 100
log_nth = 10; plot_nth = 500

# device = torch.device('cpu')
device = torch.device('cuda:0')

modelpath = 'models/'
# datapath = '/media/deepan/Backup/thesis/mead/processed/'
datapath = '/usr/stud/dasd/workspace/mead/processed/'
tr_set = prep(datapath,'train')
ev_set = prep(datapath,'eval')
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=7,drop_last=True)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=True,num_workers=6,drop_last=True)

# writer = SummaryWriter() # comment=name

def train(CE, prTr_emo_model, epoch): # D_ls
    
    CE.train(); prTr_emo_model.train()
    true_loss = []; true_acc = []; cross_loss = []; cross_acc = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
                
    for batch, (mel, target_kp) in tr_batch:
        mel = mel.to(device)
        target_kp = target_kp.to(device)
        
        # opEmo.zero_grad()
        labEmo_sp, _ = prTr_emo_model(mel)
        # lossEmo = cr_emo(lab_emo,target)
        # lossEmo.backward()
        # opEmo.step()
        
        # pred_speech = torch.argmax(lab_emo.detach_(),dim=1)
        # acc = (pred_speech == target.detach()).sum().cpu().numpy() / target.shape[0]
        # true_loss.append(lossEmo.detach().item())
        # true_acc.append(acc)
        
        opCE.zero_grad()
        labEmo_sp = labEmo_sp.clone().detach_().requires_grad_(True)
        labEmo_kp = CE(target_kp)
        
        labEmo_sp = nn.functional.softmax(labEmo_sp,dim=1)
        labEmo_kp = nn.functional.softmax(labEmo_kp,dim=1)
        
        lab = torch.ones(target_kp.shape[0],1).to(device)
        lossCE = cr_cross(labEmo_kp,labEmo_sp,lab)
        
        # if epoch > 40:
        lossCE.backward()
        opCE.step()
        
        target = torch.argmax(labEmo_sp.detach(),dim=1)
        pred = torch.argmax(labEmo_kp.detach(),dim=1)
        acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]
        cross_loss.append(lossCE.detach().item())
        cross_acc.append(acc)
        
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr:{epoch+1}, TL:{np.mean(true_loss):.2E}, '+
                                      f'TA:{np.mean(true_acc):.2E}, CL:{np.mean(cross_loss):.2E}, '
                                       + f'CA:{np.mean(cross_acc):.2E}')
                        
    # if epoch > 40:        
    torch.save(CE, modelpath+'bestTr_CE.model')
    # torch.save(emo_model, modelpath+'bestReTr_emo_classifier_seq.model')
    
def eval(CE, prTr_emo_model, epoch, schedulerCE): # schedulerEmo, D_ls
    
    CE.eval(); prTr_emo_model.eval()
    true_loss = []; true_acc = []; cross_loss = []; cross_acc = []
    ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
        
    with torch.no_grad():
        for batch, (mfcc, target_kp, target) in ev_batch:
            mfcc = mfcc.to(device)
            target_kp = target_kp.to(device)
            target = target.to(device)
            
            lab_emo, _ = prTr_emo_model(mfcc)
            lossEmo = cr_emo(lab_emo,target)
            
            pred_speech = torch.argmax(lab_emo.detach(),dim=1)
            acc = (pred_speech == target.detach()).sum().cpu().numpy() / target.shape[0]
            true_loss.append(lossEmo.detach().item())
            true_acc.append(acc)
            
            emo_real = CE(target_kp)
            lab_emo = nn.functional.softmax(lab_emo,dim=1)
            emo_real = nn.functional.softmax(emo_real,dim=1)
            lab = torch.ones(target_kp.shape[0],1).to(device)
            lossCE = cr_cross(emo_real,lab_emo,lab)
            
            pred = torch.argmax(emo_real.detach(),dim=1)
            acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]
            cross_loss.append(lossCE.detach().item())
            cross_acc.append(acc)
        
            if (batch+1)%log_nth == 0:
                ev_batch.set_description(f'Ev:{epoch+1}, TL:{np.mean(true_loss):.2E}, '+
                                     f'TA:{np.mean(true_acc):.2E}, CL:{np.mean(cross_loss):.2E}, '
                                      + f'CA:{np.mean(cross_acc):.2E}')
                                
    # if epoch > 40:
    schedulerCE.step()
    # schedulerEmo.step()
            
        
CE = kp2emo_cross().to(device)
sp2emo = torch.load('models/bestEv_sp2emo.model',map_location=device)
for param in sp2emo.parameters(): param.requires_grad = False
sp2emo.eval()

cr_cross = nn.CosineEmbeddingLoss()
cr_emo = nn.CrossEntropyLoss()
opCE = optim.Adam(CE.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
# opEmo = optim.Adam(emo_model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8)
schedulerCE = optim.lr_scheduler.StepLR(opCE, step_size=30, gamma=0.1, verbose=True)
# schedulerEmo = optim.lr_scheduler.StepLR(opEmo, step_size=20, gamma=0.5, verbose=True)

for epoch in range(epochs):
    # if epoch == 20:
    #     for param in emo_model.parameters(): param.requires_grad = False
    train(CE,sp2emo,epoch)
    eval(CE,sp2emo,epoch,schedulerCE) # schedulerEmo
