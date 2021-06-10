from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np

from dataprep_emotion_cross import prep
from networks.emo_classifier_kp import CrossEmbed

batch_size = 32
epochs = 100
log_nth = 10; plot_nth = 500

# device = torch.device('cpu')
device = torch.device('cuda:0')

modelpath = 'models/'
#datapath = '/media/deepan/Backup/thesis/mead/processed/'
datapath = '/storage/user/dasd/mead/processed/'
tr_set = prep(datapath,'train')
ev_set = prep(datapath,'eval')
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=6)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=5)

# writer = SummaryWriter() # comment=name

def train(CE, prTr_emo_model, epoch): # D_ls
    
    CE.train(); prTr_emo_model.train()
    true_loss = []; cross_loss = []; cross_acc = []; true_acc = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
        
    for batch, (mfcc, target_kp, target) in tr_batch:
        mfcc = mfcc.to(device)
        target_kp = target_kp.to(device)
        target = target.to(device)
        
        opEmo.zero_grad()
        lab_emo, _ = prTr_emo_model(mfcc)
        lossEmo = cr_emo(lab_emo,target)
        lossEmo.backward()
        opEmo.step()
        
        opCE.zero_grad()
        lab_emo.detach_().requires_grad_()
        emo_real = CE(target_kp)
        lab = torch.ones(target_kp.shape[0],1).to(device)
        lossCE = cr_cross(emo_real,lab_emo,lab)
        lossCE.backward()
        opCE.step()
        
        pred = torch.argmax(emo_real.detach(),dim=1)
        acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]
        cross_loss.append(lossCE.detach().item())
        cross_acc.append(acc)
        
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr:{epoch+1}, B:{batch+1}, L:{np.mean(tr_loss):.2E}, A:{np.mean(tr_acc):.2E}')
        
        pred = torch.argmax(lab_emo.detach(),dim=1)
        acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]
        aim_acc.append(acc)
    
    print('Target Accuracy : ', np.mean(aim_acc))
    torch.save(CE, modelpath+'bestpreTr_CE.model')
    
def eval(CE, prTr_emo_model, epoch): # D_ls
    
    CE.eval(); prTr_emo_model.eval()
    ev_loss = []; ev_acc = []; aim_acc = []
    ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
        
    for batch, (mfcc, target_kp, target) in ev_batch:
        mfcc = mfcc.to(device)
        target_kp = target_kp.to(device); target = target.to(device)
        
        lab_emo, feat_emo = prTr_emo_model(mfcc)        
        emo_real = CE(target_kp)
        lab = torch.ones(target_kp.shape[0],1).to(device)
        lossCE = cr_cross(emo_real,lab_emo,lab)
        pred = torch.argmax(emo_real.detach(),dim=1)
        acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]        
        ev_loss.append(lossCE.detach().item())
        ev_acc.append(acc)
        if (batch+1)%log_nth == 0:
            ev_batch.set_description(f'Ev:{epoch+1}, B:{batch+1}, L:{np.mean(ev_loss):.2E}, A:{np.mean(ev_acc):.2E}')
        
        pred = torch.argmax(lab_emo.detach(),dim=1)
        acc = (pred == target.detach()).sum().cpu().numpy() / target.shape[0]
        aim_acc.append(acc)

CE = CrossEmbed().to(device)
prTr_emo_model = torch.load('models/bestEv_emo_classifier_seq.model',map_location=device)

cr_cross = nn.CosineEmbeddingLoss()
cr_emo = nn.CrossEntropyLoss()
opCE = optim.Adam(CE.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
opEmo = optim.Adam(prTr_emo_model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)

for epoch in range(epochs):
    train(CE,prTr_emo_model,epoch)
    eval(CE,prTr_emo_model,epoch)#,scheduler)
