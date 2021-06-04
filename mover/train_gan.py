from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np
#import matplotlib.pyplot as plt

from dataprep_gan import prep
from networks.gan import Generator, Discriminator_RealFakeSeq, Discriminator_LipSync, CrossEmbed

batch_size = 16
epochs = 10
log_nth = 100; plot_nth = 500

# device = torch.device('cpu')
device = torch.device('cuda:0')

modelpath = 'models/'
datapath = '/media/deepan/Backup/thesis/mead/processed/'
# datapath = '/storage/user/dasd/mead/processed/'
tr_set = prep(datapath,'train')
ev_set = prep(datapath,'eval')
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=6)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=4)

writer = SummaryWriter() # comment=name

def train(G, D_rf, D_ls, CE, prTr_emo_model, epoch):
    
    G.train(); D_rf.train(); D_ls.train(); CE.train(); prTr_emo_model.eval() 
    G_loss = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
    
    for batch, (mel, mfcc, target_kp, negative_kp) in tr_batch:
        mfcc = mfcc.to(device); mel = mel.to(device)
        target_kp = target_kp.to(device); negative_kp = negative_kp.to(device)
        # lab_emo, feat_emo = prTr_emo_model(mfcc)
        
        ## REAL 1, FAKE 0
        
        # opD_ls.zero_grad()
        # pred_sync = D_ls(mel,target_kp.permute(1,0,2))
        # gt_sync = torch.ones_like(pred_sync).to(device)
        # pred_nosync = D_ls(mel,negative_kp.permute(1,0,2))
        # gt_nosync = torch.zeros_like(pred_nosync).to(device)
        # lossD_ls = crD(torch.cat((pred_sync,pred_nosync)),torch.cat((gt_sync,gt_nosync)))
        # lossD_ls.backward()
        # opD_ls.step()
               
        # opG.zero_grad()
        # pred_lip_kp, pred_kp = G(mel,feat_emo)
        # # with torch.no_grad(): # ?
        # lossG_ls = D_ls(mel,pred_lip_kp).mean(); lossG_rf = D_rf(pred_kp).mean()
        # pred_ce = CE(pred_kp); lossG_ce = crCE(pred_ce,lab_emo)
        # lossG = lossG_ce - lossG_ls - lossG_rf
        # lossG.backward()
        # opG.step()
        
        # opD_rf.zero_grad()
        # pred_kp.detach_().requires_grad_()
        # pred_real = D_rf(target_kp); gt_real = torch.ones_like(pred_real).to(device)
        # pred_fake = D_rf(pred_kp); gt_fake = torch.zeros_like(pred_fake).to(device)
        # lossD_rf = crD(torch.cat((pred_real,pred_fake)),torch.cat((gt_real,gt_fake)))
        # lossD_rf.backward()
        # opD_rf.step()
        
        # opCE.zero_grad()
        # pred_kp.detach_().requires_grad_()
        # lab_emo.detach_().requires_grad_()
        # emo_real = CE(target_kp); loss_real = crCE(emo_real,lab_emo)
        # emo_fake = CE(pred_kp); loss_fake = crCE(emo_fake,lab_emo)
        # lossCE = loss_real - loss_fake
        # lossCE.backward()
        # opCE.step()
        
        # G_loss.append(lossG.detach().item())
        # if (batch+1)%log_nth == 0:
        #     tr_batch.set_description(f'Tr E:{epoch+1}, B:{batch+1}, L:{np.mean(G_loss):.2E}')
        # if (batch+1)%plot_nth == 0:
        #     writer.add_scalar('Loss/tr', np.mean(tr_loss), epoch+batch/len(tr_loader))
        
    # torch.save(G, modelpath+'bestTr_G.model')
    # torch.save(D, modelpath+'bestTr_D.model')

# def eval(G, prTr_emo_model, epoch, best_loss, scheduler):
    
#     G.eval()
#     ev_loss = []
#     with torch.no_grad():
#         ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
        
#         for batch, (mel, mfcc, target) in ev_batch:
#             mfcc = mfcc.to(device); mel = mel.to(device); target = target.to(device)
#             lab_emo, feat_emo = prTr_emo_model(mfcc)
            
#             pred = G(mel,feat_emo)
#             loss = criterion(target,pred)
            
#             ev_loss.append(loss.detach().item())
#             if (batch+1)%(log_nth) == 0:
#                 ev_batch.set_description(f'Ev E:{epoch+1}, B:{batch+1}, L:{np.mean(ev_loss):.2E}')
        
#         loss = np.mean(ev_loss)
#         writer.add_scalar('Loss/ev', loss, epoch)
              
#         if best_loss is None or loss < best_loss:
#             best_loss = loss
#             torch.save
            
#         return best_loss

G = Generator().to(device)
D_rf = Discriminator_RealFakeSeq().to(device)
D_ls = Discriminator_LipSync().to(device)
CE = CrossEmbed().to(device)
prTr_emo_model = torch.load('models/bestEv_emo_classifier_seq.model',map_location=device)
prTr_emo_model.eval()

crD = nn.BCELoss()
crCE = nn.L1Loss()

opG = optim.Adam(G.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8)
opD_rf = optim.Adam(D_rf.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
opD_ls = optim.Adam(D_ls.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8)
opCE = optim.Adam(CE.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)

bestEv_loss = None; bestTr_loss = None
for epoch in range(epochs):
    train(G,D_rf,D_ls,CE,prTr_emo_model,epoch)
    # bestEv_loss = eval(G,prTr_emo_model,epoch,bestEv_loss)#,scheduler)        
        
