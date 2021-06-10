from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataprep_gan import prep
from networks.gan2 import (Generator, Discriminator_RealFakeSeq,
                        CrossEmbed, LossDSCreal, LossDSCfake, lip_cossim)

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
#ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=5)

# writer = SummaryWriter() # comment=name

def train(G, D_rf, CE, prTr_emo_model, epoch): # D_ls
    
    G.train(); D_rf.train(); CE.train(); prTr_emo_model.eval() # D_ls.train()
    G_loss = []; D_loss = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
        
    for batch, (mel, mfcc, target_kp, negative_kp) in tr_batch:
        mfcc = mfcc.to(device); mel = mel.to(device)
        target_kp = target_kp.to(device); negative_kp = negative_kp.to(device)
        lab_emo, feat_emo = prTr_emo_model(mfcc)
                
        ## REAL 1, FAKE 0
               
        # pred_ce = CE(pred_kp); pred_ce = pred_ce.detach_().requires_grad_()
        # lossG_ce = cr_Lnorm(pred_ce,lab_emo)
        # lossG1 = 0.2*lossG_ce - lossG_ls - lossG_rf
        
        opG.zero_grad()
        pred_kp = G(mel,feat_emo)

        # if (epoch)%2 == 0: wt_dist = 0.9
        # else : wt_dist = 0.5
        wt_dist = 0.75
        # with torch.no_grad():
        lab_rf = D_rf(pred_kp)
        lossG_rf = 1 - lab_rf.mean()
        # print(pred_kp.shape, target_kp.shape)
        lossG_Lnorm, lossG_cossim = cr_lipDist(pred_kp,target_kp)
        lossG_dist = 50*(epoch+1)*lossG_cossim + 5*(epoch+1)*lossG_Lnorm # it was 100 and 10
        loss_G = wt_dist*lossG_dist + (1-wt_dist)*lossG_rf
        loss_G.backward()
        opG.step()
                
        opD_rf.zero_grad()
        pred_kp = pred_kp.clone().detach().requires_grad_()
        pred_real = D_rf(target_kp); lossDreal = crDreal(pred_real)
        pred_fake = D_rf(pred_kp); lossDfake = crDfake(pred_fake)
        lossD_rf = 0.5*lossDreal + 0.5*lossDfake
        lossD_rf.backward()
        opD_rf.step()
        
        if (batch+1)%(5*log_nth) == 0:
            print('pred: ', pred_kp.abs().mean(dim=(1,2)))
            print('gt: ', target_kp.abs().mean(dim=(1,2)))
            print('fake',lab_rf.mean().item(), lossG_cossim.item(), lossG_Lnorm.item())
            print('real',pred_real.mean().item(), lossDreal.item(), lossDfake.item())
        
        # opCE.zero_grad()
        # pred_kp.detach_().requires_grad_()
        # lab_emo.detach_().requires_grad_()
        # emo_real = CE(target_kp); loss_real = cr_Lnorm(emo_real,lab_emo)
        # emo_fake = CE(pred_kp); loss_fake = cr_Lnorm(emo_fake,lab_emo)
        # lossCE = loss_real - loss_fake
        # lossCE.backward()
        # opCE.step()
        
        # G_loss.append(lossG1.detach().item())#+lossG2.detach().item())
        # D_loss.append(lossD_rf.detach().item()+lossD_ls.detach().item())#+lossCE.detach().item())
        G_loss.append(loss_G.detach().item())
        D_loss.append(lossD_rf.detach().item())
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr:{epoch+1}, B:{batch+1}, G:{np.mean(G_loss):.2E}, D:{np.mean(D_loss):.2E}')
        # if (batch+1)%plot_nth == 0:
        #     writer.add_scalar('GLoss/tr', np.mean(G_loss), epoch+batch/len(tr_loader))
        #     writer.add_scalar('DLoss/tr', np.mean(D_loss), epoch+batch/len(tr_loader))
        
    torch.save(G, modelpath+'bestTr_G_cnn.model')
    torch.save(D_rf, modelpath+'bestTr_D_rf_cnn.model')
    # torch.save(CE, modelpath+'bestTr_CE.model')
    
    if (epoch+1)%20 == 0:
        torch.save(G, modelpath+str(epoch+1)+'Tr_G_cnn.model')
        torch.save(D_rf, modelpath+str(epoch+1)+'Tr_D_rf_cnn.model')
            

G = Generator().to(device)
D_rf = Discriminator_RealFakeSeq().to(device)
CE = CrossEmbed().to(device)
prTr_emo_model = torch.load('models/bestEv_emo_classifier_seq.model',map_location=device)
prTr_emo_model.eval()

cr_class = nn.BCELoss()
cr_cossim = nn.CosineEmbeddingLoss(margin=0.1)
cr_fullLnorm = nn.L1Loss()

cr_lipDist = lip_cossim(device)
crDreal = LossDSCreal(); crDfake = LossDSCfake()

opG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8) # lstm 1e-4
opD_rf = optim.Adam(D_rf.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8) # lstm 1e-4
opCE = optim.Adam(CE.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)

for epoch in range(epochs):
    train(G,D_rf,CE,prTr_emo_model,epoch) # D_ls
    # eval(G,D_rf,CE,prTr_emo_model,epoch)#,scheduler)

# TODO : lr scheduler, based on losses of G and D
