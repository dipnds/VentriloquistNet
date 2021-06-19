from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os, tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataprep_gan import prep
from networks.gan_cnn_emo_2 import (Generator, Discriminator_RealFakeSeq,
                        LossGrealfake, LossDSCreal, LossDSCfake,
                        lip_cossim, emo_cossim)

batch_size = 32
epochs = 100
log_nth = 10; plot_nth = 500

# device = torch.device('cpu')
device = torch.device('cuda:0')

modelpath = 'models/'
#datapath = '/media/deepan/Backup/thesis/mead/processed/'
datapath = '/usr/stud/dasd/workspace/mead/processed/'
tr_set = prep(datapath,'train')
ev_set = prep(datapath,'eval')
tr_loader = DataLoader(tr_set,batch_size=batch_size,shuffle=True,num_workers=6)
#ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=3)

# writer = SummaryWriter() # comment=name

def train(G, D_rf, prTr_emo_model, epoch): #, prTr_CE, D_ls
    
    G.train(); D_rf.train(); prTr_emo_model.eval(); prTr_CE.train()#; D_ls.train()
    G_loss = []; D_loss = []
    tr_batch = tqdm.tqdm(enumerate(tr_loader),total=len(tr_loader))
        
    for batch, (mel, mfcc, target_kp, _) in tr_batch:
        mfcc = mfcc.to(device); mel = mel.to(device)
        target_kp = target_kp.to(device)#; negative_kp = negative_kp.to(device)
        
        with torch.no_grad():
            lab_emo, feat_emo = prTr_emo_model(mfcc)
        # print(prTr_emo_model.weight.grad)
                
        ## REAL 1, FAKE 0
        
        opG.zero_grad()
        pred_kp = G(mel,feat_emo)

        # if (epoch)%2 == 0: wt_dist = 0.9
        # else : wt_dist = 0.5
        wt_dist = 1
        lab_rf = D_rf(pred_kp)
        lossG_rf = crG_rf(lab_rf)
        lossG_rest, lossG_lower = cr_lipDist(pred_kp,target_kp)
        # lossG_dist = 50*(epoch+1)*lossG_lower + 5*(epoch+1)*lossG_rest # it was 100 and 10
        lossG_dist = 20*lossG_lower + 20*lossG_rest
        
        emo_fake = prTr_CE(pred_kp)
        lossG_ce = cr_emo(emo_fake,lab_emo)
        # lossG_ce = cr_emo(pred_kp,lab_emo)
        
        loss_G = wt_dist*lossG_dist + ((2-wt_dist)/2)*lossG_rf + ((2-wt_dist)/2)*lossG_ce
        loss_G.backward()
        opG.step()
                
        opD_rf.zero_grad()
        pred_kp = pred_kp.clone().detach().requires_grad_()
        pred_real = D_rf(target_kp); lossDreal = crDreal(pred_real)
        pred_fake = D_rf(pred_kp); lossDfake = crDfake(pred_fake)
        lossD_rf = 0.5*lossDreal + 0.5*lossDfake
        lossD_rf.backward()
        opD_rf.step()
                
        if (batch)%(20*log_nth) == 0:
            print('pred: ', pred_kp.abs().mean(dim=(1,2)))
            print('gt: ', target_kp.abs().mean(dim=(1,2)))
            print('fake',lab_rf.mean().item(), lossG_lower.item(), lossG_rest.item())
            print('real',pred_real.mean().item(), lossDreal.item(), lossDfake.item())
            print('emo', lossG_ce.item())
        
        G_loss.append(loss_G.detach().item())
        D_loss.append(lossD_rf.detach().item())
        if (batch+1)%log_nth == 0:
            tr_batch.set_description(f'Tr:{epoch+1}, B:{batch+1}, G:{np.mean(G_loss):.2E},'+
                                      f' D:{np.mean(D_loss):.2E}')
        # if (batch+1)%plot_nth == 0:
        #     writer.add_scalar('GLoss/tr', np.mean(G_loss), epoch+batch/len(tr_loader))
        #     writer.add_scalar('DLoss/tr', np.mean(D_loss), epoch+batch/len(tr_loader))
        
    torch.save(G, modelpath+'bestTr_G.model')
    torch.save(D_rf, modelpath+'bestTr_D_rf.model')
    
    if (epoch+1)%20 == 0:
        torch.save(G, modelpath+str(epoch+1)+'Tr_G.model')
        torch.save(D_rf, modelpath+str(epoch+1)+'Tr_D_rf.model')
    

G = Generator().to(device)
D_rf = Discriminator_RealFakeSeq().to(device)
prTr_CE = torch.load('models/bestpreTr_CE.model',map_location=device)
for param in prTr_CE.parameters(): param.requires_grad = False
prTr_CE.eval()
prTr_emo_model = torch.load('models/bestReTr_emo_classifier_seq.model',map_location=device)
for param in prTr_emo_model.parameters(): param.requires_grad = False
prTr_emo_model.eval()

cr_lipDist = lip_cossim(device)
crG_rf = LossGrealfake()
crDreal = LossDSCreal(); crDfake = LossDSCfake()
cr_emo = emo_cossim(device)#,prTr_CE)

opG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8) # lstm 1e-4
opD_rf = optim.Adam(D_rf.parameters(), lr=4e-5, betas=(0.9,0.999), eps=1e-8) # lstm 4e-5

for epoch in range(epochs):
    train(G,D_rf,prTr_emo_model,epoch) # ,prTr_CE,D_ls
    # eval(G,D_rf,prTr_CE,prTr_emo_model,epoch)#,scheduler)

# TODO : lr scheduler, based on losses of G and D
