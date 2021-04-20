from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np

from dataprep_gan import prep
from networks.gan import Generator, Discriminator
from loss.loss_discriminator import LossDSCreal, LossDSCfake
from loss.loss_generator import LossG

batch_size = 1
epochs = 10
log_nth = 100; plot_nth = 500

device = torch.device('cpu')
# device = torch.device('cuda:0')

modelpath = 'models'
datapath = '../processed'
ev_set = prep(datapath,'eval')
ev_set.datalist = ev_set.datalist[0]
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=7)

G = torch.load('models/lastTr_G.model').to(device)
D = torch.load('models/lastTr_D.model').to(device)

opG = optim.Adam(G.parameters(),lr=5e-5)
opD = optim.Adam(D.parameters(),lr=2e-4)

crG = LossG().to(device)#device=device)
crDreal = LossDSCreal().to(device)
crDfake = LossDSCfake().to(device)


def train(G, D, opG, opD, crG, crDreal, crDfake, epoch):
    
    G.train(); D.train()
    tr_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
    
    for batch, (face_source, sketch_source, face_gt, sketch_target) in tr_batch:
        
        face_source = face_source.to(device); face_gt = face_gt.to(device)
        sketch_source = sketch_source.to(device); sketch_target = sketch_target.to(device)
        
        opG.zero_grad(); opD.zero_grad()
        face_pred, e = G(face_source, sketch_target)
        lab, D_pred_res_list = D(face_pred, sketch_target, e)
        with torch.no_grad():
            _, D_gt_res_list = D(face_gt, sketch_target, e)
        lossG = crG(face_gt, face_pred, lab, D_gt_res_list, D_pred_res_list)
        lossG.backward(retain_graph=False)
        opG.step()
        
        opG.zero_grad(); opD.zero_grad()
        face_pred.detach_().requires_grad_()
        lab_pred, _ = D(face_pred, sketch_target, e)
        lossDfake = crDfake(lab_pred)
        lab_gt, _ = D(face_gt, sketch_target, e)
        lossDreal = crDreal(lab_gt)
        lossD = lossDfake + lossDreal
        lossD.backward(retain_graph=False)
        opD.step()
            
    meta = torch.tensor([131.0912, 103.8827, 91.4953]).unsqueeze(-1).unsqueeze(-1)
    vis_img = torch.zeros(3,sketch_source.shape[2],sketch_source.shape[3])
    vis_img[0,:,:] = vis_img[0,:,:] + sketch_source[0,:,:,:].detach().cpu()
    vis_img[2,:,:] = vis_img[2,:,:] + sketch_target[0,:,:,:].detach().cpu()
    vis_img = torch.cat((vis_img,
                         (face_source[0,:,:,:].detach().cpu() * 255 + meta)/255,
                         (face_gt[0,:,:,:].detach().cpu() * 255 + meta)/255,
                         (face_pred[0,:,:,:].detach().cpu() * 255 + meta)/255),
                         axis=1)
    torch.save('sample.pt', vis_img)
    
    
for epoch in range(epochs):
    train(G,D,opG,opD,crG,crDreal,crDfake,epoch)
