import torch.optim as optim
import torch
import matplotlib.pyplot as plt

from networks.gan import Generator, Discriminator
from loss.loss_discriminator import LossDSCreal, LossDSCfake
from loss.loss_generator import LossG

epochs = 50
# vis_nth = 5

device = torch.device('cpu')
# device = torch.device('cuda:0')

modelpath = 'models_bkup/ep70/'
result_path = 'results/'
datapath = '../../../storage/user/vox2/processed/'
face = torch.load(datapath + 'id00012/2DLq_Kkc1r8/face_00017.pt')
sketch = torch.load(datapath + 'id00012/2DLq_Kkc1r8/sketch_00017.pt')

t = 0; T = -2
meta = torch.tensor([131.0912, 103.8827, 91.4953]).unsqueeze(-1).unsqueeze(-1)

face_source = face[t:t+1,:,:,:].type(torch.float); face_source = (face_source - meta) / 255
face_source = torch.cat((face_source,face_source))
sketch_source = sketch['sketch'][t:t+1,:,:,:].type(torch.float)
sketch_source = torch.cat((sketch_source,sketch_source))
face_source = face_source.to(device); sketch_source = sketch_source.to(device)

face_gt = face[T:T+1,:,:,:].type(torch.float); face_gt = (face_gt - meta) / 255
face_gt = torch.cat((face_gt,face_gt))
sketch_target = sketch['sketch'][T:T+1,:,:,:].type(torch.float)
sketch_target = torch.cat((sketch_target,sketch_target))
face_gt = face_gt.to(device); sketch_target = sketch_target.to(device)

G = torch.load(modelpath + 'lastTr_G.model',map_location=device)
D = torch.load(modelpath + 'lastTr_D.model',map_location=device)
opG = optim.Adam(G.parameters(),lr=5e-5); opD = optim.Adam(D.parameters(),lr=2e-4)
crG = LossG().to(device)
crDreal = LossDSCreal().to(device); crDfake = LossDSCfake().to(device)
G.train(); D.train()

for param in G.enc.parameters(): param.requires_grad=True

for epoch in range(epochs):
    
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
    e = e.detach().clone()
    e.requires_grad = True
    lab_pred, _ = D(face_pred, sketch_target, e)
    lossDfake = crDfake(lab_pred)
    lab_gt, _ = D(face_gt, sketch_target, e)
    lossDreal = crDreal(lab_gt)
    lossD = lossDfake + lossDreal
    lossD.backward(retain_graph=False)
    opD.step()
            
    # vis_img = torch.zeros(3,sketch_source.shape[2],sketch_source.shape[3])
    # vis_img[0,:,:] = vis_img[0,:,:] + sketch_source[0,:,:,:].detach().cpu()
    # vis_img[2,:,:] = vis_img[2,:,:] + sketch_target[0,:,:,:].detach().cpu()
    # vis_img = torch.cat((vis_img,
    #                      (face_source[0,:,:,:].detach().cpu() * 255 + meta)/255),
    #                      axis=1)
    # temp = torch.cat((
    #                 (face_gt[0,:,:,:].detach().cpu() * 255 + meta)/255,
    #                 (face_pred[0,:,:,:].detach().cpu() * 255 + meta)/255),
    #                 axis=1)
    # vis_img = torch.cat((vis_img,temp),axis=2)
    # vis_img = vis_img.permute((1,2,0))
    
    # plt.imshow(vis_img); plt.axis('off')
    # plt.savefig(result_path + 'ep' + str(epoch) + '.png', bbox_inches='tight', dpi=200)
    
    if epoch == 0:
        
        source = (face_source[0,:,:,:].detach().cpu() * 255 + meta)/255
        plt.imshow(source.permute((1,2,0))); plt.axis('off')
        plt.savefig(result_path + 'source.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        target = (face_gt[0,:,:,:].detach().cpu() * 255 + meta)/255
        plt.imshow(target.permute((1,2,0))); plt.axis('off')
        plt.savefig(result_path + 'target.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    pred = (face_pred[0,:,:,:].detach().cpu() * 255 + meta)/255
    plt.imshow(pred.permute((1,2,0))); plt.axis('off')
    plt.savefig(result_path + 'ep' + str(epoch) + '.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(epoch)
    