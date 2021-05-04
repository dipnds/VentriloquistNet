import torch.nn as nn
from loss.vgg_face_dag import vgg_face_dag as vgg_face
from loss.vgg19_pt_mcn import vgg19_pt_mcn as vgg_19

class LossCnt(nn.Module):
    def __init__(self):#, device):
        super(LossCnt, self).__init__()
        
        self.VGG19 = vgg_19(weights_path='models/vgg19_pt_mcn.pth')
        for param in self.VGG19.parameters(): param.requires_grad=False
        
        self.VGGFace = vgg_face(weights_path='models/vgg_face_dag.pth')
        for param in self.VGGFace.parameters(): param.requires_grad=False

        self.l1_loss = nn.L1Loss()
        self.conv_idx_list = [2,7,12,21,30] #idxes of conv layers in VGG19 cf.paper

    def forward(self, gt, pred, vgg19_weight=1.5e-1, vggface_weight=2.5e-2):
        # vgg19_weight=1.5e-1   vggface_weight=2.5e-2

        gt_face_f = self.VGGFace(gt)
        pred_face_f = self.VGGFace(pred)

        loss_face = 0
        for a, b in zip(gt_face_f, pred_face_f):
            loss_face += self.l1_loss(a, b)

        gt_19_f = self.VGG19(gt)
        pred_19_f = self.VGG19(pred)

        loss_19 = 0
        for a, b in zip(gt_19_f, pred_19_f):
            loss_19 += self.l1_loss(a, b)

        loss = vgg19_weight * loss_19 + vggface_weight * loss_face

        return loss


class LossAdv(nn.Module):
    def __init__(self, FM_weight=1e1):
        super(LossAdv, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.FM_weight = FM_weight
        
    def forward(self, lab, D_gt_res_list, D_pred_res_list):

        lossFM = 0
        for a, b in zip(D_gt_res_list, D_pred_res_list):
            lossFM += self.l1_loss(a, b)
        
        loss = -lab.mean() + lossFM * self.FM_weight

        return loss

    
class LossG(nn.Module):
    def __init__(self):#, device):
        super(LossG, self).__init__()
        self.lossCnt = LossCnt()#device)
        self.lossAdv = LossAdv()
        
    def forward(self, face_gt, face_pred, lab, D_gt_res_list, D_pred_res_list):
        loss_cnt = self.lossCnt(face_gt, face_pred)
        loss_adv = self.lossAdv(lab, D_gt_res_list, D_pred_res_list)
        return loss_cnt + loss_adv
