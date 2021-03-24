import torch
import torch.nn as nn
import imp
import torchvision
from torchvision.models import vgg19
from networks.vgg_face_dag import vgg_face_dag as vgg_face


class LossCnt(nn.Module):
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossCnt, self).__init__()
        
        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        self.VGG19.to(device)
        
        
        # MainModel = imp.load_source('MainModel', VGGFace_body_path)
        # full_VGGFace = torch.load(VGGFace_weight_path, map_location = 'cpu')
        # cropped_VGGFace = Cropped_VGG19()
        # cropped_VGGFace.load_state_dict(full_VGGFace.state_dict(), strict = False)
        # self.VGGFace = cropped_VGGFace
        self.VGGFace = vgg_face(weights_path='models/vgg_face_dag.pth')
        self.VGGFace.eval()
        self.VGGFace.to(device)

        self.l1_loss = nn.L1Loss()
        self.conv_idx_list = [2,7,12,21,30] #idxes of conv layers in VGG19 cf.paper

    def forward(self, x, x_hat, vgg19_weight=1.5e-1, vggface_weight=2.5e-2):        
        """Retrieve vggface feature maps"""
        with torch.no_grad(): #no need for gradient compute
            vgg_x_features = self.VGGFace(x) #returns a list of feature maps at desired layers

        with torch.autograd.enable_grad():
            vgg_xhat_features = self.VGGFace(x_hat)

        lossface = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            lossface += self.l1_loss(x_feat, xhat_feat)


        """Retrieve vggface feature maps"""
        #define hook
        def vgg_x_hook(module, input, output):
            output.detach_() #no gradient compute
            vgg_x_features.append(output)
        def vgg_xhat_hook(module, input, output):
            vgg_xhat_features.append(output)
            
        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []
        
        conv_idx_iter = 0
        
        
        #place hooks
        for i,m in enumerate(self.VGG19.features.modules()):
            if i == self.conv_idx_list[conv_idx_iter]:
                if conv_idx_iter < len(self.conv_idx_list)-1:
                    conv_idx_iter += 1
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        #run model for x
        with torch.no_grad():
            self.VGG19(x)

        #retrieve features for x
        for h in vgg_x_handles:
            h.remove()

        #retrieve features for x_hat
        #conv_idx_iter = 0
        #for i,m in enumerate(self.VGG19.modules()):
        #    if i <= 30: #30 is last conv layer
        #        if type(m) is not torch.nn.Sequential and type(m) is not torchvision.models.vgg.VGG:
        #        #only pass through nn.module layers
        #            if i == self.conv_idx_list[conv_idx_iter]:
        #                if conv_idx_iter < len(self.conv_idx_list)-1:
        #                    conv_idx_iter += 1
        #                x_hat = m(x_hat)
        #                vgg_xhat_features.append(x_hat)
        #                x_hat.detach_() #reset gradient from output of conv layer
        #            else:
        #                x_hat = m(x_hat)
                        
                        
        vgg_xhat_handles = []
        conv_idx_iter = 0
        
        #place hooks
        with torch.autograd.enable_grad():
            for i,m in enumerate(self.VGG19.features.modules()):
                if i == self.conv_idx_list[conv_idx_iter]:
                    if conv_idx_iter < len(self.conv_idx_list)-1:
                        conv_idx_iter += 1
                    vgg_xhat_handles.append(m.register_forward_hook(vgg_xhat_hook))
            self.VGG19(x_hat)
        
            #retrieve features for x
            for h in vgg_xhat_handles:
                h.remove()
        
        loss19 = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            loss19 += self.l1_loss(x_feat, xhat_feat)

        loss = vgg19_weight * loss19 + vggface_weight * lossface

        return loss


class LossAdv(nn.Module):
    def __init__(self, FM_weight=1e1):
        super(LossAdv, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.FM_weight = FM_weight
        
    def forward(self, r_hat, D_res_list, D_hat_res_list):
        lossFM = 0
        for res, res_hat in zip(D_res_list, D_hat_res_list):
            lossFM += self.l1_loss(res, res_hat)
        
        return -r_hat.mean() + lossFM * self.FM_weight


class LossMatch(nn.Module):
    def __init__(self, device, match_weight=1e1):
        super(LossMatch, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.match_weight = match_weight
        self.device = device
        
    def forward(self, e_vectors, W, i):
        # loss = torch.zeros(e_vectors.shape[0],1).to(self.device)
        # for b in range(e_vectors.shape[0]):
        #     for k in range(e_vectors.shape[1]):
        #         loss[b] += torch.abs(e_vectors[b,k].squeeze() - W[:,b]).mean()
        #     loss[b] = loss[b]/e_vectors.shape[1]
        # loss = loss.mean()
        
        # W = W.unsqueeze(-1).expand(512, W.shape[1], e_vectors.shape[1]).transpose(0,1).transpose(1,2)
        #B,8,512
        # W = W.reshape(-1,512)
        W = W.transpose(0,1)
        #B*8,512
        e_vectors = e_vectors.squeeze(-1)
        #B,8,512
        e_vectors = e_vectors.reshape(-1,512)
        #B*8,512
        return self.l1_loss(e_vectors, W) * self.match_weight
    
class LossG(nn.Module):
    """
    Loss for generator meta training
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossG, self).__init__()
        
        self.lossCnt = LossCnt(VGGFace_body_path, VGGFace_weight_path, device)
        self.lossAdv = LossAdv()
        self.lossMatch = LossMatch(device=device)
        
    def forward(self, x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, W, i):
        loss_cnt = self.lossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        loss_match = self.lossMatch(e_vectors, W, i)
        #print(loss_cnt.item(), loss_adv.item(), loss_match.item())
        return loss_cnt + loss_adv + loss_match
