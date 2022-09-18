import torch.nn as nn
import torch

# lrs were 1e-4 and 4e-5
# 1*(20* + 0*) + 0.5* + 0.5*

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight,mean=0,std=0.008)
    if isinstance(m, nn.LSTM):
        nn.init.orthogonal_(m.weight_ih_l0, gain=nn.init.calculate_gain('tanh'))
        nn.init.orthogonal_(m.weight_hh_l0, gain=nn.init.calculate_gain('tanh'))
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu',0.1)*0.5)


class ResBlock(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(ResBlock,self).__init__()
        
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # 80, 92
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # 80, 92
            nn.BatchNorm2d(out_channels)
            )
        if in_channels == out_channels:
            self.id_branch = nn.Identity()
        else:
            self.id_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.nonlin = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        
    def forward(self, mel):
        
        mel_id = mel
        mel_main = self.conv_branch(mel)
        mel_id = self.id_branch(mel_id)
        mel_main += mel_id
        mel_main = self.nonlin(mel_main)
        
        return mel_main

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator,self).__init__()
        
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,5), padding=(1,2)), # 80, 92
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.MaxPool2d((2,1)) # 40, 92
            )
        self.res1_a = ResBlock(32,64)
        self.res1_b = ResBlock(64,64)
        self.pool1 = nn.MaxPool2d((2,1)) # 20, 92
        self.res2_a = ResBlock(64,128)
        self.res2_b = ResBlock(128,128)
        self.pool2 = nn.MaxPool2d((2,1)) # 10, 92
        self.res3_a = ResBlock(128,256)
        self.res3_b = ResBlock(256,256)
        self.pool3 = nn.MaxPool2d((2,1)) # 5, 92
        self.res4_a = ResBlock(256,512)
        self.res4_b = ResBlock(512,512)
        self.pool4 = nn.MaxPool2d((1,3),stride=(1,3),padding=(0,0)) # 5, 30
        
        self.combo_conv = nn.Sequential(
            nn.Conv1d(512*5+128, 512, 3, padding=1), # 512, 30 # 512*5+128
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv1d(512, 136*2, 3, padding=1), # 272, 30
            nn.BatchNorm1d(136*2),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv1d(136*2, 136*2, 3, padding=1), # 272, 30
            nn.BatchNorm1d(136*2),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Conv1d(136*2, 136, 5, padding=2), # 136, 30
            nn.Tanh()
            )
        
        # self.mel_conv.apply(init_weights)
        self.combo_conv.apply(init_weights)
        
    def forward(self, mel, feat_emo):
        
        mel = self.mel_conv(mel)
        mel = self.pool1(self.res1_b(self.res1_a(mel)))
        mel = self.pool2(self.res2_b(self.res2_a(mel)))
        mel = self.pool3(self.res3_b(self.res3_a(mel)))
        mel = self.pool4(self.res4_b(self.res4_a(mel)))
        mel = mel.view(-1, 512*5, 30) # B, F, T
        feat_emo = feat_emo.squeeze(2) # B, F, T
        
        combo_kp = torch.cat((mel,feat_emo),dim=1) # B, F, T
        combo_kp = self.combo_conv(combo_kp)
        combo_kp = combo_kp.permute(0,2,1) # B, T, F
                
        return combo_kp # lip_kp, emo_kp
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    

class Discriminator_RealFakeSeq(nn.Module):
    
    def __init__(self):
        super(Discriminator_RealFakeSeq,self).__init__()
        
        self.seq_lstm_1 = nn.LSTM(136,32,bidirectional=True)
        self.seq_lstm_2 = nn.LSTM(32*2,8,bidirectional=True)
        
        self.seq_fc = nn.Sequential(
            nn.Linear(30*16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid() # implies r_hat used instead of 1+r_hat in lossDfake
            # !!! if changing to Tanh, change lossDfake also
            )
        
        self.seq_fc.apply(init_weights)
        self.seq_lstm_1.apply(init_weights)
        self.seq_lstm_2.apply(init_weights)
        
    def forward(self, kp_seq): # B, T, F
        
        kp_seq = kp_seq.permute(1,0,2) # T, B, F
        kp_seq, _ = self.seq_lstm_1(kp_seq)
        kp_seq, _ = self.seq_lstm_2(kp_seq)
        
        kp_seq = kp_seq.permute(1,0,2)
        kp_seq = kp_seq.reshape(-1, 30*16)
        lab_realfake = self.seq_fc(kp_seq) # check if sigmoid is there in criterion
        
        return lab_realfake
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    
class LossDSCreal(nn.Module):
    def __init__(self):
        super(LossDSCreal, self).__init__()
        self.relu = nn.ReLU()        
    def forward(self, r):
        loss = self.relu(0.9-r)
        return loss.mean()

class LossDSCfake(nn.Module):
    def __init__(self):
        super(LossDSCfake, self).__init__()
        self.relu = nn.ReLU()        
    def forward(self, rhat):
        loss = self.relu(rhat) # !!! 1.0+r_hat not used since D last layer is Sigmoid
        return loss.mean()

class LossGrealfake(nn.Module):
    def __init__(self):
        super(LossGrealfake, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, lab):
        loss = self.relu(1.0-lab)
        # loss = 1 - lab.mean()
        return loss.mean()


class emo_cossim(nn.Module):
    def __init__(self, device):
        super(emo_cossim, self).__init__()
        self.crit = nn.CosineEmbeddingLoss()  
        self.device = device
    def forward(self, pred, gt):
        pred = nn.functional.softmax(pred,dim=1)
        gt = nn.functional.softmax(gt,dim=1)
        lab = torch.ones(gt.shape[0]).to(self.device)
        loss = self.crit(pred,gt,lab)
        return loss


class lip_cossim(nn.Module):
    def __init__(self, device):
        super(lip_cossim, self).__init__()
        
        self.wt_mouth = torch.zeros(136)
        self.wt_mouth[-40:] = 1; self.wt_mouth[:34] = 1 # lips and jaw only
        self.wt_mouth = torch.diag(self.wt_mouth)
        self.wt_mouth.requires_grad = False
        self.wt_mouth = self.wt_mouth.to(device)
        
        self.wt_rest = torch.ones(136)
        self.wt_rest[-40:] = 0; self.wt_rest[:34] = 0 # other than lips and jaw
        self.wt_rest = torch.diag(self.wt_rest)
        self.wt_rest.requires_grad = False
        self.wt_rest = self.wt_rest.to(device)
        
        self.crit_lower1 = nn.L1Loss()
        # self.crit_lower2 = nn.MSELoss()
        self.crit_lower2 = nn.CosineEmbeddingLoss(margin=0.0)
        self.crit_energy = nn.MSELoss()
        self.device = device
        
    def forward(self, pred_kp, target_kp):
        
        # remove temporal variation, keep shape variation
        pred_mouth = torch.matmul(pred_kp,self.wt_mouth)
        target_mouth = torch.matmul(target_kp,self.wt_mouth)
        
        pred_mouth = pred_mouth.reshape((-1,30,68,2))
        pred_mouth = pred_mouth - (pred_mouth.sum(dim=2,keepdim=True))/37
        pred_mouth = pred_mouth.reshape((-1,30,136))
        target_mouth = target_mouth.reshape((-1,30,68,2))
        target_mouth = target_mouth - (target_mouth.sum(dim=2,keepdim=True))/37
        target_mouth = target_mouth.reshape((-1,30,136))
        
        # lab = - torch.ones(target_kp.shape[0],1).to(self.device)
        loss_lower = self.crit_lower1(pred_mouth,target_mouth)#,lab)
        
        pred_mouth = torch.diff(pred_mouth,dim=1)
        target_mouth = torch.diff(target_mouth,dim=1)
        loss_lower += self.crit_lower1(pred_mouth,target_mouth)
        
        # keep temporal variation, remove person's shape
        loss_rest = self.crit_energy(pred_kp.abs().mean(dim=1),target_kp.abs().mean(dim=1))
        
        pred_rest = torch.matmul(pred_kp,self.wt_rest)
        target_rest = torch.matmul(target_kp,self.wt_rest)
        pred_rest = torch.diff(pred_rest,dim=1)
        # pred_rest = pred_rest.abs().mean(dim=1)
        target_rest = torch.diff(target_rest,dim=1)
        # target_rest = target_rest.abs().mean(dim=1)
        loss_rest += self.crit_energy(pred_rest.mean(dim=1).abs(),target_rest.mean(dim=1).abs())
                
        return loss_rest, loss_lower
    