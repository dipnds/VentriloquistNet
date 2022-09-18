import torch.nn as nn
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight,mean=0,std=0.008)
        # nn.init.constant_(m.bias,0.001)
    if isinstance(m, nn.LSTM):
        nn.init.orthogonal_(m.weight_ih_l0, gain=nn.init.calculate_gain('tanh'))
        nn.init.orthogonal_(m.weight_hh_l0, gain=nn.init.calculate_gain('tanh'))
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh')*0.2)
    #     nn.init.constant_(m.bias, -0.1) # cancelled by batchnorm


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator,self).__init__()
        
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,5), padding=(1,2)), # 80, 92
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.AvgPool2d((2,1)), # 40, 92
            nn.Conv2d(32, 64, 3, padding=1), # 40, 92
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.AvgPool2d((2,1)), # 20, 92
            nn.Conv2d(64, 128, 3, padding=(1,0)), # 20, 90
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.AvgPool2d((2,1)), # 10, 90
            nn.Conv2d(128, 256, 3, padding=1), # 10, 90
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.AvgPool2d((1,3)), # 10, 30
            nn.Conv2d(256, 512, 3, padding=1), # 5, 30
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.AvgPool2d((2,1)), # 10, 30
            nn.Conv2d(512, 512, 3, padding=1), # 5, 30
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1,inplace=True)
            )
        
        self.mel_lstm_1 = nn.LSTM(512*5,256,bidirectional=True)
        self.mel_lstm_2 = nn.LSTM(256*2,136,bidirectional=True)
        
        self.mel_fc = nn.Sequential(
            nn.Linear(136*2, 136*2),
            nn.BatchNorm1d(136*2),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            # nn.Tanh(),
            nn.Linear(136*2, 136),
            nn.Tanh()
            )
        
        # self.mfcc_lstm_1 = nn.LSTM(128,136,bidirectional=True)
        # self.mfcc_lstm_2 = nn.LSTM(136*2,136,bidirectional=False)
        
        # self.common1 = nn.LSTM(136,136,bidirectional=True)
        # self.common2 = nn.LSTM(136*2,136,bidirectional=True)
        # self.common3 = nn.LSTM(136*2,136,bidirectional=False)
        
        self.mel_lstm_1.apply(init_weights)
        self.mel_lstm_2.apply(init_weights)
        self.mel_fc.apply(init_weights)
    
    def forward(self, mel, feat_emo):
        
        mel = self.mel_conv(mel)
        mel = mel.view(-1, 512*5, 30) # B, F, T
        mel = mel.permute(2,0,1) # T, B, F
        
        lip_kp, _ = self.mel_lstm_1(mel)
        lip_kp, _ = self.mel_lstm_2(lip_kp)
        
        lip_kp = lip_kp.reshape((-1,136*2))
        lip_kp = self.mel_fc(lip_kp)
        lip_kp = lip_kp.reshape((30,-1,136))
        
        # feat_emo = feat_emo.view(-1, 128, 30)
        # feat_emo = feat_emo.permute(2,0,1)
        
        # expression_residue, _ = self.mfcc_lstm_1(feat_emo)
        # expression_residue, _ = self.mfcc_lstm_2(expression_residue)
        
        # full_kp = lip_kp + expression_residue
        
        # full_kp, _ = self.common1(full_kp)
        # full_kp, _ = self.common2(full_kp)
        # full_kp, _ = self.common3(full_kp)
        
        # make batch first dim again
        lip_kp = lip_kp.permute(1,0,2) # B, T, F
        
        return lip_kp#, full_kp
    
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
    

class CrossEmbed(nn.Module):
    
    def __init__(self):
        super(CrossEmbed,self).__init__()
        
        self.cross_lstm_1 = nn.LSTM(136,32,bidirectional=True)
        self.cross_lstm_2 = nn.LSTM(32*2,16,bidirectional=True)
        self.cross_lstm_3 = nn.LSTM(16*2,8,bidirectional=True)
        
        self.cross_fc = nn.Sequential(
            nn.Linear(30*16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 7)
            )
            
    def forward(self, kp_seq):
        
        kp_seq, _ = self.cross_lstm_1(kp_seq)
        kp_seq, _ = self.cross_lstm_2(kp_seq)
        kp_seq, _ = self.cross_lstm_3(kp_seq)
        
        kp_seq = kp_seq.permute(1,2,0)
        kp_seq = kp_seq.reshape(-1, 30*16)
        lab_emo = self.cross_fc(kp_seq)
        
        return lab_emo
    
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

class lip_cossim(nn.Module):
    def __init__(self, device):
        super(lip_cossim, self).__init__()
        
        self.wt_mouth = torch.zeros(136)
        self.wt_mouth[-40:] = 1; self.wt_mouth[:34] = 1 # lips and jaw only
        self.wt_mouth = torch.diag(self.wt_mouth)
        self.wt_mouth.requires_grad = False
        self.wt_mouth = self.wt_mouth.to(device)
        
        # self.wt_rest = torch.ones(136)
        # self.wt_rest[-40:] = 0; self.wt_rest[:34] = 0 # lips and jaw only
        # self.wt_rest = torch.diag(self.wt_rest)
        # self.wt_rest.requires_grad = False
        # self.wt_rest = self.wt_rest.to(device)
        
        # self.crit_Lnorm = nn.L1Loss()
        self.crit_Lnorm = nn.MSELoss()
        self.crit_cossim = nn.CosineEmbeddingLoss(margin=0.0)
        self.device = device
    
    def forward(self, pred_kp, target_kp):
        
        # !!! remove time varying mean first
                
        pred_mouth = torch.matmul(pred_kp,self.wt_mouth)
        target_mouth = torch.matmul(target_kp,self.wt_mouth)
        lab = - torch.ones(target_kp.shape[0],1).to(self.device)
        loss_cossim = self.crit_cossim(pred_mouth,target_mouth,lab)
        
        pred_kp = pred_kp.reshape((-1,30,68,2))
        pred_kp = pred_kp - pred_kp.mean(dim=2,keepdim=True)
        pred_kp = pred_kp.reshape((-1,30,136))
        target_kp = target_kp.reshape((-1,30,68,2))
        target_kp = target_kp - target_kp.mean(dim=2,keepdim=True)
        target_kp = target_kp.reshape((-1,30,136))
        
        # pred_rest = torch.matmul(pred_kp,self.wt_rest)
        # target_rest = torch.matmul(target_kp,self.wt_rest)
        # dist = (pred_rest - target_rest).mean(dim=-1,keepdim=True)
        # pred_kp = pred_kp - dist
        loss_Lnorm = self.crit_Lnorm(pred_kp,target_kp)
                
        return loss_Lnorm, loss_cossim
    