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
