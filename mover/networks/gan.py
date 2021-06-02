import torch.nn as nn
import torch

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator,self).__init__()
        
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,5), padding=(1,2)), # 80, 80
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)), # 40, 40
            nn.Conv2d(32, 64, (3,5), padding=(1,0)), # 40, 36
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), # 20, 36
            nn.Conv2d(64, 128, 3, padding=(1,0)), # 20, 34
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), # 10, 34
            nn.Conv2d(128, 256, 3, padding=(1,0)), # 10, 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), # 5, 32
            nn.Conv2d(256, 512, 3, padding=(1,0)), # 5, 30
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )
        
        self.mel_lstm_1 = nn.LSTM(512*5,256,bidirectional=True)
        self.mel_lstm_2 = nn.LSTM(256*2,136,bidirectional=True)
        self.mel_lstm_3 = nn.LSTM(136*2,136,bidirectional=False)
        
        self.mfcc_lstm_1 = nn.LSTM(128,136,bidirectional=True)
        self.mfcc_lstm_2 = nn.LSTM(136*2,136,bidirectional=False)
        
        # self.common = nn.LSTM()
    
    def forward(self, mel, feat_emo):
        
        mel = self.mel_conv(mel)
        mel = mel.view(-1, 512*5, 30) # B, F, T
        mel = mel.permute(2,0,1) # T, B, F
        
        lip_kp, _ = self.mel_lstm_1(mel)
        lip_kp, _ = self.mel_lstm_2(lip_kp)
        lip_kp, _ = self.mel_lstm_3(lip_kp)
        
        feat_emo = feat_emo.view(-1, 128, 30)
        feat_emo = feat_emo.permute(2,0,1)
        
        expression_residue, _ = self.mfcc_lstm_1(feat_emo)
        expression_residue, _ = self.mfcc_lstm_2(expression_residue)
        
        full_kp = lip_kp + expression_residue
        
        return lip_kp, full_kp
    
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
            nn.Sigmoid()
            )
        
    def forward(self, kp_seq):
        
        kp_seq, _ = self.seq_lstm_1(kp_seq)
        kp_seq, _ = self.seq_lstm_2(kp_seq)
        kp_seq = kp_seq.permute(1,2,0)
        kp_seq = kp_seq.reshape(-1, 30*16)
        lab_realfake = self.seq_fc(kp_seq) # check if sigmoid is there in criterion
        
        return lab_realfake
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda


class Discriminator_LipSync(nn.Module):
    
    def __init__(self):
        super(Discriminator_LipSync,self).__init__()
        
        self.sync_conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,5), padding=(1,2)), # 80, 80
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)), # 40, 40
            nn.Conv2d(32, 64, (3,5), padding=(1,0)), # 40, 36
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), # 20, 36
            nn.Conv2d(64, 128, 3, padding=(1,0)), # 20, 34
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), # 10, 34
            nn.Conv2d(128, 128, 3, padding=(1,0)), # 10, 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), # 5, 32
            nn.Conv2d(128, 256, 3, padding=(1,0)), # 5, 30
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )
        
        self.sync_lstm_1 = nn.LSTM(256*5,256,bidirectional=True)
        self.sync_lstm_2 = nn.LSTM(256*2,136,bidirectional=False)
        
        self.sync_lstm_3 = nn.LSTM(136*2,8,bidirectional=True)
        
        self.sync_fc = nn.Sequential(
            nn.Linear(30*16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
            )
            
    def forward(self, mel, kp_seq):
        
        mel = self.sync_conv(mel)
        mel = mel.view(-1, 256*5, 30) # B, F, T
        mel = mel.permute(2,0,1) # T, B, F
        
        mel, _ = self.sync_lstm_1(mel)
        mel, _ = self.sync_lstm_2(mel)
        
        combo = torch.cat((mel,kp_seq),dim=2)
        combo, _ = self.sync_lstm_3(combo)
        
        combo = combo.permute(1,2,0)
        combo = combo.reshape(-1, 30*16)
        lab_lipsync = self.sync_fc(combo)
        
        return lab_lipsync
    
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