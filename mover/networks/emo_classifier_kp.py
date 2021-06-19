import torch.nn as nn
import torch

class CrossEmbed(nn.Module):
    
    def __init__(self):
        super(CrossEmbed,self).__init__()
        
        self.cross_lstm_1 = nn.LSTM(136,32,bidirectional=True)
        self.cross_lstm_2 = nn.LSTM(32*2,8,bidirectional=True)
        
        self.cross_fc = nn.Sequential(
            nn.Linear(30*16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8)
            ) # softmax not required since included in losses
            
    def forward(self, kp_seq):
        
        kp_seq = kp_seq.permute(1,0,2) # T, B, F
        
        kp_seq, _ = self.cross_lstm_1(kp_seq)
        kp_seq, _ = self.cross_lstm_2(kp_seq)
        
        kp_seq = kp_seq.permute(1,0,2)
        kp_seq = kp_seq.reshape(-1, 30*16)
        lab_emo = self.cross_fc(kp_seq)
        
        return lab_emo