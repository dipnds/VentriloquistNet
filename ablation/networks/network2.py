import torch.nn as nn
import torch

def init_weights(m):
    if type(m) == nn.LSTM:
        nn.init.orthogonal_(m.weight_ih_l0, gain=nn.init.calculate_gain('tanh'))
        nn.init.orthogonal_(m.weight_hh_l0, gain=nn.init.calculate_gain('tanh'))
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=10*nn.init.calculate_gain('tanh'))
        nn.init.constant_(m.bias, 0.0)

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        # 50
        self.conv_mel = nn.Sequential(
            nn.Conv1d(128, 256, 7, 1), # 44
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256, track_running_stats=True),
            nn.MaxPool1d(2, 2), # 22
            nn.Conv1d(256, 512, 3, 1), # 20
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, track_running_stats=True),
            nn.MaxPool1d(2, 2), # 10
            nn.Conv1d(512, 1024, 1, 1), # 10
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024, track_running_stats=True),
            )
        
        self.fc_key = nn.Sequential(
            nn.Linear(68*2, 64*5),
            nn.ReLU(inplace=True),
            )
        
        self.rnn1 = nn.LSTM(64*5+1024, 320, 1, batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(640, 160, 1, batch_first=True, bidirectional=True)
        self.rnn3 = nn.LSTM(320, 68*2, 1, bias=True, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(68*4, 68*2, bias=True)
        
        self.rnn1.apply(init_weights)
        self.rnn2.apply(init_weights)
        self.rnn3.apply(init_weights)
        self.fc.apply(init_weights)
    
    def forward(self, key0, mel):
        
        mel = self.conv_mel(mel)
        
        key0 = self.fc_key(key0)
        key0 = torch.unsqueeze(key0, dim=2)
        key0 = key0.repeat(1, 1, mel.shape[2])
        
        keyT = torch.cat((key0,mel), axis=1)
        keyT.transpose_(1,2)
        
        keyT,_ = self.rnn1(keyT)
        keyT,_ = self.rnn2(keyT)
        _,(keyT,_) = self.rnn3(keyT)
        
        keyT.transpose_(0,1)
        keyT = torch.flatten(keyT, start_dim=1)
        keyT = self.fc(keyT)
        
        return keyT
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda