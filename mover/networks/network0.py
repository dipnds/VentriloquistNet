import torch.nn as nn
import torch

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        # if m.bias is not None:
        #     nn.init.constant_(m.bias, 0.01)    

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        # 50
        self.conv_mel = nn.Sequential(
            nn.Conv1d(128, 128, 7, 1), # 44
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, track_running_stats=True),
            nn.MaxPool1d(2, 2), # 22
            nn.Conv1d(128, 128, 3, 1), # 20
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, track_running_stats=True),
            nn.MaxPool1d(2, 2), # 10
            nn.Conv1d(128, 128, 1, 1), # 10
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, track_running_stats=True)
            )
        
        # self.fc_key = nn.Sequential(
        #     nn.Linear(68*2, 64*5),
        #     nn.ReLU(inplace=True),
        #     )
        
        self.fc = nn.Sequential(
            nn.Linear(128*10+68*2, 450),
            nn.ReLU(inplace=True),
            nn.Linear(450, 68*2)
            )
        self.fc.apply(init_weights)
    
    def forward(self, key0, mel):
        
        mel = self.conv_mel(mel)
        mel = torch.flatten(mel, start_dim=1)
                
        # key0 = self.fc_key(key0)
        keyT = torch.cat((key0,mel), axis=1)
        keyT = self.fc(keyT)
        
        return keyT
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda