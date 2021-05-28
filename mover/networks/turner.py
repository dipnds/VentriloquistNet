import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        # 138
        self.fc = nn.Sequential(
            nn.Linear(138, 138),
            nn.ReLU(inplace=True),
            nn.Linear(138, 136),
            nn.ReLU(inplace=True),
            nn.Linear(136, 136)
            )
    
    def forward(self, kp):
        
        kp = self.fc(kp)
        return kp
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda