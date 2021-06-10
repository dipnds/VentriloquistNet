import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        # 30, 46 (f, t=0.5s)
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (3,7), padding=(1,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), #15,46
            #nn.Dropout(p=0.2),
            nn.Conv2d(32, 32, (3,5), padding=(1,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1), padding=(1,0)), #8,46
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), #4,46
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,3)) #1,15
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128*15, 256),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 7)
            )
    
    def forward(self, mel):
        
        feat = self.feature(mel)
        lab = self.classifier(feat)
        return lab, feat
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
