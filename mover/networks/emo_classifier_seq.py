import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        # 30, 92 (f, t=1s)
        self.feature = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Conv2d(1, 32, (3,7), padding=(1,3)),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), #15,92
            nn.Conv2d(32, 32, (3,5), padding=(1,2)),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1), padding=(1,0)), #8,92
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), #4,92
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((4,1)),
            nn.MaxPool2d((1,3)) #1,30
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*30, 128),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8)
            ) # softmax not required since included in CrossEntropyLoss
    
    def forward(self, mel):
        
        feat = self.feature(mel)
        lab = self.classifier(feat)
        
        return lab, feat
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
