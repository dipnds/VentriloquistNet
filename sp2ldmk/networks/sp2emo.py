import torch.nn as nn

class sp2emo(nn.Module):
    
    def __init__(self):
        super(sp2emo,self).__init__()
        
        # 80, 75 (f, t=1s)
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (3,7), padding=(1,3)),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), #40,75
            nn.Conv2d(32, 32, (3,5), padding=(1,2)),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), #20,75
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)), #10,75
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2,1)),
            nn.MaxPool2d((1,3)) #5,25
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*25, 128),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 7)
            ) # softmax not required since included in CrossEntropyLoss
    
    def forward(self, mel):
        
        feat = self.feature(mel)
        lab = self.classifier(feat)
        
        return lab, feat
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
