#https://www.kaggle.com/ejlok1/audio-emotion-part-6-2d-cnn-66-accuracy
import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, (3,7), padding=(1,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            #nn.Dropout(p=0.2),
            nn.Conv2d(32, 32, (3,5), padding=(1,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            #nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            #nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            #nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(128*5, 128)
            )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 7)
            )
    
    def forward(self, mel):
        
        feat = self.feature(mel)
        lab = self.classifier(feat)
        return lab, feat
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
