import torch.nn as nn
import torch
from torchvision.transforms import Resize
from networks.vgg_face_dag import vgg_face_dag as preTr_enc

def init_weights(m):
    pass
    # if type(m) == nn.LSTM:
    #     nn.init.orthogonal_(m.weight_ih_l0, gain=nn.init.calculate_gain('tanh'))
    #     nn.init.orthogonal_(m.weight_hh_l0, gain=nn.init.calculate_gain('tanh'))
    # if type(m) == nn.Linear:
    #     nn.init.xavier_normal_(m.weight, gain=10*nn.init.calculate_gain('tanh'))
    #     nn.init.constant_(m.bias, 0.0)

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.enc = preTr_enc('models/vgg_face_dag.pth')
        for param in self.enc.parameters():
            param.requires_grad=False # this is missing ReLU and Pool
        self.enc.pool5 = None
        self.enc.fc6 = None; self.enc.relu6 = None; self.enc.dropout6 = None
        self.enc.fc7 = None; self.enc.relu7 = None; self.enc.dropout7 = None
        self.enc.fc8 = None
        
        self.resize4 = Resize([14,14])
        self.dec4 = nn.Sequential(
            nn.Conv2d(512+3+1, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2) # 256, 28, 28
            )
        self.resize3 = Resize([28,28])
        self.dec3 = nn.Sequential(
            nn.Conv2d(256+3+1, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2) # 128, 56, 56
            )
        self.resize2 = Resize([56,56])
        self.dec2 = nn.Sequential(
            nn.Conv2d(128+3+1, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2) # 64, 112, 112
            )
        self.resize1 = Resize([112,112])
        self.dec1 = nn.Sequential(
            nn.Conv2d(64+3+1, 32, 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(32, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2), # 64, 224, 224
            )
        self.dec_final = nn.Conv2d(32+3+1, 3, 3, 1, padding=2, dilation=2)
    
    def forward(self, key0, keyT, face0, faceT):
        
        img0 = face0.clone(); imgT = faceT.clone()
        face0 = self.enc(face0); faceT = self.enc(faceT)
        
        # !!! remember to concatenate 0 to T and vice-versa
        face0 = torch.cat((face0,self.resize4(keyT),self.resize4(img0)), 1)
        faceT = torch.cat((faceT,self.resize4(key0),self.resize4(imgT)), 1)
        face0 = self.dec4(face0); faceT = self.dec4(faceT)
        
        face0 = torch.cat((face0,self.resize3(keyT),self.resize3(img0)), 1)
        faceT = torch.cat((faceT,self.resize3(key0),self.resize3(imgT)), 1)
        face0 = self.dec3(face0); faceT = self.dec3(faceT)
        
        face0 = torch.cat((face0,self.resize2(keyT),self.resize2(img0)), 1)
        faceT = torch.cat((faceT,self.resize2(key0),self.resize2(imgT)), 1)
        face0 = self.dec2(face0); faceT = self.dec2(faceT)
        
        face0 = torch.cat((face0,self.resize1(keyT),self.resize1(img0)), 1)
        faceT = torch.cat((faceT,self.resize1(key0),self.resize1(imgT)), 1)
        face0 = self.dec1(face0); faceT = self.dec1(faceT)
        
        face0 = torch.cat((face0,keyT,img0), 1)
        faceT = torch.cat((faceT,key0,imgT), 1)
        face0 = self.dec_final(face0); faceT = self.dec_final(faceT)

        return face0, faceT
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda