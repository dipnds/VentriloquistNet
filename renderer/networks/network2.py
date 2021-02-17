import torch.nn as nn
import torch
from torchvision.transforms import Resize
import face_alignment
from networks.alexnet_face_fer_bn_dag import alexnet_face_fer_bn_dag as preTr_enc

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
        
        self.enc = preTr_enc('models/alexnet_face_fer_bn_dag.pth')
        for param in self.enc.parameters():
            param.requires_grad=False # this is missing ReLU and Pool
        self.enc.pool5 = None
        self.enc.fc6 = None; self.enc.bn6 = None; self.enc.relu6 = None
        self.enc.fc7 = None; self.enc.bn7 = None; self.enc.relu7 = None
        self.enc.fc8 = None
        
        self.resize1 = Resize([13,13])
        # 256, 13, 13
        self.dec1 = nn.Sequential(
            nn.Conv2d(260, 64, 3, 1, padding=1), # 64, 13, 13
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2) # 64, 26, 26
            )
        self.resize2 = Resize([26,26])
        self.dec2 = nn.Sequential(
            nn.Conv2d(68, 16, 3, 1, padding=1), # 16, 26, 26
            nn.BatchNorm2d(16, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=3) # 16, 78, 78
            )
        self.resize3 = Resize([78,78])
        self.dec3 = nn.Sequential(
            nn.Conv2d(20, 4, 3, 1, padding=1), # 4, 78, 78
            nn.BatchNorm2d(4, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(227,227)) # 4, 227, 227
            )
        self.dec4 = nn.Conv2d(8, 3, 3, 1, padding=1) # 3, 227, 227
    
    def forward(self, key0, keyT, face0, faceT):
        
        img0 = face0.clone(); imgT = faceT.clone()
        face0 = self.enc(face0); faceT = self.enc(faceT)
        
        # !!! remember to concatenate 0 to T and vice-versa
        face0 = torch.cat((face0,self.resize1(keyT),self.resize1(img0)), 1)
        faceT = torch.cat((faceT,self.resize1(key0),self.resize1(imgT)), 1)
        face0 = self.dec1(face0); faceT = self.dec1(faceT)
        
        face0 = torch.cat((face0,self.resize2(keyT),self.resize2(img0)), 1)
        faceT = torch.cat((faceT,self.resize2(key0),self.resize2(imgT)), 1)
        face0 = self.dec2(face0); faceT = self.dec2(faceT)
        
        face0 = torch.cat((face0,self.resize3(keyT),self.resize3(img0)), 1)
        faceT = torch.cat((faceT,self.resize3(key0),self.resize3(imgT)), 1)
        face0 = self.dec3(face0); faceT = self.dec3(faceT)
        
        face0 = torch.cat((face0,keyT,img0), 1)
        faceT = torch.cat((faceT,key0,imgT), 1)
        face0 = self.dec4(face0); faceT = self.dec4(faceT)
        
        return face0, faceT
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    
class faceKP(nn.Module):
    
    def __init__(self, device):
        super(faceKP, self).__init__()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                       device='cuda', face_detector='blazeface')
        norm = torch.load('../norm.pt')
        mean_face = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(norm['mean_face'], 1), 2), 0)
        self.mean_face = mean_face.to(device)
        std_face = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(norm['std_face'], 1), 2), 0)
        self.std_face = std_face.to(device)
        
    def forward(self, face, key):
        
        face = face * 255*self.std_face + self.mean_face
        kp = self.fa.get_landmarks_from_batch(face)
        
        for i in len(kp):
            if kp[i][0].shape[0] == 0:
                err = (key[i,:,:]**2).sum()
        
        return 0