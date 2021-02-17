import torch.nn as nn
import torch
from torchvision.transforms import Resize
from networks.alexnet_face_fer_bn_dag import alexnet_face_fer_bn_dag as preTr_enc

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        enc = preTr_enc('models/alexnet_face_fer_bn_dag.pth')
        for param in enc.parameters():
            param.requires_grad=False
        enc.conv5 = None; enc.bn5 = None; enc.relu5 = None
        enc.pool5 = None
        enc.fc6 = None; enc.bn6 = None; enc.relu6 = None
        enc.fc7 = None; enc.bn7 = None; enc.relu7 = None
        enc.fc8 = None
        
        self.enc1 = nn.Sequential(enc.conv1,enc.bn1,enc.relu1,enc.pool1)
        self.enc2 = nn.Sequential(enc.conv2,enc.bn2,enc.relu2,enc.pool2)
        self.enc3 = nn.Sequential(enc.conv3,enc.bn3,enc.relu3)
        self.enc4 = nn.Sequential(enc.conv4,enc.bn4,enc.relu4)
        
        self.resize2 = Resize([13,13])
        self.resize1 = Resize([27,27])
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384, track_running_stats=True)
            )
        self.dec3 = nn.Sequential(
            nn.Conv2d(384*2, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, track_running_stats=True)
            )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256*2+3, 96, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, track_running_stats=True),
            nn.Upsample(size=(27,27))
            )
        self.dec1 = nn.Sequential(
            nn.Conv2d(96*2+3, 96, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96, track_running_stats=True),
            nn.Upsample(size=(227,227))
            )
        self.final = nn.Conv2d(96+3, 3, 3, 1, padding=1)
        
        self.shift_1 = nn.Sequential(
            nn.Conv2d(2, 16, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(14*14*32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
            )
        self.shift_final = nn.Sequential(
            nn.Conv2d(2, 16, 7, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(25*25*32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
            )
        
        self.unfold = nn.Unfold(kernel_size=(75, 75),stride=38) # (227-75)/38 + 1 = 5
        self.fold = nn.Fold((227,227),kernel_size=(75, 75),stride=38)
    
    def unroll(self, t, numCh):
        t = self.unfold(t)
        t = t.view(-1,numCh,75,75,5*5)
        t = t.permute(0,4,1,2,3)
        t = t.reshape(-1,numCh,75,75)
        return t
    
    def deform(self, theta, t):
        grid = nn.functional.affine_grid(theta, t.shape)
        t = nn.functional.grid_sample(t, grid)
        return t
    
    def forward(self, sketch0, sketchT, face0, faceT):
        
        # for now, it is 1-way, i.e. face0 -> faceT, but not vice versa
        
        img0 = face0.clone() # 227
        
        # initialise theta to the 'no affine transform' case, predict residues
        # [[1 0 0]
        #   0 1 0]]
        theta_init = torch.zeros(1,2,3)
        theta_init[0,0,0] = 1; theta_init[0,1,1] = 1
        
        face0_1 = self.enc1(face0) # 27
        face0_2 = self.enc2(face0_1) # 13
        face0_3 = self.enc3(face0_2) # 13
        face0_4 = self.enc4(face0_3) # 13
        
        # sketch resize, estimate theta for flow
        sketch0_1 = self.resize1(sketch0); sketchT_1 = self.resize1(sketchT)
        theta_1 = self.shift_1(torch.cat((sketch0_1,sketchT_1),axis=1))
        theta_1 = theta_1.reshape(-1,2,3) + theta_init
        
        # deform feature maps
        face0_1 = self.deform(theta_1,face0_1)
        face0_2 = self.deform(theta_1,face0_2)
        face0_3 = self.deform(theta_1,face0_3)
        face0_4 = self.deform(theta_1,face0_4)
        
        # deform face
        img0_2 = self.resize2(img0)
        img0_2 = self.deform(theta_1, img0_2)
        
        img0_1 = self.resize1(img0)
        img0_1 = self.deform(theta_1, img0_1)

        # !!! remember to concatenate 0 to T and vice-versa
        
        # decode levels 4 to 1
        face0_4 = self.dec4(face0_4) # 13
        face0_3 = torch.cat((face0_4,face0_3), 1) # 13
        face0_3 = self.dec3(face0_3) # 13
        face0_2 = torch.cat((face0_3,face0_2,img0_2), 1) # 13
        face0_2 = self.dec2(face0_2) # 27
        face0_1 = torch.cat((face0_2,face0_1,img0_1), 1) # 27
        face0_1 = self.dec1(face0_1) # 227
        
        # unroll sketches into blocks and estimate theta per block
        sketch0 = self.unroll(sketch0,1); sketchT = self.unroll(sketchT,1)
        theta_final = self.shift_final(torch.cat((sketch0,sketchT),axis=1))
        theta_final = theta_final.reshape(-1,2,3) + theta_init
        
        # unroll image, deform, then roll back
        img0 = self.unroll(img0,3)
        img0 = self.deform(theta_final,img0)
        img0 = img0.reshape(-1,5*5,3,75,75)
        img0 = img0.permute(0,2,3,4,1)
        img0 = img0.view(face0.shape[0],-1,5*5)
        img0 = self.fold(img0)
                
        face0 = torch.cat((face0_1,img0), 1) # 227
        face0 = self.final(face0) # 227
        
        return face0, self.resize1(faceT), img0_1, img0
    # Can add MSE loss on img0 and img0_1, (perceptual + identity + style + edge + ...) loss on face0
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    