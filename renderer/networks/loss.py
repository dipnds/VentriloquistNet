import torch.nn as nn
from networks.alexnet_face_fer_bn_dag import alexnet_face_fer_bn_dag as preTr_enc

class struct_loss(nn.Module):
    
    def __init__(self):
        super(struct_loss,self).__init__()
        
        self.enc = preTr_enc('models/alexnet_face_fer_bn_dag.pth')
        for param in self.enc.parameters():
            param.requires_grad=False # this is missing ReLU and Pool
        self.enc.bn5 = nn.Identity(); self.enc.relu5 = nn.Identity()
        self.enc.pool5 = None
        self.enc.fc6 = None; self.enc.bn6 = None; self.enc.relu6 = None
        self.enc.fc7 = None; self.enc.bn7 = None; self.enc.relu7 = None
        self.enc.fc8 = None
        
    def forward(self, faceA, faceB):
        
        faceA = self.enc(faceA); faceB = self.enc(faceB)
        err = nn.functional.mse_loss(faceA,faceB)
        
        return err