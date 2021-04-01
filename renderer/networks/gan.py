import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlockDown, SelfAttention, ResBlock, ResBlockD, ResBlockUp, Padding, adaIN
import math
import sys

from networks.senet50_ft_dag import senet50_ft_dag as preTr_enc

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        self.enc = preTr_enc('models/senet50_ft_dag.pth')
        for param in self.enc.parameters(): param.requires_grad=False
        self.embed = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,512))
        
        self.proj_len = 2*(512*2*5 + 512+256 + 256+128 + 128+64 + 64+32 + 32)
        self.slice_idx = [0,
                    512*4, #res1
                    512*4, #res2
                    512*4, #res3
                    512*4, #res4
                    512*4, #res5
                    512*2 + 256*2, #resUp1
                    256*2 + 128*2, #resUp2
                    128*2 + 64*2, #resUp3
                    64*2 + 32*2, #resUp4
                    32*2] #last adain
        for i in range(1, len(self.slice_idx)):
            self.slice_idx[i] = self.slice_idx[i-1] + self.slice_idx[i]
    
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace = False)
        
        #in C*224*224 for voxceleb2
        self.pad = nn.ZeroPad2d((256-224)//2) #out C*256*256
        
        #Down
        self.resDown1 = ResBlockDown(1, 64, conv_size=9, padding_size=4) #out 64*128*128
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        
        self.self_att_Down = SelfAttention(256) #out 256*32*32
        
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.in4 = nn.InstanceNorm2d(512, affine=True)
        
        #Res
        #in 512*16*16
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.res4 = ResBlock(512)
        self.res5 = ResBlock(512)
        #out 512*16*16
        
        #Up
        #in 512*16*16
        self.resUp1 = ResBlockUp(512, 256) #out 256*32*32
        self.resUp2 = ResBlockUp(256, 128) #out 128*64*64
        
        self.self_att_Up = SelfAttention(128) #out 128*64*64

        self.resUp3 = ResBlockUp(128, 64) #out 64*128*128
        self.resUp4 = ResBlockUp(64, 32, out_size=(224, 224), scale=None, conv_size=3, padding_size=1) #out 3*224*224
        self.final = nn.Conv2d(32, 3, 3, padding = 1)
        
        self.proj = nn.Parameter(torch.rand(1,self.proj_len,512).normal_(0.0,0.02))
        # self.proj = self.p.unsqueeze(0)

    def forward(self, face_source, sketch_target):
        
        _, e = self.enc(face_source); e = e.squeeze(-1).squeeze(-1)
        p = self.proj.expand(e.shape[0],-1,-1)
        emb = self.embed(e)
        e_psi = torch.bmm(p, emb.unsqueeze(-1)) #B, p_len, 1
        
        #in 1*224*224 for voxceleb2
        out = self.pad(sketch_target)
        
        #Encoding
        out = self.resDown1(out); out = self.in1(out)
        out = self.resDown2(out); out = self.in2(out)
        out = self.resDown3(out); out = self.in3(out)
        out = self.self_att_Down(out)
        out = self.resDown4(out); out = self.in4(out)
        
        #Residual
        out = self.res1(out, e_psi[:, self.slice_idx[0]:self.slice_idx[1], :])
        out = self.res2(out, e_psi[:, self.slice_idx[1]:self.slice_idx[2], :])
        out = self.res3(out, e_psi[:, self.slice_idx[2]:self.slice_idx[3], :])
        out = self.res4(out, e_psi[:, self.slice_idx[3]:self.slice_idx[4], :])
        out = self.res5(out, e_psi[:, self.slice_idx[4]:self.slice_idx[5], :])
        
        
        #Decoding
        out = self.resUp1(out, e_psi[:, self.slice_idx[5]:self.slice_idx[6], :])
        out = self.resUp2(out, e_psi[:, self.slice_idx[6]:self.slice_idx[7], :])
        out = self.self_att_Up(out)
        out = self.resUp3(out, e_psi[:, self.slice_idx[7]:self.slice_idx[8], :])
        out = self.resUp4(out, e_psi[:, self.slice_idx[8]:self.slice_idx[9], :])
        out = adaIN(out,
                    e_psi[:,
                          self.slice_idx[9]:(self.slice_idx[10]+self.slice_idx[9])//2,
                          :],
                    e_psi[:,
                          (self.slice_idx[10]+self.slice_idx[9])//2:self.slice_idx[10],
                          :]
                   )
        
        out = self.relu(out)
        out = self.final(out)
        
        # out = self.sigmoid(out) # WHY?????????!!!!!!!!!!!!!!!!!!!!!!!
        # OR resize to 256 and center crop of 224?????????????????? 
        
        #out 3*224*224
        return out, e


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.relu = nn.LeakyReLU()
        
        #in 4*224*224
        self.pad = Padding(224) #out 4*256*256
        self.resDown1 = ResBlockDown(4, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.res = ResBlockD(512) #out 512*4*4
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,1)) #out 512*1*1
        self.fc_id = nn.Sequential(
            nn.Linear(2048,256),
            nn.ReLU())
        self.fc_combo = nn.Sequential(
            nn.Linear(512+256,64),
            nn.ReLU(),
            nn.Linear(64,1))
    
    def forward(self, face, sketch, e):
        out = torch.cat((face,sketch), dim=-3) #out B*4*224*224
        out = self.pad(out)
        out1 = self.resDown1(out)
        out2 = self.resDown2(out1)
        out3 = self.resDown3(out2)
        out = self.self_att(out3)
        out4 = self.resDown4(out)
        out5 = self.resDown5(out4)
        out6 = self.resDown6(out5)
        out7 = self.res(out6)
        out = self.sum_pooling(out7)
        out = self.relu(out)
        out = out.squeeze(-1).squeeze(-1) #out B*512*1

        e = self.fc_id(e)
        out = torch.cat((out,e), dim=-1)
        out = self.fc_combo(out)
        
        return out, [out1 , out2, out3, out4, out5, out6, out7]
