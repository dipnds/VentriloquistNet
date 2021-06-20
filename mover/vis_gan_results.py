import torch
import numpy as np
import pickle as pkl
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

def func(x, a, b, c):    
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def interpPoints(x, y, lim):    
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x, lim)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
            
    curve_x = np.clip(curve_x,0,lim[1]-1); curve_y = np.clip(curve_y,0,lim[0]-1)
    return curve_x.astype(int), curve_y.astype(int)

def kp2sketch(kp,h,w):
    lim = list([h,w])
    sketch = torch.zeros(h,w).type(torch.uint8)
    curve_x, curve_y = interpPoints(kp[0:5,0],kp[0:5,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[4:9,0],kp[4:9,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[8:13,0],kp[8:13,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[12:17,0],kp[12:17,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[17:22,0],kp[17:22,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[22:27,0],kp[22:27,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[27:31,0],kp[27:31,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[31:36,0],kp[31:36,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[36:40,0],kp[36:40,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[[39,40,41,36],0],kp[[39,40,41,36],1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[42:46,0],kp[42:46,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[[45,46,47,42],0],kp[[45,46,47,42],1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[48:55,0],kp[48:55,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[[54,55,56,57,58,59,48],0],kp[[54,55,56,57,58,59,48],1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[60:65,0],kp[60:65,1],lim); sketch[curve_y,curve_x] = 1
    curve_x, curve_y = interpPoints(kp[[64,65,66,67,60],0],kp[[64,65,66,67,60],1],lim); sketch[curve_y,curve_x] = 1
    sketch = sketch.type(torch.bool)
    return sketch

# M013 angry 028; M031 happy 010; M034 neutral 038; W036 sad 029; W029 disgusted 033; W036 surprised 029
# M003 neutral 017; angry 026; happy 028; sad 007; disgusted 015; surprised 026

person = 'M003/' 
emotion = 'happy/'
level = 'level_3/'
utterance = '028'

# device = torch.device('cpu')
device = torch.device('cuda:0')

modelpath = 'models/'#'G1iterD1iter/'
#datapath = '/media/deepan/Backup/thesis/mead/processed/'
datapath = '/usr/stud/dasd/workspace/mead/processed/'
file_list = [datapath + person + emotion + level + utterance]

mfcc = pkl.load(open('/usr/stud/dasd/workspace/'+'emo_data.pkl','rb'))['feat_train']
mfcc_mean = np.mean(mfcc, axis=(0,2))
mfcc_mean = torch.tensor(mfcc_mean).float().unsqueeze(-1)
mfcc_std = np.std(mfcc, axis=(0,2))
mfcc_std = torch.tensor(mfcc_std).float().unsqueeze(-1)

mel = torch.load('mel_norm.pt',map_location=(torch.device('cpu')))
mel_mean = mel['m']
mel_mean = mel_mean.unsqueeze(0).unsqueeze(-1)
mel_std = mel['s']
mel_std = mel_std.unsqueeze(0).unsqueeze(-1)

kp_init = torch.load('kp_general.pt').flatten().unsqueeze(0).to(device)

def infer(G, prTr_emo_model, mel, mfcc):
    
    G.eval(); prTr_emo_model.eval() 
    with torch.no_grad():
        mfcc = mfcc.to(device); mel = mel.to(device)
        lab_emo, feat_emo = prTr_emo_model(mfcc)
        pred_kp = G(mfcc,feat_emo)
        return pred_kp

G = torch.load(modelpath+'bestTr_G.model',map_location=device)
# G = torch.load(modelpath+'20Tr_G.model',map_location=device)
prTr_emo_model = torch.load('models/bestReTr_emo_classifier_seq.model',map_location=device)

for file in file_list:
    
    duration = 4; offset = 0
    
    mel = torch.load(file + '/mel.pt')[:,offset*3:92*duration+offset*3,:]
    mel = mel.permute((2,0,1)) # C,F,T
    mel = (mel - mel_mean) / mel_std
    mel = torch.stack((mel[:,:,0:92],mel[:,:,92:184],mel[:,:,184:276],mel[:,:,276:368]))
    
    mfcc = torch.load(file + '/mfcc.pt')[:,offset*3:92*duration+offset*3,:]
    mfcc = (mfcc - mfcc_mean) / mfcc_std
    mfcc = mfcc.permute((2,0,1))
    mfcc = torch.stack((mfcc[:,:,0:92],mfcc[:,:,92:184],mfcc[:,:,184:276],mfcc[:,:,276:368]))
    
    gt = torch.load(file + '/kp_seq.pt')[offset:30*duration+offset].float()
    m = gt.mean(dim=(0,1),keepdims=True)
    s = gt.std(dim=(0,1),keepdims=True)
    
    pred = infer(G,prTr_emo_model,mel.float(),mfcc.float()) # B, T, F
    if isinstance(pred, tuple):
        pred = pred[0] + pred[1]
    pred = torch.cat((pred[0,:,:],pred[1,:,:],pred[2,:,:],pred[3,:,:]),dim=0)

    print(pred.abs().mean(dim=1))
    temp = ((gt - m) / s).flatten(start_dim=1) - kp_init.to(torch.device('cpu'))
    print(temp.abs().mean(dim=1))

    pred = pred + kp_init

    print(pred.abs().mean(dim=1))
    temp = ((gt - m) / s).flatten(start_dim=1)
    print(temp.abs().mean(dim=1))

    pred = pred.reshape((-1,68,2)).cpu()
    pred = pred * s + m

    print(pred.abs().mean(dim=(1,2)))
    print(gt.abs().mean(dim=(1,2)))
    torch.save(pred,'results/pred_kp.pt')
     
    # pred = torch.load('results/pred_kp.pt')
    
    # pred_m = pred.mean(dim=1,keepdim=True)
    # pred_s = pred.std(dim=1,keepdim=True)
    # pred = (pred - pred_m) / pred_s
    # pred = pred.flatten(start_dim=1)
    # turn = torch.tensor([2/3,0])
    
    # ip = []
    # for i in range(pred.shape[0]):
    #     temp = torch.cat((pred[i],turn))
    #     ip.append(temp)
    # ip = torch.stack(ip).to(device)
    
    # turner = torch.load('models/bestEv_turner.model')
    # turned_pred = turner(ip)
    
    # turned_pred = (turned_pred.detach().cpu().reshape(-1,68,2) * pred_s) + pred_m
    # print(turned_pred)
    # torch.save(turned_pred,'results/turned_kp.pt')
    
    l_min = gt.min(dim=0,keepdim=True).values.min(dim=1,keepdim=True).values - 100
    l_max = gt.max(dim=0,keepdim=True).values.max(dim=1,keepdim=True).values + 100
    gt = (gt - l_min).round().numpy().astype(int); pred = (pred - l_min).round().numpy().astype(int)
    
        
    w,h = int((l_max-l_min)[0,0,0]), int((l_max-l_min)[0,0,1])
    
    for i in range(gt.shape[0]):
        
        print(i)
        
        gt_sketch = (kp2sketch(gt[i,:,:],h,w)).type(torch.uint8) * 255
        pred_sketch = (kp2sketch(pred[i,:,:],h,w)).type(torch.uint8) * 255
        dummy = torch.ones_like(gt_sketch) - gt_sketch - pred_sketch
        dummy = dummy.type(torch.uint8) * 255
        gt_sketch = 255 - gt_sketch; pred_sketch = 255 - pred_sketch
        full = torch.stack((gt_sketch,pred_sketch,dummy),dim=-1)
        plt.imshow(full)
        
        plt.savefig('results/'+str(i)+'.png')
        plt.close()
