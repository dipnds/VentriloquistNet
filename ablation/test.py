import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import librosa
from torchvision.io import read_image
from scipy.signal import savgol_filter

import audio
import face_alignment

# def kp2sketch(kp,h,w):
#     lim = list([h,w])
#     sketch = torch.zeros(h,w).type(torch.uint8)
#     # curve_x, curve_y = interpPoints(kp[0:5,0],kp[0:5,1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[4:9,0],kp[4:9,1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[8:13,0],kp[8:13,1],lim); sketch[curve_y,curve_x] = 1
#     # curve_x, curve_y = interpPoints(kp[12:17,0],kp[12:17,1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[17:22,0],kp[17:22,1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[22:27,0],kp[22:27,1],lim); sketch[curve_y,curve_x] = 1
#     # curve_x, curve_y = interpPoints(kp[27:31,0],kp[27:31,1],lim); sketch[curve_y,curve_x] = 1
#     # curve_x, curve_y = interpPoints(kp[31:36,0],kp[31:36,1],lim); sketch[curve_y,curve_x] = 1
#     # curve_x, curve_y = interpPoints(kp[36:40,0],kp[36:40,1],lim); sketch[curve_y,curve_x] = 1
#     # curve_x, curve_y = interpPoints(kp[[39,40,41,36],0],kp[[39,40,41,36],1],lim); sketch[curve_y,curve_x] = 1
#     # curve_x, curve_y = interpPoints(kp[42:46,0],kp[42:46,1],lim); sketch[curve_y,curve_x] = 1
#     # curve_x, curve_y = interpPoints(kp[[45,46,47,42],0],kp[[45,46,47,42],1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[48:55,0],kp[48:55,1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[[54,55,56,57,58,59,48],0],kp[[54,55,56,57,58,59,48],1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[60:65,0],kp[60:65,1],lim); sketch[curve_y,curve_x] = 1
#     curve_x, curve_y = interpPoints(kp[[64,65,66,67,60],0],kp[[64,65,66,67,60],1],lim); sketch[curve_y,curve_x] = 1
#     sketch = sketch.type(torch.bool)
#     return sketch

device = torch.device('cuda:0')

datapath = '/usr/stud/dasd/storage/user/testbed/'
file = 'id03862'
print('******************************',file)

mfcc = pkl.load(open('/usr/stud/dasd/workspace/'+'emo_data_mfcc.pkl','rb'))['feat_train']
mfcc_mean = np.mean(mfcc, axis=(0,2))
mfcc_mean = torch.tensor(mfcc_mean).float().unsqueeze(0).unsqueeze(0)
mfcc_std = np.std(mfcc, axis=(0,2))
mfcc_std = torch.tensor(mfcc_std).float().unsqueeze(0).unsqueeze(0)

mel = torch.load('mel_norm.pt',map_location=(torch.device('cpu')))
mel_mean = mel['m']
mel_mean = mel_mean.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
mel_std = mel['s']
mel_std = mel_std.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

kp_init = torch.load('kp_general.pt')#.flatten().unsqueeze(0).to(device)

face_source = read_image(datapath+file+'/source.jpg')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,device='cuda')
target_kp = torch.tensor(fa.get_landmarks(face_source.permute(1,2,0).numpy())[0])

print(kp_init.max(dim=0).values - kp_init.min(dim=0).values)
print(target_kp.max(dim=0).values - target_kp.min(dim=0).values)
r_init = kp_init.max(dim=0).values - kp_init.min(dim=0).values
r_target = target_kp.max(dim=0).values - target_kp.min(dim=0).values
r = (r_target[1] / r_target[0]) / (r_init[1] / r_init[0])
kp_init[:,1] = kp_init[:,1] * r
print(kp_init.max(dim=0).values - kp_init.min(dim=0).values)

def infer(G, prTr_emo_model, prTr_CE, mel, mfcc):
    
    G.eval(); prTr_emo_model.eval()
    
    with torch.no_grad():
        mfcc = mfcc.to(device)
        mel = mel.to(device)
        noise = torch.rand((mel.shape[0],512*5,30)) * 0.01; noise = noise.to(device)
        lab_emo_sp, feat_emo = prTr_emo_model(mfcc)
        print(nn.functional.softmax(lab_emo_sp,dim=1))
        pred_kp = G(mel,feat_emo)#,noise)
        if isinstance(pred_kp, tuple):
            pred_kp = pred_kp[0]
        lab_emo_kp = prTr_CE(pred_kp)
        print(nn.functional.softmax(lab_emo_kp,dim=1))
        return pred_kp

# G = torch.load('models/final/80Tr_G.model',map_location=device) # main
G = torch.load('models/ablation/80Tr_G.model',map_location=device) # ablation
print(G)
prTr_emo_model = torch.load('models/final/bestReTr_emo_classifier_seq.model',map_location=device)
# prTr_emo_model = torch.load('models/bestEv_emo_classifier_seq.model',map_location=device)
prTr_CE = torch.load('models/final/bestpreTr_CE.model',map_location=device)

####################################    
duration = 4; offset = 0

(speech,_) = librosa.core.load(datapath+file+'/speech.wav', sr=48000)
speech = np.pad(speech,8000,mode='reflect')

mel = []
mfcc = []
for s in range(6):
    sp = speech[32000*s:32000*s+48000]
    print(sp.shape)
    mel.append(torch.tensor(audio.melspectrogram(sp)))
    mfcc.append(torch.tensor(librosa.feature.mfcc(sp,sr=48000,n_fft=2048,hop_length=522,n_mfcc=30)))
mel = torch.stack(mel)
mel = mel.unsqueeze(1)
mfcc = torch.stack(mfcc)
mfcc = mfcc.unsqueeze(1)
mel = (mel - mel_mean) / mel_std
mfcc = (mfcc - mfcc_mean) / mfcc_std

print(mel.shape, mfcc.shape)
pred = infer(G,prTr_emo_model,prTr_CE,mel.float(),mfcc.float()) # B, T, F
pred = torch.cat((pred[0,5:25,:],pred[1,5:25,:],pred[2,5:25,:],pred[3,5:25,:],pred[4,5:25,:],pred[5,5:25,:]),dim=0)

kp_init = kp_init.flatten().unsqueeze(0).to(device)
pred = pred + kp_init
pred = pred.reshape((-1,68,2)).cpu()

temp = torch.cat((pred[:,0:3,:],pred[:,14:17,:]),axis=1)
print(temp.shape)
tm = temp.mean(dim=(1),keepdim=True)
pred = pred - tm
tm = tm.numpy()
tm[:,:,0] = torch.tensor(savgol_filter(tm[:,:,0],27,0,axis=0,mode='constant'))
tm[:,:,1] = torch.tensor(savgol_filter(tm[:,:,1],27,1,axis=0))#,mode='nearest'))
# tm[:,:,]
pred = pred + tm

# l = pred.min(dim=0,keepdim=True).values.min(dim=1,keepdim=True).values
# L = pred.max(dim=0,keepdim=True).values.max(dim=1,keepdim=True).values

l = pred.min(dim=0,keepdim=True).values.min(dim=1,keepdim=True).values.min(dim=2,keepdim=True).values
L = pred.max(dim=0,keepdim=True).values.max(dim=1,keepdim=True).values.max(dim=2,keepdim=True).values

print(l,L)
pred = (pred - l) / (L - l)
pred = pred * 224 + (256-224)/2
torch.save(pred,datapath+file+'/pred_kp.pt')

tm = pred.mean(dim=1)
plt.plot(range(tm.shape[0]),tm[:,0]); plt.plot(range(tm.shape[0]),tm[:,1])
plt.savefig(datapath+file+'/_cropped/'+'mean_plot.png',bbox_inches='tight')
plt.close()
