import os
import torch
from torchvision.io import read_video
import pickle as pkl
import librosa
import numpy as np
from scipy import signal
from scipy import interpolate
import sys
import random
from PIL import Image
from torchvision.io import read_video, write_video
from torchvision.transforms import Resize

import audio

device = torch.device('cuda:0')
modelpath = 'models/final/'
path = '/storage/user/dasd/ppt/'
vid_list = os.listdir(path)
vid_list.sort()

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

kp_init = torch.load('kp_general.pt').unsqueeze(0)
# correct lip opening
shift = (kp_init[:,61:64,1] - kp_init[:,65:68,1])/2
kp_init[:,65:68,1] += shift; kp_init[:,61:64,1] -= shift
kp_init[:,56:59,1] += shift; kp_init[:,50:53,1] -= shift
kp_init[:,55,1] += shift.mean(); kp_init[:,59,1] += shift.mean()
kp_init[:,49,1] -= shift.mean(); kp_init[:,53,1] -= shift.mean()

G = torch.load(modelpath+'80Tr_G.model',map_location=device)
prTr_emo_model = torch.load('models/bestReTr_emo_classifier_seq.model',map_location=device)
prTr_CE = torch.load('models/bestpreTr_CE.model',map_location=device)

G.eval(); prTr_emo_model.eval(); prTr_CE.eval()

for vid_name in vid_list[1:]:
    
    (_,speech,fps) = read_video(path+vid_name)
    speech = speech[0].numpy(); fps = fps['audio_fps']
    if fps != 48000:
        speech = librosa.resample(speech,fps,48000)
        fps = 48000
     
    start = 0; all_mfcc = []; all_mel = []
    for i in range(min((speech.shape[0]-48000)//32000+1,8)):
        
        speech_seg = speech[start:start+fps]
        start = start + fps - 16000
        # print(i,speech_seg.shape)
        
        mfcc = librosa.feature.mfcc(speech_seg,sr=48000,n_fft=2048,hop_length=522,n_mfcc=30)
        mfcc = (torch.as_tensor(mfcc) - mfcc_mean) / mfcc_std
        all_mfcc.append(mfcc)
        
        mel = audio.melspectrogram(speech_seg)
        mel = torch.as_tensor(mel).unsqueeze(0).unsqueeze(0); mel = (mel - mel_mean) / mel_std
        all_mel.append(mel)
    
    mfcc = torch.vstack(all_mfcc).float().to(device)
    mel = torch.vstack(all_mel).float().to(device)
    # print(mfcc.shape, mel.shape)
    
    with torch.no_grad():
        lab_emo_sp, feat_emo = prTr_emo_model(mfcc)
        noise = torch.rand((4,512*5,30)) * 0.01
        pred_kp = G(mel,feat_emo,noise)
        if isinstance(pred_kp, tuple):
            pred_kp = pred_kp[0]
    
    # print(pred_kp.shape)
    all_pred = []
    for i in range(pred_kp.shape[0]):
        all_pred.append(pred_kp[i,5:25,:])
    pred_kp = torch.vstack(all_pred)
    # print(pred_kp.shape)
    pred_kp = signal.resample_poly(pred_kp.cpu(), 5, 6, padtype='smooth')
    pred_kp = pred_kp.reshape((-1,68,2))
    # print(pred_kp.shape)
    
    for up_idx in [49,50,51,52,53,61,62,63]:
        pos = (signal.argrelmax(pred_kp[:,up_idx,1], order=8))[0]
        if pos.shape[0] >= 2:
            inter = interpolate.interp1d(pos, pred_kp[pos,up_idx,1])
            base = np.zeros_like(pred_kp[:,up_idx,1])
            base[pos[0]:pos[-1]+1] = inter(range(pos[0],pos[-1]+1))
            base[:pos[0]] = base[pos[0]]; base[pos[-1]:] = base[pos[-1]]
            pred_kp[:,up_idx,1] = pred_kp[:,up_idx,1] - base
    for lo_idx in [55,56,57,58,59,65,66,67]:
        pos = (signal.argrelmin(pred_kp[:,lo_idx,1], order=8))[0]
        if pos.shape[0] >= 2:
            inter = interpolate.interp1d(pos, pred_kp[pos,lo_idx,1])
            base = np.zeros_like(pred_kp[:,lo_idx,1])
            base[pos[0]:pos[-1]+1] = inter(range(pos[0],pos[-1]+1))
            base[:pos[0]] = base[pos[0]]; base[pos[-1]:] = base[pos[-1]]
            pred_kp[:,lo_idx,1] = pred_kp[:,lo_idx,1] - base
    
    pred_kp = torch.as_tensor(pred_kp)
    torch.save(pred_kp,path+'0000/pred_kp'+vid_name+'.pt')

G.cpu(); prTr_emo_model.cpu(); prTr_CE.cpu()    

os.chdir('../../bilayer-model/examples/')
sys.path.append('../')

from infer import InferenceWrapper
args_dict = {
    'project_dir': '../',
    'init_experiment_dir': '../runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}
module = InferenceWrapper(args_dict)

resi = Resize(224)

for idx in range(len(vid_list)-1):
    kp = torch.load(path+'0000/pred_kp'+vid_list[idx+1]+'.pt')
    
    kp = kp + kp_init
    l = kp.min(dim=0,keepdim=True).values.min(dim=1,keepdim=True).values
    L = kp.max(dim=0,keepdim=True).values.max(dim=1,keepdim=True).values
    kp = (kp - l) / (L - l)
    kp = kp * 224 + (256-224)/2
    
    (frame,audio,fps) = read_video(path+vid_list[idx+1])
    fps = fps['audio_fps']
    if fps != 16000:
        speech = librosa.resample(speech,fps,16000)
        fps = 16000
    # assert fps['audio_fps'] == 16000
    
    source_name = random.choice(os.listdir('/usr/stud/dasd/storage/user/sources/'))
    source_img = np.asarray(Image.open('/usr/stud/dasd/storage/user/sources/'+source_name))
    
    input_data_dict = {'source_imgs': source_img,'target_kp': kp.float()}
    pred_output_dict = module(input_data_dict)
    
    op = pred_output_dict['pred_enh_target_imgs']
    seg = pred_output_dict['pred_target_segs']
    op = (((op.squeeze(0).permute(0,2,3,1).clamp(-1, 1) + 1) / 2) * 255)
    seg = seg.squeeze(0).permute(0,2,3,1)
    op = op * seg + 0. * (1 - seg) # 255. * (1 - seg)
    
    op = op.permute(0,3,1,2)
    op = resi(op)
    op = op.permute(0,2,3,1).cpu()
    op = op.type(torch.uint8)
    
    name = path+'0000/'+vid_list[idx+1]
    end = int(kp.shape[0]/25*16000)
    print(end)
    write_video(name, op, fps=25, video_codec='libx264', audio_array=audio[:,2666:end+2666], audio_fps=16000, audio_codec='aac')
    