import torch
import os
import pickle as pkl
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
from torchvision.io import read_video, write_jpeg, write_video
from torchvision.transforms import Resize
import sys
import random
from PIL import Image

device = torch.device('cuda:0')
datapath = '/usr/stud/dasd/storage/user/vox2/test/processed_mead/'
modelpath = 'models/final/'
testpath = '/usr/stud/dasd/storage/user/syncnet_target/'

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
# plt.scatter(kp_init[:,:,0],-kp_init[:,:,1])
# plt.show()

# correct lip opening
shift = (kp_init[:,61:64,1] - kp_init[:,65:68,1])/2
kp_init[:,65:68,1] += shift; kp_init[:,61:64,1] -= shift
kp_init[:,56:59,1] += shift; kp_init[:,50:53,1] -= shift
kp_init[:,55,1] += shift.mean(); kp_init[:,59,1] += shift.mean()
kp_init[:,49,1] -= shift.mean(); kp_init[:,53,1] -= shift.mean()
# plt.scatter(kp_init[:,:,0],-kp_init[:,:,1])
# plt.show()

if not os.path.isfile('../datalist_vox2mead_test.pkl'):

    person_list = pkl.load(open('../split_vox2.pkl','rb'))['test']
    datalist = []
    
    for person in person_list:
        
        try:
            for utter in os.listdir(datapath+person+'/'):
                        
                temp = os.listdir(datapath+person+'/'+utter)
                if (('mel1.pt' in temp) and ('mel2.pt' in temp)
                    and ('mel3.pt' in temp) and ('mel4.pt' in temp)
                    and ('mfcc1.pt' in temp) and ('mfcc2.pt' in temp)
                    and ('mfcc3.pt' in temp) and ('mfcc4.pt' in temp)):
                    datalist.append(datapath+person+'/'+utter+'/')
        except: print(person,'*******error')
                    
    datalist = datalist
    pkl.dump(datalist,open('../datalist_vox2mead_test.pkl','wb'))

else: datalist = pkl.load(open('../datalist_vox2mead_test.pkl','rb'))

G = torch.load(modelpath+'80Tr_G.model',map_location=device)
prTr_emo_model = torch.load('models/bestReTr_emo_classifier_seq.model',map_location=device)
prTr_CE = torch.load('models/bestpreTr_CE.model',map_location=device)

G.eval(); prTr_emo_model.eval(); prTr_CE.eval()

random.seed(10)
datalist = random.sample(datalist,10)

up_b, up_a = signal.butter(3, 1/(25/2), btype='low', analog=False)
lo_b, lo_a = signal.butter(3, 0.5/(25/2), btype='low', analog=False)

# eg_ratio = []
agree = []
for idx in range(len(datalist)):
    path = datalist[idx]
    mel1 = torch.load(path + 'mel1.pt'); mfcc1 = torch.load(path + 'mfcc1.pt')
    mel2 = torch.load(path + 'mel2.pt'); mfcc2 = torch.load(path + 'mfcc2.pt')
    mel3 = torch.load(path + 'mel3.pt'); mfcc3 = torch.load(path + 'mfcc3.pt')
    mel4 = torch.load(path + 'mel4.pt'); mfcc4 = torch.load(path + 'mfcc4.pt')
    
    mfcc1 = (torch.tensor(mfcc1) - mfcc_mean) / mfcc_std
    mfcc2 = (torch.tensor(mfcc2) - mfcc_mean) / mfcc_std
    mfcc3 = (torch.tensor(mfcc3) - mfcc_mean) / mfcc_std
    mfcc4 = (torch.tensor(mfcc4) - mfcc_mean) / mfcc_std
    
    mel1 = torch.tensor(mel1).unsqueeze(0).unsqueeze(0); mel1 = (mel1 - mel_mean) / mel_std
    mel2 = torch.tensor(mel2).unsqueeze(0).unsqueeze(0); mel2 = (mel2 - mel_mean) / mel_std
    mel3 = torch.tensor(mel3).unsqueeze(0).unsqueeze(0); mel3 = (mel3 - mel_mean) / mel_std
    mel4 = torch.tensor(mel4).unsqueeze(0).unsqueeze(0); mel4 = (mel4 - mel_mean) / mel_std
    
    mel = torch.vstack((mel1,mel2,mel3,mel4)).float()
    mfcc = torch.vstack((mfcc1,mfcc2,mfcc3,mfcc4)).float()
    
    with torch.no_grad():
        mfcc = mfcc.to(device)
        mel = mel.to(device)
        lab_emo_sp, feat_emo = prTr_emo_model(mfcc)
        noise = torch.rand((4,512*5,30)) * 0.01
        pred_kp = G(mel,feat_emo,noise)
        if isinstance(pred_kp, tuple):
            pred_kp = pred_kp[0]
            
    #     emo_pred = prTr_CE(pred_kp)
    #     x = torch.argmax(emo_pred,dim=1); y = torch.argmax(lab_emo_sp,dim=1)
        
    # agree.append((x==y).sum().cpu().numpy())
    # eg_ratio.append(((pred_kp.mean().abs())/(pred_kp.abs().mean())).cpu().numpy())
    
    pred_kp = torch.vstack((pred_kp[0,5:25,:],pred_kp[1,5:25,:],pred_kp[2,5:25,:],pred_kp[3,5:25,:]))
    
    # post processing here?
    # resample to 25 fps
    pred_kp = signal.resample_poly(pred_kp.cpu(), 5, 6, padtype='smooth')
    pred_kp = pred_kp.reshape((-1,68,2))
    
    # plt.plot(-pred_kp[:,62,1],'b.') # upper lip inside
    # plt.plot(-pred_kp[:,66,1],'r.') # lower lip inside
    
    ##############
    # temp = signal.filtfilt(up_b, up_a, pred_kp[:,62,1])
    # temp = pred_kp[:,62,1] - temp
    # temp -= temp.max()
    # plt.plot(-temp,'b')
    
    # temp = signal.filtfilt(lo_b, lo_a, pred_kp[:,66,1])
    # temp = pred_kp[:,66,1] - temp
    # temp -= temp.min()
    # plt.plot(-temp,'r')
    
    # for up_idx in [49,50,51,52,53,61,62,63]:
    #     temp = signal.filtfilt(up_b, up_a, pred_kp[:,up_idx,1])
    #     pred_kp[:,up_idx,1] -= temp
    #     pred_kp[:,up_idx,1] -= pred_kp[:,up_idx,1].max()
    # for lo_idx in [55,56,57,58,59,65,66,67]:
    #     temp = signal.filtfilt(lo_b, lo_a, pred_kp[:,lo_idx,1])
    #     pred_kp[:,lo_idx,1] -= temp
    #     pred_kp[:,lo_idx,1] -= pred_kp[:,lo_idx,1].min()
    #     # pred_kp[:,lo_idx,1] *= 2
    
    ##############
    # pos = (signal.argrelmax(pred_kp[:,62,1], order=8))[0]
    # inter = interpolate.interp1d(pos, pred_kp[pos,62,1])
    # base = np.zeros_like(pred_kp[:,62,1])
    # base[pos[0]:pos[-1]+1] = inter(range(pos[0],pos[-1]+1))
    # base[:pos[0]] = base[pos[0]]; base[pos[-1]:] = base[pos[-1]]
    # temp = pred_kp[:,62,1] - base
    # plt.plot(-temp,'b')
    
    # pos = (signal.argrelmin(pred_kp[:,66,1], order=8))[0]
    # inter = interpolate.interp1d(pos, pred_kp[pos,66,1])
    # base = np.zeros_like(pred_kp[:,66,1])
    # base[pos[0]:pos[-1]+1] = inter(range(pos[0],pos[-1]+1))
    # base[:pos[0]] = base[pos[0]]; base[pos[-1]:] = base[pos[-1]]
    # temp = pred_kp[:,66,1] - base
    # plt.plot(-temp,'r')
    
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
    
    ##############
    # plt.show()
    
    pred_kp = torch.as_tensor(pred_kp)
    torch.save(pred_kp,path+'pred_kp.pt')

# print(np.mean(np.array(eg_ratio)))
# print(np.sum(np.array(agree))/200)

# exit()
   
os.chdir('../../bilayer-model/examples/')
# os.getcwd()
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

for idx in range(len(datalist)):
    path = datalist[idx]
    kp = torch.load(path + 'pred_kp.pt')
    
    kp = kp + kp_init
    l = kp.min(dim=0,keepdim=True).values.min(dim=1,keepdim=True).values
    L = kp.max(dim=0,keepdim=True).values.max(dim=1,keepdim=True).values
    kp = (kp - l) / (L - l)
    kp = kp * 224 + (256-224)/2
    
    raw_path = path.split('/')
    raw_path[8] = 'mp4'
    raw_path = '/'.join(raw_path)
    
    raw_path += os.listdir(raw_path)[0]
    (frame,audio,fps) = read_video(raw_path)
    assert fps['audio_fps'] == 16000
    
    write_jpeg(frame[0].permute(2,0,1),'source.jpg')
    source_img = np.asarray(Image.open('source.jpg'))
    
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
    
    path = path.split('/')
    name = testpath + path[-3] + path[-2] + '.mp4'
    # 6400:80000-6400 for 48000
    # 2133:22400+2133 for 16000
    # 2666:42667+2666 when starting at frame 5 and going for 80 frames
    write_video(name, op, fps=25, video_codec='libx264', audio_array=audio[:,2666:42667+2666], audio_fps=16000, audio_codec='aac')

    if idx%10 == 0: print(idx)