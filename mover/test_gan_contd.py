import os
import torch
import pickle as pkl
datalist = pkl.load(open('../datalist_vox2mead_test.pkl','rb'))

######
from torchvision.io import read_video, write_jpeg, write_video
from torchvision.transforms import Resize
import sys
import random
from PIL import Image
import numpy as np

os.chdir('../../bilayer-model/examples/')
os.getcwd()

sys.path.append('../')
testpath = '/usr/stud/dasd/storage/user/syncnet_target/'

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

random.seed(100)
datalist = random.sample(datalist,100)

resi = Resize(224)

for idx in range(len(datalist[0:2])):
    path = datalist[idx]
    kp = torch.load(path + 'pred_kp.pt')
    
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
    write_video(name, op, fps=25, video_codec='libx264', audio_array=audio[:,2133:22400+2133], audio_fps=16000, audio_codec='aac')

    if idx%10 == 0: print(idx)