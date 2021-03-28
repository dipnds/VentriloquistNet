import os
import numpy as np
import torch
from torchvision.io import read_video
import csv
from scipy.optimize import curve_fit
import warnings
import tqdm
import pickle as pkl

import face_alignment

path_root = '/storage/user/dasd/vox2/dev/'
path_in = path_root+'mp4/'
path_out = path_root+'processed/'

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
    
def processing_loop(person_list, path_in, path_out, fa):
    
    person_list = tqdm.tqdm(person_list,total=len(person_list))
    
    for person in person_list:
	person_list.set_description(person)
        if os.path.isdir(path_in+person):
            vid_list = os.listdir(path_in+person)
            for vid in vid_list:
                utter_list = os.listdir(path_in+person+'/'+vid)
                # limit to 1
                lim = 1
                if len(utter_list) > lim: utter_list = utter_list[:lim]
                
                for utter in utter_list:
                    
                    try:
                        (frame,_,_) = read_video(path_in+person+'/'+vid+'/'+utter,
                                                 0,1.59,pts_unit='sec') # 32+8 frames
                        frame = frame.permute(0,3,1,2)
                        corner = [np.array([0,0,frame.shape[2],frame.shape[3]])]
                        bbox = [corner]*frame.shape[0]
                        video_kp = fa.get_landmarks_from_batch(frame,bbox)
                        
                        sketch = []
                        for frame_kp in video_kp:
                            frame_kp = frame_kp.astype(int)
                            sketch.append(kp2sketch(frame_kp,frame.shape[2],frame.shape[3]))
                        sketch = torch.stack(sketch); sketch = torch.unsqueeze(sketch,1)
                        video_kp = torch.tensor(np.stack(video_kp)); video_kp = video_kp.type(torch.uint8)
                        sample = {'sketch':sketch, 'kp':video_kp}
                        
                        if not os.path.isdir(path_out+person+'/'+vid+'/'):
                            os.makedirs(path_out+person+'/'+vid+'/')    
                        torch.save(frame,path_out+person+'/'+vid+'/'+'face_'+utter[:-4]+'.pt')
                        torch.save(sample,path_out+person+'/'+vid+'/'+'sketch_'+utter[:-4]+'.pt')
                        
                    except:
                        print(person+'/'+vid)
                        
    return 0
                    
# make a list of person IDs in the target subset
with open('vox2_meta.csv') as meta:
    csv_reader = csv.reader(meta, delimiter=',')
    next(csv_reader, None) # skip header row
    id_list = {'dev':[], 'test':[]}
    for i,row in enumerate(csv_reader):
        id_list[row[-1].strip()].append(row[0].strip())

# train eval split
file_list = id_list['dev']
if not os.path.isfile('split.pkl'):
    np.random.seed(0)
    idx = np.random.choice(len(id_list['dev']),len(id_list['test']),replace=False)
    id_list['eval'] = [id_list['dev'][i] for i in idx]
    id_list['train'] = list(set(id_list['dev']) - set(id_list['eval']))
    del id_list['dev']
    pkl.dump(id_list,open('split.pkl','rb'))

# init face alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,
                                  device='cuda',face_detector='blazeface') # default 'sfd'

# loop over subsets
a = 0; b = 600
print(a,b)
processing_loop(file_list[a:b], path_in, path_out, fa)
# processing_loop(id_list['test'], path_in, path_out, fa)
