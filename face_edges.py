import torch
import numpy as np
import os 
from scipy.optimize import curve_fit
import warnings

path = '/media/deepan/Backup/thesis/dataset_voxceleb/triplets/train/'
file_list = os.listdir(path)

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

for file in file_list:
    tens = torch.load(path+file)
    if 'sketch0' not in tens:
    
        temp = tens['face0']; temp = temp.type(torch.uint8); tens['face0'] = temp
        temp = tens['faceT']; temp = temp.type(torch.uint8); tens['faceT'] = temp
        
        kp = tens['key0'].numpy().astype(int)
        lim = list(tens['face0'].shape[0:2]); kp[:,1] = np.clip(kp[:,1],0,lim[0]-1); kp[:,0] = np.clip(kp[:,0],0,lim[1]-1)
        sketch0 = torch.zeros(tens['face0'].shape[0:2]).type(torch.uint8)
        curve_x, curve_y = interpPoints(kp[0:5,0],kp[0:5,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[4:9,0],kp[4:9,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[8:13,0],kp[8:13,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[12:17,0],kp[12:17,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[17:22,0],kp[17:22,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[22:27,0],kp[22:27,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[27:31,0],kp[27:31,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[31:36,0],kp[31:36,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[36:40,0],kp[36:40,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[39,40,41,36],0],kp[[39,40,41,36],1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[42:46,0],kp[42:46,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[45,46,47,42],0],kp[[45,46,47,42],1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[48:55,0],kp[48:55,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[54,55,56,57,58,59,48],0],kp[[54,55,56,57,58,59,48],1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[60:65,0],kp[60:65,1],lim); sketch0[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[64,65,66,67,60],0],kp[[64,65,66,67,60],1],lim); sketch0[curve_y,curve_x] = 1
        
        kp = tens['keyT'].numpy().astype(int)
        lim = list(tens['faceT'].shape[0:2]); kp[:,1] = np.clip(kp[:,1],0,lim[0]-1); kp[:,0] = np.clip(kp[:,0],0,lim[1]-1)
        sketchT = torch.zeros(tens['faceT'].shape[0:2]).type(torch.uint8)
        curve_x, curve_y = interpPoints(kp[0:5,0],kp[0:5,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[4:9,0],kp[4:9,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[8:13,0],kp[8:13,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[12:17,0],kp[12:17,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[17:22,0],kp[17:22,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[22:27,0],kp[22:27,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[27:31,0],kp[27:31,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[31:36,0],kp[31:36,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[36:40,0],kp[36:40,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[39,40,41,36],0],kp[[39,40,41,36],1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[42:46,0],kp[42:46,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[45,46,47,42],0],kp[[45,46,47,42],1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[48:55,0],kp[48:55,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[54,55,56,57,58,59,48],0],kp[[54,55,56,57,58,59,48],1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[60:65,0],kp[60:65,1],lim); sketchT[curve_y,curve_x] = 1
        curve_x, curve_y = interpPoints(kp[[64,65,66,67,60],0],kp[[64,65,66,67,60],1],lim); sketchT[curve_y,curve_x] = 1
        
        tens['sketch0'] = sketch0;tens['sketchT'] = sketchT
        torch.save(tens,path+file)