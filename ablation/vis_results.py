from torch.utils.data import DataLoader
import torch
import os, tqdm
import numpy as np
import csv
from skimage import io
import matplotlib.pyplot as plt

from dataprep import prep

del_mode = False
kp_mode = 'norm'
model = torch.load('models/bestEv_network3_'+kp_mode+'kp.model')

device = torch.device('cuda:0')
path = '/media/deepan/Backup/thesis/dataset_voxceleb/'
batch_size = 4
ev_set = prep(path+'triplets/','train')#, del_mode=del_mode, kp_mode=kp_mode)
ev_loader = DataLoader(ev_set,batch_size=batch_size,shuffle=False,num_workers=0)

norm = torch.load(path+'triplets/norm.pt')

keypointsT = np.zeros((0,68,2)); keypointsP = np.zeros((0,68,2)); keypoints0 = np.zeros((0,68,2))
model.eval()
with torch.no_grad():
    ev_batch = tqdm.tqdm(enumerate(ev_loader),total=len(ev_loader))
    
    for batch, triplet in ev_batch:
        key0 = triplet['key0']; keyT = triplet['keyT']; mel = triplet['mel']; orienT = triplet['orienT']
        key0 = key0.to(device); keyT = keyT.to(device); mel = mel.to(device); orienT = orienT.to(device)
        keyP = model(key0,mel,orienT)
        
        keyP = torch.reshape(keyP,(-1,68,2))
        keyP = keyP.detach().cpu()
        keyP = keyP * ev_set.std_kp + ev_set.mean_kp
        keyP = keyP.numpy()
        keypointsP = np.concatenate((keypointsP,keyP),axis=0)
        
        keyT = torch.reshape(keyT,(-1,68,2))
        keyT = keyT.detach().cpu()
        keyT = keyT * ev_set.std_kp + ev_set.mean_kp
        keyT = keyT.numpy()
        keypointsT = np.concatenate((keypointsT,keyT),axis=0)
        
        key0 = torch.reshape(key0,(-1,68,2))
        key0 = key0.detach().cpu()
        key0 = key0 * ev_set.std_kp + ev_set.mean_kp
        key0 = key0.numpy()
        keypoints0 = np.concatenate((keypoints0,key0),axis=0)
        
count = 0
with open('vox1_meta.csv') as match_key:
    csv_reader = csv.reader(match_key, delimiter='\t')
    split = 1211
    for i,row in enumerate(csv_reader):
        if i>100 and i<=112:
        # if i>split:# and i<=1214:
            file_face = path+'unzippedIntervalFaces/data/'+row[1]+'/1.6/'
            vid_list = os.listdir(file_face); vid_list.sort()
            file_face = file_face+vid_list[0]+'/1/'
            utter_list = os.listdir(file_face); utter_list.sort()
            
            for j,face in enumerate(utter_list[1:11]):
                img0 = io.imread(file_face+face)
                imgT = io.imread(file_face+utter_list[j+2])
                
                key0 = keypoints0[count,:,:]
                keyT = keypointsT[count,:,:]
                keyP = keypointsP[count,:,:]
                keyD = np.copy(key0)
                count += 1
                
                key0[:,0] = key0[:,0] * ((img0.shape[1])/2) + (img0.shape[1]/2)
                key0[:,1] = key0[:,1] * ((img0.shape[0])/2) + (img0.shape[0]/2)
                
                keyT[:,0] = keyT[:,0] * ((imgT.shape[1])/2) + (imgT.shape[1]/2)
                keyT[:,1] = keyT[:,1] * ((imgT.shape[0])/2) + (imgT.shape[0]/2)
                
                keyP[:,0] = keyP[:,0] * ((imgT.shape[1])/2) + (imgT.shape[1]/2)
                keyP[:,1] = keyP[:,1] * ((imgT.shape[0])/2) + (imgT.shape[0]/2)
                
                keyD[:,0] = keyD[:,0] * ((imgT.shape[1])/2) + (imgT.shape[1]/2)
                keyD[:,1] = keyD[:,1] * ((imgT.shape[0])/2) + (imgT.shape[0]/2)
                
                plt.figure(count)
                
                # plt.subplot(121)
                # plt.imshow(img0)
                # plt.scatter(key0[:,0], key0[:,1], 0.5, c='red')
                
                # plt.subplot(122)
                imgT = imgT[:,:,0]
                imgT[:,:] = 1
                plt.imshow(imgT,vmin=0,vmax=1,cmap='gray')
                plt.plot(keyT[0:17,0], keyT[0:17,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyT[17:22,0], keyT[17:22,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyT[22:27,0], keyT[22:27,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyT[27:31,0], keyT[27:31,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyT[31:36,0], keyT[31:36,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyT[36:42,0], keyT[36:42,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyT[42:48,0], keyT[42:48,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyT[48:,0], keyT[48:,1], 'ro-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[0:17,0], keyP[0:17,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[17:22,0], keyP[17:22,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[22:27,0], keyP[22:27,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[27:31,0], keyP[27:31,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[31:36,0], keyP[31:36,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[36:42,0], keyP[36:42,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[42:48,0], keyP[42:48,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.plot(keyP[48:,0], keyP[48:,1], 'bo-', linewidth=0.5, markersize=0.5)
                plt.scatter(keyD[:,0], keyD[:,1], 0.5, c='black')
                
                plt.show()
                
                
                
                
                
                
                
                