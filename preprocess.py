import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io
import csv

import face_alignment
import librosa
import librosa.display

path_root = '/media/deepan/Backup/thesis/dataset_voxceleb/'
path_face = path_root+'unzippedIntervalFaces/data/'
path_speech = path_root+'vox1_dev_wav/wav/'
path_out = path_root+'triplets/'

with open('vox1_meta.csv') as match_key:
    
    csv_reader = csv.reader(match_key, delimiter='\t')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                       device='cuda', face_detector='sfd')
    avg_mel = torch.zeros(128); std_mel = torch.zeros(128); count_mel = 1
    avg_kp = torch.zeros(2); std_kp = torch.zeros(2); count_kp = 1
    avg_face = torch.zeros(3); std_face = torch.zeros(3); count_face = 1
    split = 1211 # eval
    
    for i,row in enumerate(csv_reader):
        if (i==0) or (i>269 and i<310): # exclude header row and test split
            pass
        else: # elif i<5:
            
            file_face = path_face+row[1]+'/1.6/'
            vid_list = os.listdir(file_face); vid_list.sort()
            file_face = file_face+vid_list[0]+'/1/' 
            utter_list = os.listdir(file_face); utter_list.sort()
            
            file_speech = path_speech+row[0]+'/'+vid_list[0]+'/00001.wav'
            speech, fs = librosa.load(file_speech)
            speech = speech[:int(3*fs)]
            mel = librosa.feature.melspectrogram(speech, fs, n_fft=512, hop_length=128)
            mel = librosa.power_to_db(mel)#, ref=np.max)
            
            keypoints = []; faces = []
            for name in utter_list[1:12]:
                
                face = io.imread(file_face+name)
                kp = torch.tensor(fa.get_landmarks(face)[0])
                face = torch.tensor(face,dtype=torch.float)
                
                # plt.imshow(face)
                # plt.scatter(kp[17:27,0], kp[17:27,1], 5, c='red')
                # plt.show()
                
                keypoints.append(kp); faces.append(face)
                
                if i<=split:
                    kp = kp - (kp.max(dim=0).values+kp.min(dim=0).values)/2
                    kp = kp / ((kp.max(dim=0).values-kp.min(dim=0).values)/2)
                    
                    avg_kp = avg_kp*((count_kp-1)/count_kp) + (kp.mean(dim=0))/count_kp
                    std_kp = std_kp*((count_kp-1)/count_kp) + (kp.pow(2).mean(dim=0))/count_kp
                    count_kp += 1
                    
                    avg_face = avg_face*((count_face-1)/count_face) + (face.mean(dim=[0,1]))/count_face
                    std_face = std_face*((count_face-1)/count_face) + (face.pow(2).mean(dim=[0,1]))/count_face
                    count_face += 1
            
            if i>split:
                path = path_out+'eval/'
            else:
                path = path_out+'train/'
                
                avg_mel = avg_mel*((count_mel-1)/count_mel) + torch.tensor((np.mean(mel,axis=1))/count_mel)
                std_mel = std_mel*((count_mel-1)/count_mel) + torch.tensor((np.mean(mel**2,axis=1))/count_mel)
                count_mel += 1
                
            interval = ((mel.shape[1])/3)*(6/25)
            for j in range(1,11):
                frame_mel = mel[:,round(j*interval)-4:round((j+1)*interval)+5]
                sample = {'mel':torch.tensor(frame_mel),
                          'key0':keypoints[j-1], 'keyT':keypoints[j],
                          'face0':faces[j-1], 'faceT':faces[j]}
                torch.save(sample, path+str(row[0])+'_'+str(j)+'.pt')
            
            # plt.figure(i)
            # librosa.display.specshow(mel[:,:], y_axis='mel', x_axis='time')#, fmax=8000)
            # plt.colorbar(format='%+2.0f dB')
            
        if i%50==0: print(i)
        
    std_mel = std_mel - avg_mel.pow(2); std_mel = torch.sqrt(std_mel)
    std_kp = std_kp - avg_kp.pow(2); std_kp = torch.sqrt(std_kp)
    std_face = std_face - avg_face.pow(2); std_face = torch.sqrt(std_face)
    norm = {'mean_mel':avg_mel, 'std_mel':std_mel,
            'mean_kp':avg_kp, 'std_kp':std_kp,
            'mean_face':avg_face, 'std_face':std_face}
    torch.save(norm, 'norm.pt')