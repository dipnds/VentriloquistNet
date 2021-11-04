import pickle as pkl
import os
import torch
import numpy as np

path = '/usr/stud/dasd/storage/user/vox2/dev/processed/'

for split in ['train', 'eval']:
    
    ME_ldmk = torch.zeros((68,3)); VAR_ldmk = torch.zeros((68,3))
    ME_mel = torch.zeros((68,3)); VAR_mel = torch.zeros((68,3))
    count = 1
    
    person_list = pkl.load(open('../split_vox2.pkl','rb'))[split]
    datalist = []
    
    for person in person_list[0:2]:
        for utter in os.listdir(path+person):
               
            temp = os.listdir(path+person+'/'+utter)
            if any('ldmk3d' in s for s in temp) and any('melframe' in s for s in temp):
                flag = 0
                
                try:
                    for i,s in enumerate(temp):
                        if 'ldmk3d' in s:
                            ldmk = torch.load(path+person+'/'+utter+'/'+s)
                            
                            facing_all = []
                            for frame in range(len(ldmk)):
                                v_Leye = ldmk[frame][36,:] - ldmk[frame][30,:]
                                v_Reye = ldmk[frame][45,:] - ldmk[frame][30,:]
                            
                                facing = np.cross(v_Leye,v_Reye)
                                facing[1] = 0 # keep x-z projection, remove y component
                                facing = facing/np.linalg.norm(facing)
                                facing_all.append(facing)
                                
                            facing = np.mean(facing_all,axis=0)
                            
                            front = np.array([0, 0, 1])
                            around = np.cross(facing,front)
                            angle = np.dot(facing,front)
                            magnitude = np.linalg.norm(around)
                            I = np.identity(3)
                            skewsymM = '{} {} {}; {} {} {}; {} {} {}'.format(0,-around[2],around[1],around[2],0,-around[0],-around[1],around[0],0)
                            skewsymM = np.matrix(skewsymM)
                            R = I + skewsymM + np.matmul(skewsymM,skewsymM) * ((1-angle)/(magnitude**2))
                            R = torch.tensor(np.expand_dims(R,axis=0)).expand(68,-1,-1)

                            for frame in range(len(ldmk)):
                                temp = torch.tensor(np.expand_dims(ldmk[frame],axis=-1))
                                turned = torch.bmm(R.double(),temp.double())
                                ldmk[frame] = turned
                                                            
                            ldmk = torch.stack(ldmk)
                            
                            me = torch.mean(ldmk,axis=0).squeeze(-1)
                            ME_ldmk = ME_ldmk*((count-1)/count) + me/count
                            var = torch.mean(ldmk**2,axis=0).squeeze(-1)
                            VAR_ldmk = VAR_ldmk*((count-1)/count) + var/count
                            count += 1
                    
                    flag = 1
                    # datalist.append(path+person+'/'+utter+'/')
                except: print('Landmark Error ************** ', person, utter)
                
                try:
                    for i,s in enumerate(temp):
                        if 'melframe' in s:
                            mel = torch.load(path+person+'/'+utter+'/'+s)
                            
                            print(mel.shape)
                            
                            me = torch.mean(ldmk,axis=0).squeeze(-1)
                            ME_mel = ME_mel*((count-1)/count) + me/count
                            var = torch.mean(ldmk**2,axis=0).squeeze(-1)
                            VAR_mel = VAR_mel*((count-1)/count) + var/count
                            count += 1
                            
                    if flag == 1: datalist.append(path+person+'/'+utter+'/')
                except: print('Spectrogram Error ************** ', person, utter)
                                        
    pkl.dump(datalist,open('../datalist_vox2_'+split+'.pkl','wb'))
    