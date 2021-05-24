import os
import numpy as np
import torch
from torchvision.io import read_video
import tqdm
import pickle as pkl
import time

import face_alignment

path_root = '/storage/user/dasd/mead/'
#path_root = '/media/deepan/Backup/thesis/mead/'
path_in = path_root+'raw/'
path_out = path_root+'processed/'

def video_processing_loop(person_list, path_in, path_out, fa):
    
    pov_list = ['front','left_30','right_30','left_60','right_60','top','down']
    emo_list = ['neutral','angry','contempt','disgusted','fear','happy','sad','surprised']
    
    for person in person_list:
        if os.path.isdir(path_in+person):
            pov_list = tqdm.tqdm(pov_list,total=len(pov_list))
            for pov in pov_list:
                pov_list.set_description(person+'/'+pov)
                for emo in emo_list:
                    
                    if emo == 'neutral':
                        lvl_list = ['level_1']
                    else: lvl_list = ['level_1', 'level_2', 'level_3']
                    for lvl in lvl_list:
                        
                        utter_list = os.listdir(path_in+person+'/video/'+pov+'/'+emo+'/'+lvl)
                        for utter in utter_list:
                                                
                            try:
                                (frame,_,_) = read_video(path_in+person+'/video/'+pov+'/'+emo+'/'+lvl+'/'+utter)
                                frame = frame.permute(0,3,1,2)
                                video_kp = fa.get_landmarks_from_batch(frame)
                                video_kp = torch.tensor(np.stack(video_kp)); video_kp = video_kp.type(torch.int16)
                                
                                if not os.path.isdir(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]):
                                    os.makedirs(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4])
                                torch.save(video_kp,path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]+'/'+pov+'.pt')
                                
                            except:
                                print(person+'/video/'+pov+'/'+emo+'/'+lvl+'/'+utter)
                                                        
    return 0
                    
# make a list of person IDs in the target subset
id_list = {'dev':os.listdir(path_in)}; id_list['dev'].sort()

# train eval split
file_list = id_list['dev']
if not os.path.isfile('split_mead.pkl'):
    np.random.seed(0)
    idx = np.random.choice(len(id_list['dev']),6,replace=False)
    id_list['eval'] = [id_list['dev'][i] for i in idx]
    id_list['train'] = list(set(id_list['dev']) - set(id_list['eval']))
    del id_list['dev']
    pkl.dump(id_list,open('split_mead.pkl','wb'))

# init face alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,
                                  device='cuda',face_detector='blazeface') # default 'sfd'

# loop over subsets
a = 6; b = 8
print(a,b)
video_processing_loop(file_list[a:b], path_in, path_out, fa)
