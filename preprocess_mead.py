import os
import numpy as np
import torch
from torchvision.io import read_video
import tqdm
import pickle as pkl
import subprocess
import librosa

import face_alignment
import audio

path_root = '/storage/user/dasd/mead/'
#path_root = '/media/deepan/Backup/thesis/mead/'
path_in = path_root+'raw/'
path_out = path_root+'processed/'

def video_process(person_list, path_in, path_out, fa):
    
    pov_list = ['left_30','right_30','left_60','right_60','top','down']
    emo_list = ['neutral','angry','contempt','disgusted','fear','happy','sad','surprised']
    
    for person in person_list:
        if os.path.isdir(path_in+person):
            emo_list = tqdm.tqdm(emo_list,total=len(emo_list))
            for emo in emo_list:
                emo_list.set_description(person+'/'+emo)
                
                if emo == 'neutral':
                    lvl_list = ['level_1']
                else: lvl_list = ['level_1', 'level_2', 'level_3']
                for lvl in lvl_list:
                    
                    utter_list = os.listdir(path_in+person+'/video/front/'+emo+'/'+lvl)
                    for utter in utter_list:
                                            
                        try:
                            (frame,_,_) = read_video(path_in+person+'/video/front/'+emo+'/'+lvl+'/'+utter)
                            frame = frame.permute(0,3,1,2)
                            video_kp = fa.get_landmarks_from_batch(frame)
                            video_kp = torch.tensor(np.stack(video_kp)); video_kp = video_kp.type(torch.int16)
                            
                            if not os.path.isdir(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]):
                                os.makedirs(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4])
                            torch.save(video_kp,path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]+'/kp_seq.pt')
                            
                            for pov in pov_list:
                                try:
                                    (frame,_,_) = read_video(path_in+person+'/video/'+pov+'/'+emo+'/'+lvl+'/'+utter,
                                                            0,0.03,pts_unit='sec')
                                    kp = fa.get_landmarks_from_image(frame[0])
                                    kp = torch.tensor(kp[0]).type(torch.int16)
                                    torch.save(kp,path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]+'/'+pov+'_kp.pt')
                                except:
                                    print(person+'/video/'+pov+'/'+emo+'/'+lvl+'/'+utter)
                            
                        except:
                            print(person+'/video/'+emo+'/'+lvl+'/'+utter)
                                                        
    return 0
 
def audio_process_wav2lip(person_list, path_in, path_out):
    
    emo_list = ['neutral','angry','contempt','disgusted','fear','happy','sad','surprised']
    
    for person in person_list:
        if os.path.isdir(path_in+person):
            emo_list = tqdm.tqdm(emo_list,total=len(emo_list))
            for emo in emo_list:
                emo_list.set_description(person+'/'+emo)
                
                if emo == 'neutral':
                    lvl_list = ['level_1']
                else: lvl_list = ['level_1', 'level_2', 'level_3']
                for lvl in lvl_list:
                    
                    utter_list = os.listdir(path_in+person+'/audio/'+emo+'/'+lvl)
                    for utter in utter_list:
                        
                        try:
                            fname = path_in+person+'/audio/'+emo+'/'+lvl+'/'+utter
                            fname_wav = path_in+person+'/audio/'+emo+'/'+lvl+'/'+utter[:-4]+'.wav'
                            command = 'ffmpeg -hide_banner -loglevel error -y -i {} -strict -2 {}'.format(fname, fname_wav)
                            subprocess.call(command, shell=True)
    
                            speech = audio.load_wav(fname_wav, 16000)
                            mel = audio.melspectrogram(speech)
                            mel = torch.tensor(mel)
                            os.remove(fname_wav)
    
                            if not os.path.isdir(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]):
                                    os.makedirs(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4])
                            torch.save(mel,path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]+'/mel.pt')
                        
                        except:
                            print(person+'/audio/'+emo+'/'+lvl+'/'+utter)
                    
def audio_process_emo(person_list, path_in, path_out):
    
    emo_list = ['neutral','angry','contempt','disgusted','fear','happy','sad','surprised']
    
    for person in person_list:
        if os.path.isdir(path_in+person):
            emo_list = tqdm.tqdm(emo_list,total=len(emo_list))
            for emo in emo_list:
                emo_list.set_description(person+'/'+emo)
                
                if emo == 'neutral':
                    lvl_list = ['level_1']
                else: lvl_list = ['level_1', 'level_2', 'level_3']
                for lvl in lvl_list:
                    
                    utter_list = os.listdir(path_in+person+'/audio/'+emo+'/'+lvl)
                    for utter in utter_list:
                        
                        try:
                            fname = path_in+person+'/audio/'+emo+'/'+lvl+'/'+utter
                            fname_wav = path_in+person+'/audio/'+emo+'/'+lvl+'/'+utter[:-4]+'.wav'
                            command = 'ffmpeg -hide_banner -loglevel error -y -i {} -strict -2 {}'.format(fname, fname_wav)
                            subprocess.call(command, shell=True)
    
                            speech,_ = librosa.load(fname_wav, sr=44100, res_type="kaiser_fast")
                            mfcc = librosa.feature.mfcc(speech, sr=44100, n_mfcc=30)
                            mfcc = torch.tensor(mfcc); mfcc = mfcc.unsqueeze(-1)
                            os.remove(fname_wav)
    
                            if not os.path.isdir(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]):
                                    os.makedirs(path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4])
                            torch.save(mfcc,path_out+person+'/'+emo+'/'+lvl+'/'+utter[:-4]+'/mfcc.pt')
                        
                        except:
                            print(person+'/audio/'+emo+'/'+lvl+'/'+utter)
                    

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
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,
                                  # device='cuda',face_detector='blazeface') # default 'sfd'

# loop over subsets
a = 0; b = 3
print(a,b)
# video_process(file_list[a:b], path_in, path_out, fa)
# audio_process_wav2lip(file_list, path_in, path_out)
audio_process_emo(file_list, path_in, path_out)