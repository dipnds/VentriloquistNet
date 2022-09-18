import os
import torch
from torchvision.io import read_video
import tqdm
import pickle as pkl
import librosa

import audio

path_root = '/storage/user/dasd/vox2/test/'
path_in = path_root+'mp4/'
path_out = path_root+'processed_mead/'

def audio_processing_loop(person_list, path_in, path_out):
    
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
                        (_,speech,fps) = read_video(path_in+person+'/'+vid+'/'+utter)
                        speech = speech[0].numpy(); fps = fps['audio_fps']
                        if fps != 48000:
                            speech = librosa.resample(speech,fps,48000)
                            fps = 48000
                        
                        start = 0; speech1 = speech[start:start+48000]
                        start = start+48000-16000; speech2 = speech[start:start+48000]
                        start = start+48000-16000; speech3 = speech[start:start+48000]
                        start = start+48000-16000; speech4 = speech[start:start+48000]
                        # print(start+48000)
                        
                        if not os.path.isdir(path_out+person+'/'+vid+'/'):
                            os.makedirs(path_out+person+'/'+vid+'/')
                        
                        mfcc1 = librosa.feature.mfcc(speech1,sr=48000,n_fft=2048,hop_length=522,n_mfcc=30)
                        mel1 = audio.melspectrogram(speech1)
                        torch.save(mel1,path_out+person+'/'+vid+'/'+'mel1.pt')
                        torch.save(mfcc1,path_out+person+'/'+vid+'/'+'mfcc1.pt')
                        
                        mfcc2 = librosa.feature.mfcc(speech2,sr=48000,n_fft=2048,hop_length=522,n_mfcc=30)
                        mel2 = audio.melspectrogram(speech2)
                        torch.save(mel2,path_out+person+'/'+vid+'/'+'mel2.pt')
                        torch.save(mfcc2,path_out+person+'/'+vid+'/'+'mfcc2.pt')
                        
                        mfcc3 = librosa.feature.mfcc(speech3,sr=48000,n_fft=2048,hop_length=522,n_mfcc=30)
                        mel3 = audio.melspectrogram(speech3)
                        torch.save(mel3,path_out+person+'/'+vid+'/'+'mel3.pt')
                        torch.save(mfcc3,path_out+person+'/'+vid+'/'+'mfcc3.pt')
                        
                        mfcc4 = librosa.feature.mfcc(speech4,sr=48000,n_fft=2048,hop_length=522,n_mfcc=30)
                        mel4 = audio.melspectrogram(speech4)
                        torch.save(mel4,path_out+person+'/'+vid+'/'+'mel4.pt')
                        torch.save(mfcc4,path_out+person+'/'+vid+'/'+'mfcc4.pt')
                        
                    except:
                        print('Audio Error:', person+'/'+vid)
                        
file_list = pkl.load(open('split_vox2.pkl','rb'))['test']
print(len(file_list))
audio_processing_loop(file_list, path_in, path_out)