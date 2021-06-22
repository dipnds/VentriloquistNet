import torch
from torch.utils.data import Dataset
import os
import pickle as pkl

class prep(Dataset):
    
    def __init__(self, path, split):
        
        if not os.path.isfile('../datalist_turner_'+split+'.pkl'):

            person_list = pkl.load(open('../split_mead.pkl','rb'))[split]
            datalist = []
            for person in person_list:
                for emo in os.listdir(path+person):
                    for level in os.listdir(path+person+'/'+emo):
                        for utter in os.listdir(path+person+'/'+emo+'/'+level):
                            temp = os.listdir(path+person+'/'+emo+'/'+level+'/'+utter)
                            try:
                                temp.remove('kp_seq.pt')
                                try:
                                    temp.remove('mel.pt')
                                except:
                                    print('Missing mel : ',path+person+'/'+emo+'/'+level+'/'+utter)
                                try:
                                    temp.remove('mfcc.pt')
                                except:
                                    print('Missing mfcc : ',path+person+'/'+emo+'/'+level+'/'+utter)

                                temp.sort()
                                datalist.append((path+person+'/'+emo+'/'+level+'/'+utter+'/',temp))
                            except:
                                print('Missing front : ',path+person+'/'+emo+'/'+level+'/'+utter)
            self.datalist = datalist
            pkl.dump(datalist,open('../datalist_turner_'+split+'.pkl','wb'))

        else: self.datalist = pkl.load(open('../datalist_turner_'+split+'.pkl','rb'))
        
        self.label_dict = {'down_kp.pt':torch.tensor([0,-1]), 'top_kp.pt':torch.tensor([0,1]),
                           'left_30_kp.pt':torch.tensor([-1/3,0]), 'right_30_kp.pt':torch.tensor([1/3,0]),
                           'left_60_kp.pt':torch.tensor([-2/3,0]), 'right_60_kp.pt':torch.tensor([2/3,0])
                            }
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        
        path, pose_list = self.datalist[idx]
        front = torch.load(path+'kp_seq.pt')[0]
        front = front.type(torch.float)
        front = front - front.mean(dim=0); front = front / front.std(dim=0)
        front = front.flatten()
        
        ip = []; target = []
        for filename in pose_list:
            if filename == 'left_30_kp.pt' or filename == 'right_30_kp.pt':
                gt = torch.load(path+filename)
                gt = gt.type(torch.float)
                gt = gt - gt.mean(dim=0); gt = gt / gt.std(dim=0)
                gt = gt.flatten()
                target.append(gt)
                
                label = self.label_dict[filename]
                temp = torch.cat((front,label))
                ip.append(temp)
        
        ip = torch.stack(ip); target = torch.stack(target)
        ip = torch.nn.functional.pad(ip,(0,0,0,2 - ip.shape[0]),"constant",0)
        target = torch.nn.functional.pad(target,(0,0,0,2 - target.shape[0]),"constant",0)
                
        return ip, target
