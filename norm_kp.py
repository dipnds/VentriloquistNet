import torch
import os

path = '/media/deepan/Backup/thesis/dataset_voxceleb/triplets/train/'
file_list = os.listdir(path)

avg_kp = torch.zeros(2); std_kp = torch.zeros(2); count_kp = 1
avg_delta = torch.zeros(68,2); std_delta = torch.zeros(68,2); count_delta = 1


for file in file_list:
    kp = torch.load(path+file)['key0']
    # kp = kp - (kp.max(dim=0).values+kp.min(dim=0).values)/2
    # kp = kp / ((kp.max(dim=0).values-kp.min(dim=0).values)/2)
    
    avg_kp = avg_kp*((count_kp-1)/count_kp) + (kp.mean(dim=0))/count_kp
    std_kp = std_kp*((count_kp-1)/count_kp) + (kp.square().mean(dim=0))/count_kp
    count_kp += 1
    
    if file[-5:-3] == '10':
        kp = torch.load(path+file)['keyT']
        # kp = kp - (kp.max(dim=0).values+kp.min(dim=0).values)/2
        # kp = kp / ((kp.max(dim=0).values-kp.min(dim=0).values)/2)
        
        avg_kp = avg_kp*((count_kp-1)/count_kp) + (kp.mean(dim=0))/count_kp
        std_kp = std_kp*((count_kp-1)/count_kp) + (kp.square().mean(dim=0))/count_kp
        count_kp += 1
        
    del_kp = torch.load(path+file)['keyT']
    # del_kp = del_kp - (del_kp.max(dim=0).values+del_kp.min(dim=0).values)/2
    # del_kp = del_kp / ((del_kp.max(dim=0).values-del_kp.min(dim=0).values)/2)
    del_kp -= kp
    avg_delta = avg_delta*((count_delta-1)/count_delta) + (del_kp)/count_delta
    std_delta = std_delta*((count_delta-1)/count_delta) + (del_kp.square())/count_delta
    count_delta += 1
    
    if count_delta%1000 == 0:
        print(count_delta)
        
std_kp = std_kp - avg_kp.square(); std_kp = torch.sqrt(std_kp)
std_delta = std_delta - avg_delta.square(); std_delta = torch.sqrt(std_delta)

old = torch.load('/media/deepan/Backup/thesis/dataset_voxceleb/triplets/norm.pt')
norm = {'mean_mel':old['mean_mel'],'std_mel':old['std_mel']}
norm['mean_kp'] = avg_kp; norm['std_kp'] = std_kp
norm['mean_delta'] = avg_delta; norm['std_delta'] = std_delta