import os
import torch
import pickle as pkl
import time

id_list = pkl.load(open('split.pkl','rb'))
id_list = id_list['train'] + id_list['eval']

path = 'processed/'
# T = time.time()

# count = 0
# for person in id_list:
#     if os.path.isdir(path+person):
#         utter_list = os.listdir(path+person)
#         for utter in utter_list:
            
#             t = time.time()
            
#             file_list = os.listdir(path+person+'/'+utter)
#             file_list.sort() # face, sketch
#             face = torch.load(path+person+'/'+utter+'/'+file_list[0])
#             sketch = torch.load(path+person+'/'+utter+'/'+file_list[1])
            
#             for i in range(face.shape[0]):
#                 frame = torch.cat((face[i,:,:,:],sketch['sketch'][i,:,:,:]))
#                 torch.save(frame,path+person+'/'+utter+'/fatch_'+str(i)+'.pt')
            
#             kp_seq = sketch['kp']
#             torch.save(kp_seq,path+person+'/'+utter+'/kp_seq.pt')
            
#             print(time.time() - t)
#             count += 1
            
# print(time.time() - T)
# print(count)

T = time.time()
for person in id_list:
    if os.path.isdir(path+person):
        utter_list = os.listdir(path+person)
        for utter in utter_list:
            file_list = os.listdir(path+person+'/'+utter)
            file_list.sort() # face, sketch
            face = torch.load(path+person+'/'+utter+'/'+file_list[0])
            sketch = torch.load(path+person+'/'+utter+'/'+file_list[-1])
print(time.time() - T)

t = time.time()
for person in id_list:
    if os.path.isdir(path+person):
        utter_list = os.listdir(path+person)
        for utter in utter_list:
            file_list = os.listdir(path+person+'/'+utter)
            file_list.sort() # face, sketch
            face = torch.load(path+person+'/'+utter+'/'+file_list[4])
            sketch = torch.load(path+person+'/'+utter+'/'+file_list[-4])
print(time.time() - t)
