import torch
import os
import pickle as pkl

print(os.getcwd())
x = pkl.load(open('split.pkl','rb'))['train']
print(x.len)
x = torch.load('/usr/stud/dasd/workspace/processed/id00016/0cLseJLGw2o/face_00002.mp4')
