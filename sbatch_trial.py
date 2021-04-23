import os
import torch

print(torch.cuda.is_available())
print(os.getcwd())

face = torch.load('/storage/user/dasd/vox2/dev/processed/id00012/2DLq_Kkc1r8/face_00017.pt')
print(face.shape)
sketch = torch.load('/storage/user/dasd/vox2/dev/processed/id00012/2DLq_Kkc1r8/sketch_00017.pt')
print(sketch['sketch'].shape)