# www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/alexnet_face_fer_bn_dag.py
import torch
import torch.nn as nn


class Alexnet_face_fer_bn_dag(nn.Module):

    def __init__(self):
        super(Alexnet_face_fer_bn_dag, self).__init__()
        self.meta = {'mean': [131.09375, 103.88607788085938, 91.47599792480469],
                     'std': [1, 1, 1],
                     'imageSize': [227, 227, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[11, 11], stride=(4, 4))
        self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(1, 1), padding=(2, 2), groups=2)
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), groups=2)
        self.bn4 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), groups=2)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(256, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn6 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn7 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=4096, out_features=7, bias=True)

    def forward(self, data):
        data = self.conv1(data)
        data = self.bn1(data)
        data = self.relu1(data)
        data = self.pool1(data)
        data = self.conv2(data)
        data = self.bn2(data)
        data = self.relu2(data)
        data = self.pool2(data)
        data = self.conv3(data)
        data = self.bn3(data)
        data = self.relu3(data)
        data = self.conv4(data)
        data = self.bn4(data)
        data = self.relu4(data)
        data = self.conv5(data)
        data = self.bn5(data)
        data = self.relu5(data)
        return data

def alexnet_face_fer_bn_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Alexnet_face_fer_bn_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
