
import torch
import torch.nn as nn


class Vgg19_pt_mcn(nn.Module):

    def __init__(self):
        super(Vgg19_pt_mcn, self).__init__()
        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [224, 224]}
        self.features_0 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_1 = nn.ReLU()
        self.features_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_3 = nn.ReLU()
        self.features_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_5 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_6 = nn.ReLU()
        self.features_7 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_8 = nn.ReLU()
        self.features_9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_10 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_11 = nn.ReLU()
        self.features_12 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_13 = nn.ReLU()
        self.features_14 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_15 = nn.ReLU()
        self.features_16 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_17 = nn.ReLU()
        self.features_18 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_19 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_20 = nn.ReLU()
        self.features_21 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_22 = nn.ReLU()
        self.features_23 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_24 = nn.ReLU()
        self.features_25 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_26 = nn.ReLU()
        self.features_27 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_28 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_29 = nn.ReLU()
        self.features_30 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_31 = nn.ReLU()
        self.features_32 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_33 = nn.ReLU()
        self.features_34 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_35 = nn.ReLU()
        self.features_36 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.classifier_0 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.classifier_1 = nn.ReLU()
        self.classifier_2 = nn.Dropout(p=0.5)
        self.classifier_3 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.classifier_4 = nn.ReLU()
        self.classifier_5 = nn.Dropout(p=0.5)
        self.classifier_6 = nn.Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, data):
        features_0 = self.features_0(data)
        features_1 = self.features_1(features_0)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        features_4 = self.features_4(features_3)
        features_5 = self.features_5(features_4)
        features_6 = self.features_6(features_5)
        features_7 = self.features_7(features_6)
        features_8 = self.features_8(features_7)
        features_9 = self.features_9(features_8)
        features_10 = self.features_10(features_9)
        features_11 = self.features_11(features_10)
        features_12 = self.features_12(features_11)
        features_13 = self.features_13(features_12)
        features_14 = self.features_14(features_13)
        features_15 = self.features_15(features_14)
        features_16 = self.features_16(features_15)
        features_17 = self.features_17(features_16)
        features_18 = self.features_18(features_17)
        features_19 = self.features_19(features_18)
        features_20 = self.features_20(features_19)
        features_21 = self.features_21(features_20)
        features_22 = self.features_22(features_21)
        features_23 = self.features_23(features_22)
        features_24 = self.features_24(features_23)
        features_25 = self.features_25(features_24)
        features_26 = self.features_26(features_25)
        features_27 = self.features_27(features_26)
        features_28 = self.features_28(features_27)
        features_29 = self.features_29(features_28)
        features_30 = self.features_30(features_29)
        # features_31 = self.features_31(features_30)
        # features_32 = self.features_32(features_31)
        # features_33 = self.features_33(features_32)
        # features_34 = self.features_34(features_33)
        # features_35 = self.features_35(features_34)
        # features_36 = self.features_36(features_35)
        # classifier_flatten = features_36.view(features_36.size(0), -1)
        # classifier_0 = self.classifier_0(classifier_flatten)
        # classifier_1 = self.classifier_1(classifier_0)
        # classifier_2 = self.classifier_2(classifier_1)
        # classifier_3 = self.classifier_3(classifier_2)
        # classifier_4 = self.classifier_4(classifier_3)
        # classifier_5 = self.classifier_5(classifier_4)
        # classifier_6 = self.classifier_6(classifier_5)
        return [features_2, features_7, features_12, features_21, features_30]

def vgg19_pt_mcn(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg19_pt_mcn()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
