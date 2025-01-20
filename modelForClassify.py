"""
@File    : modelForCar.py
@Author  : GiperHsiue
@Time    : 2024/11/7 18:51
"""
import torch
import torch.nn as nn
import torchvision.models as models
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # 定义模型

    def forward(self, x):
        # 定义前向传播
        return x

# ResNet
# from torchvision.models import ResNet18_Weights
# class Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(Classifier, self).__init__()
#         self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         num_features = self.resnet.fc.in_features
#         # 替换原始的全连接层
#         self.resnet.fc = nn.Identity()  # 用一个恒等映射替换原始的全连接层
#         self.fc1 = nn.Linear(num_features, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.resnet(x)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         return self.fc2(x)

# MobileNetV2
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# class Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(Classifier, self).__init__()
#         self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
#         self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

#     def forward(self, x):
#         return self.model(x)