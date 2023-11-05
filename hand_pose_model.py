import torch
from torch import nn
import torchvision
from torchvision import models

class HandPoseModel(nn.Module):
    def __init__(self):
        super(HandPoseModel, self).__init__()        
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(512, 21*3)


    def forward(self, x):
        return self.base_model(x)

# print(HandPoseModel())
    

