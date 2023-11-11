import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights

class HandPoseModel(nn.Module):
    def __init__(self):
        super(HandPoseModel, self).__init__()        
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(2048, 21*3)

    def forward(self, x):
        return self.base_model(x)    

