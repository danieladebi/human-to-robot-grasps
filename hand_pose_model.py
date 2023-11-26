import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights
from collections import OrderedDict

def load_from_file(filepath):
    model = HandPoseModel()
    state_dict = torch.load(filepath)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

class HandPoseModel(nn.Module):
    def __init__(self):
        super(HandPoseModel, self).__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(2048, 21*3)
        
    def forward(self, x):
        return self.base_model(x)
