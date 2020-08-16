import torchvision
from fastai2.vision.all import *

def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": xresnet18(pretrained=pretrained),
        "resnet34": xresnet34(pretrained=pretrained),
        "resnet50": xresnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
