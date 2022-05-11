import torchvision


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained), # get resnet 18, with random init
        "resnet50": torchvision.models.resnet50(pretrained=pretrained), #get resnet 50, with random init
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]