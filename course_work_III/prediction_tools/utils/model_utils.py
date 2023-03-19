import torchvision
from .models.resnet import resnet1d18
from .models.densenet import densenet1d121


def get_model(cfg):

    assert cfg.model in ["resnet1d18", "resnet2d18", "densenet1d121", "densenet2d121"]
    if cfg.model == "resnet2d18":
        model = torchvision.models.resnet.resnet18(num_classes=1, weights=None)
    elif cfg.model == "densenet2d121":
        model = torchvision.models.densenet.densenet121(num_classes=1, weights=None)
    elif cfg.model == "resnet1d18":
        model = resnet1d18(num_classes=1, input_channels=12)
    else:
        model = densenet1d121(num_classes=1, input_channels=12)

    return model
