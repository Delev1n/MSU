from ecglib.models.model_builder import create_model
from ecglib.models.config.model_configs import ResNetConfig
from ecglib.models.architectures.resnet1d import BasicBlock1d


def get_model(cfg):

    configs = ResNetConfig(
        block_type=BasicBlock1d,
        layers=[2, 2, 2, 2],
    )

    model = create_model(
        model_name=["resnet1d18"],
        config=configs,
        pretrained=False,
        pretrained_path=None,
        pathology=["AFIB"],
        leads_count=12,
        num_classes=1,
    )

    if cfg.device == "cuda":
        device = "{}:{}".format(cfg.device, cfg.device_ids[0])
        model.to(device)

    return model
