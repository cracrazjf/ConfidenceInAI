from psychai.nn_builder import ModelSpec, Model, save_config, build_config_dict
from psychai.config import ModelConfig, update_config

def build_cnn(cfg: ModelConfig):
    spec = ModelSpec(image_shape=cfg.image_shape)

    spec.add_layer({
        "type": "cnn",
        "in_channels": cfg.image_shape[0],
        "out_channels": cfg.image_shape[1],
        "kernel_size": 3,
        "num_classes": cfg.num_classes,
        "dropout_p": 0.0,
    }, name="cnn") 

    model = Model(spec)
    return model

def build_resnet(cfg: ModelConfig):
    spec = ModelSpec(image_shape=cfg.image_shape)

    spec.add_layer({
        "type": "resnet18",
        "num_classes": cfg.num_classes,
        "in_channels": cfg.image_shape[0],
        "base_channels": 64,
        "embed_size": cfg.embed_size,
        "dropout_p": 0.0,
    }, name="resnet18")

    model = Model(spec)
    return model

def main():
    cfg = ModelConfig()
    cfg = update_config(cfg, {
        "name": "resnet18",
        "model_type": "resnet_custom",
        "path": "/models",
        "image_shape": (3, 32, 32),
        "embed_size": 512,
        "num_classes": 100,
        "path": "./models/resnet18",
    })
    model = None

    if cfg.name == "cnn":
        model = build_cnn(cfg)
        config_dict = build_config_dict(model, model_type=cfg.name,)
    elif cfg.name == "resnet18":
        model = build_resnet(cfg)
        config_dict = build_config_dict(model, model_type=cfg.name,)

    save_config(cfg.path, config_dict)


if __name__ == "__main__":
    main()