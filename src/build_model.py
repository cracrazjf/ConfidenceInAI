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

def main():
    cfg = ModelConfig()
    cfg = update_config(cfg, {
        "name": "cnn",
        "model_type": "cnn_custom",
        "path": "/models",
        "image_shape": (3, 64, 64),
        "embed_size": 128,
        "num_classes": 10,
        "path": "./models"
    })
    model = None

    if cfg.name == "cnn":
        model = build_cnn(cfg)
        config_dict = build_config_dict(model, model_type=cfg.name,)
        save_config(cfg.path, config_dict)

    save_config(cfg.path, config_dict)


if __name__ == "__main__":
    main()