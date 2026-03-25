import os
from prepare_data import load_split_info
from torch.utils.data import DataLoader
from psychai.config import TrainingConfig, update_config, save_yaml_config
from psychai.vision.vm import ModelManager, TrainingManager


def main():
    cfg = TrainingConfig()
    exp_name = "cia_cifar100_resnet18"
    model_name = "resnet18"
    updates = {
        "model": {
            "name": f"{model_name}",
            "path": f"./models/{model_name}",
            "model_type": "custom",   
            "wrapper": "classification",
        },
        "data": {
            "name": "cifar100",
            "data_process_batch_size": 20,
            "data_process_num_proc": 4,
            "batch_size": 128,
            "num_workers": 0
        },
        "optim": {
            "lr": 0.1,
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "lr_scheduler": "multistep",
            "lr_steps": [100, 150],
            "gamma": 0.1,
        },
        "logging": {
            "log_strategy": "epoch",
            "log_interval": 20,
            "save_interval": 10,
            "save_total_limit": 5,
            "eval_strategy": "epoch",
            "eval_interval": 20,
            "prefer_safetensors": True,
            "return_embeddings": False,
            "return_weights": False
        },
        "exp_name": exp_name,
        "exp_dir": f"./trained/{exp_name}",
        "num_runs": 1,
        "num_epochs": 200,
        "seed": 66,
        "device": "cpu",
    }
    cfg = update_config(cfg, updates)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    save_yaml_config(cfg, f"{cfg.exp_dir}/config.yaml")

    train_dataset, val_dataset, calib_dataset, num_classes = load_split_info(
        root="./data",
        split_path=f"./data/{cfg.data.name}_split_info.pt",
        dataset_name=cfg.data.name,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle_dataloader, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers) 

    for batch in train_loader:
        print(batch["pixel_values"].shape, batch["labels"].shape)
        break

    tm = TrainingManager(cfg)
    tm.train(train_loader, val_loader, eval_fn=None)


if __name__ == "__main__":
    main()