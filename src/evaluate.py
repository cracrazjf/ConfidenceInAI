from psychai.config import EvaluationConfig, update_config
from psychai.vision.vm import TrainingManager
from pathlib import Path
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import json
import os
import math
from prepare_data import get_transforms, TransformSubset, load_split_info
import torch.nn.functional as F
import pandas as pd
import time
import numpy as np

def aug_to_str(aug):
    if isinstance(aug, str):
        return aug
    aug_type = aug["type"]
    
    params = [
        f"{k}{v}" for k, v in aug.items()
        if k != "type"
    ]
    
    if not params:
        return aug_type
    
    return f"{aug_type}_" + "_".join(params)

def evaluate_behavior(cfg, run_dirs, test_augment="clean", num_eval_runs=1, accuracy_log_file=None):
    tm = TrainingManager(cfg)
    if cfg.data.name == "cifar10":
        base_test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=None)
    else:
        base_test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    test_aug_info = f"test_aug_{aug_to_str(test_augment)}"
    save_path = Path(cfg.exp_dir) / f"{test_aug_info}_behavior_evaluation.jsonl"
    os.makedirs(cfg.exp_dir, exist_ok=True)
    with open(save_path, "w") as f:
        pass
    
    for eval_run in range(num_eval_runs):
        _, _, test_transform = get_transforms(cfg.data.name, augment=False, test_augment=test_augment)
        test_dataset = TransformSubset(base_test_dataset, indices=list(range(len(base_test_dataset))), transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

        accuracies = []
        for i, run_dir in enumerate(run_dirs[:1]):
            def eval_fn(mm, cfg, idxs, inputs, labels, logits, preds, embeddings, weights):
                with open(save_path, "a") as f:
                    for i in range(len(inputs)):
                        probs = F.softmax(logits[i], dim=0)
                        num_classes = probs.shape[0]
                        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
                        max_entropy = math.log(num_classes)
                        normalized_entropy = 1.0 - (entropy / max_entropy)
                        rec = {
                            "eval_run": eval_run,
                            "model_path": mm.model_path.parent.name,
                            "sample_id": idxs[i].item(),
                            "labels": labels[i].item(),
                            "prob": probs[preds[i]].item(),
                            "correct": float(preds[i].item() == labels[i].item()),
                            "entropy": entropy,
                            "normalized_entropy": normalized_entropy,
                            "preds": preds[i].item(),
                        }
                        f.write(json.dumps(rec) + "\n")
                accuracies.append((preds == labels).float().mean().item())


            tm.mm.load_model(cfg.model.name, run_dir / "export", cfg.model.model_type, cfg.model.wrapper, cfg.device)
            tm.evaluate(test_loader, epoch=0, step=0, eval_fn=eval_fn)

            with open(accuracy_log_file, "a") as acc_f:
                acc_rec = {
                    "timestamp": int(time.time()),
                    "eval_run": eval_run,
                    "model_path": run_dir.name,
                    "accuracy": np.mean(accuracies) if accuracies else 0.0,
                }
                acc_f.write(json.dumps(acc_rec) + "\n")

def evaluate_embeddings(cfg, run_dirs, train_augment=False, test_augment="clean", k=50):
    tm = TrainingManager(cfg)

    train_dataset, _, _, num_classes = load_split_info(
        root="./data",
        split_path=f"./data/{cfg.data.name}_split_info.pt",
        dataset_name=cfg.data.name,
        augment=train_augment,
    )

    if cfg.data.name == "cifar10":
        base_test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=None
        )
    else:
        base_test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=None
        )

    test_aug_info = f"test_aug_{aug_to_str(test_augment)}"
    os.makedirs(cfg.exp_dir, exist_ok=True)

    _, _, test_transform = get_transforms(
        cfg.data.name, augment=False, test_augment=test_augment
    )
    test_dataset = TransformSubset(
        base_test_dataset,
        indices=list(range(len(base_test_dataset))),
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    def collect_outputs_for_loader(tm, cfg, run_dir, loader):
        embed_list = []
        label_list = []
        pred_list = []
        correct_list = []

        def eval_fn(mm, cfg, idxs, inputs, labels, logits, preds, embeddings, weights):
            embed_list.append(embeddings.detach().cpu())
            label_list.append(labels.detach().cpu())
            pred_list.append(preds.detach().cpu())
            correct_list.append((preds == labels).float().detach().cpu())

        tm.mm.load_model(
            cfg.model.name,
            run_dir / "export",
            cfg.model.model_type,
            cfg.model.wrapper,
            cfg.device,
        )
        tm.evaluate(loader, epoch=0, step=0, eval_fn=eval_fn)

        return {
            "embeddings": torch.cat(embed_list, dim=0),
            "labels": torch.cat(label_list, dim=0),
            "preds": torch.cat(pred_list, dim=0),
            "correct": torch.cat(correct_list, dim=0),
        }

    all_records = {}

    for run_dir in run_dirs[:1]:
        train_outputs = collect_outputs_for_loader(tm, cfg, run_dir, train_loader)
        test_outputs = collect_outputs_for_loader(tm, cfg, run_dir, test_loader)

        # Normalize embeddings for cosine-based kNN / prototype similarity
        train_embeddings = F.normalize(train_outputs["embeddings"].float(), p=2, dim=1)
        test_embeddings = F.normalize(test_outputs["embeddings"].float(), p=2, dim=1)

        train_labels = train_outputs["labels"].long()
        test_labels = test_outputs["labels"].long()
        test_preds = test_outputs["preds"].long()

        # ------------------------------------------------------------
        # 1) Class prototypes (mean normalized embedding per class)
        # ------------------------------------------------------------
        class_means = []
        for c in range(num_classes):
            feats_c = train_embeddings[train_labels == c]
            if feats_c.size(0) == 0:
                raise ValueError(f"No training examples found for class {c}")
            mean_c = feats_c.mean(dim=0)
            mean_c = F.normalize(mean_c.unsqueeze(0), p=2, dim=1).squeeze(0)
            class_means.append(mean_c)
        class_means = torch.stack(class_means, dim=0)  # [C, D]

        knn_conf_list = []

        # ------------------------------------------------------------
        # 2) Compute confidence scores in batches
        # ------------------------------------------------------------
        for start in range(0, test_embeddings.size(0), 512):
            end = min(start + 512, test_embeddings.size(0))
            x = test_embeddings[start:end]                       # [B, D]
            preds = test_preds[start:end]                        # [B]
            labels = test_labels[start:end]                      # [B]

            # -------------------------
            # kNN confidence
            # -------------------------
            sim = x @ train_embeddings.T                         # [B, N_train]
            topk_sim, nn_idx = torch.topk(sim, k=k, dim=1, largest=True)
            nn_labels = train_labels[nn_idx]                     # [B, k]

            # similarity-weighted local class probability
            weights = torch.softmax(topk_sim, dim=1)            # [B, k]
            class_probs = torch.zeros(
                x.size(0), num_classes, device=x.device, dtype=x.dtype
            )
            class_probs.scatter_add_(1, nn_labels, weights)

            knn_conf = class_probs[
                torch.arange(x.size(0), device=x.device), preds
            ]                                                   # support for predicted class
            knn_conf_list.append(knn_conf.cpu())


        all_records[run_dir.name] = {
            "sample_id": torch.arange(len(test_dataset)),
            "knn_confidence": torch.cat(knn_conf_list, dim=0),
            "correct": test_outputs["correct"],
            "preds": test_outputs["preds"],
            "labels": test_outputs["labels"],
        }

    df_all = pd.concat(
        [
            pd.DataFrame(
                {
                    "model_path": [run_name] * len(record["sample_id"]),
                    "sample_id": record["sample_id"].cpu().numpy(),
                    "knn_confidence": record["knn_confidence"].cpu().numpy(),
                    "correct": record["correct"].cpu().int().numpy(),
                    "preds": record["preds"].cpu().numpy(),
                    "labels": record["labels"].cpu().numpy(),
                }
            )
            for run_name, record in all_records.items()
        ],
        ignore_index=True,
    )

    save_path = Path(cfg.exp_dir) / f"{test_aug_info}_embedding_knn.csv"
    df_all.to_csv(save_path, index=False)
    print(f"Saved to: {save_path}")
    

def compute_coherence(base_eval_path, compare_eval_path, save_path):
    base_df = pd.read_json(base_eval_path, lines=True)
    compare_df = pd.read_json(compare_eval_path, lines=True)

    compare_df = compare_df.merge(base_df[["model_path", "sample_id","correct", "preds"]], on=["model_path", "sample_id"], suffixes=("", "_base"))
    compare_df["match"] = (compare_df["preds"] == compare_df["preds_base"]).astype(float)
    model_coherence = (
        compare_df
        .groupby(["model_path", "sample_id"], as_index=False)
        .agg(
            prob=("prob", "mean"),
            normalized_entropy=("normalized_entropy", "mean"),
            coherence=("match", "mean"),
            correct=("correct_base", "first"),
        )
    )
    model_coherence.to_csv(save_path, index=False)


def main():
    cfg = EvaluationConfig()
    updates = {
        "model": {
            "name": "resnet18",
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
        "logging": {
            "return_embeddings": True,
            "layer_of_interest": "resnet18",
            "embed_type": "features",
        },
        "root_dir": "./trained",
        "exp_name": "cia_cifar100_resnet18",
        "exp_dir": "./evaluated_resnet18",
        "device": "cpu",
    }
    cfg = update_config(cfg, updates)
    exp_dir = Path(cfg.root_dir) / cfg.exp_name
    run_dirs = [r for r in exp_dir.iterdir() if r.is_dir()]
    run_dirs.sort(key=lambda r: r.name)
    os.makedirs(cfg.exp_dir, exist_ok=True)

    accuracy_log_file = Path(f"{cfg.exp_dir}/accuracy_log.json")

    # evaluate_behavior(cfg, run_dirs, test_augment={"type": "sparse_gaussian", "std": 0.1, "p": 0.1}, num_eval_runs=30, accuracy_log_file=accuracy_log_file)
    # evaluate_behavior(cfg, run_dirs, test_augment={"type": "salt_pepper", "p": 0.5}, num_eval_runs=30, accuracy_log_file=accuracy_log_file)
    # evaluate_behavior(cfg, run_dirs, test_augment="MC_clean_p0.1", num_eval_runs=50, accuracy_log_file=accuracy_log_file)
    # evaluate_behavior(cfg, run_dirs, test_augment="clean", num_eval_runs=1, accuracy_log_file=accuracy_log_file)

    # clean_df = pd.read_json(f"{cfg.exp_dir}/test_aug_clean_behavior_evaluation.jsonl", lines=True)
    # clean_df = clean_df[["model_path", "sample_id", "prob", "normalized_entropy", "correct"]]
    # clean_df.to_csv(f"{cfg.exp_dir}/clean.csv", index=False)

    # compute_coherence(
    #     base_eval_path=f"{cfg.exp_dir}/test_aug_clean_behavior_evaluation.jsonl",
    #     compare_eval_path=f"{cfg.exp_dir}/test_aug_MC_clean_p0.1_behavior_evaluation.jsonl",
    #     save_path=f"{cfg.exp_dir}/MC_clean_p0.1.csv"
    # )

    # compute_coherence(
    #     base_eval_path=f"{cfg.exp_dir}/test_aug_clean_behavior_evaluation.jsonl",
    #     compare_eval_path=f"{cfg.exp_dir}/test_aug_sparse_gaussian_std0.1_p0.1_behavior_evaluation.jsonl",
    #     save_path=f"{cfg.exp_dir}/sparse_gaussian_std0.1_p0.1.csv"
    # )

    # compute_coherence(
    #     base_eval_path=f"{cfg.exp_dir}/test_aug_clean_behavior_evaluation.jsonl",
    #     compare_eval_path=f"{cfg.exp_dir}/test_aug_salt_pepper_p0.5_behavior_evaluation.jsonl",
    #     save_path=f"{cfg.exp_dir}/salt_pepper_p0.5.csv"
    # )

    # evaluate_embeddings(cfg, run_dirs, train_augment=False, test_augment="clean", k=30)

if __name__ == "__main__":
    main()