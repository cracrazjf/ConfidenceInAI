import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar_stats(dataset_name="cifar10"):
    if dataset_name.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name.lower() == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mean, std


class AddGaussianNoise:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, x):
        return torch.clamp(x + torch.randn_like(x) * self.std, 0.0, 1.0)


class AddSparseGaussianNoise:
    def __init__(self, std=0.1, p=0.1):
        self.std = std
        self.p = p

    def __call__(self, x):
        mask = torch.rand_like(x) < self.p
        noise = torch.randn_like(x) * self.std
        x = x.clone()
        x[mask] += noise[mask]
        return torch.clamp(x, 0.0, 1.0)


class AddSaltPepperNoise:
    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, x):
        rand = torch.rand_like(x)
        x = x.clone()
        x[rand < self.p / 2] = 0.0
        x[rand > 1 - self.p / 2] = 1.0
        return x


def get_transforms(dataset_name="cifar10", augment=True, test_augment=None):
    """
    test_augment options:
        None
        {"type": "gaussian", "std": 0.1}
        {"type": "sparse_gaussian", "std": 0.1, "p": 0.1}
        {"type": "salt_pepper", "p": 0.05}
    """
    mean, std = get_cifar_stats(dataset_name)

    # ----- Train -----
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # ----- Eval (clean) -----
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ----- Test (possibly noisy) -----
    if test_augment is None:
        test_transform = eval_transform
    else:
        noise_type = test_augment.get("type")

        if noise_type == "gaussian":
            noise = AddGaussianNoise(std=test_augment.get("std", 0.1))

        elif noise_type == "sparse_gaussian":
            noise = AddSparseGaussianNoise(
                std=test_augment.get("std", 0.1),
                p=test_augment.get("p", 0.1),
            )

        elif noise_type == "salt_pepper":
            noise = AddSaltPepperNoise(p=test_augment.get("p", 0.05))

        else:
            raise ValueError(f"Unknown test_augment type: {noise_type}")

        # IMPORTANT: apply noise BEFORE normalization
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            noise,
            transforms.Normalize(mean, std),
        ])

    return train_transform, eval_transform, test_transform


class TransformSubset(torch.utils.data.Dataset):
    """
    Wrap a subset so that we can apply different transforms
    to train / val / calibration splits even if they come
    from the same original dataset.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return {"pixel_values": x, "labels": y}


def prepare_cifar_data(
    dataset_name="cifar10",
    root="./data",
    val_ratio=0.1,
    calib_ratio=0.1,
    seed=66,
    split_path="./data/split_info.pt",
):
    set_seed(seed)

    dataset_name = dataset_name.lower()
    if dataset_name not in ["cifar10", "cifar100"]:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    if dataset_name == "cifar10":
        base_train = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    else:
        base_train = datasets.CIFAR100(root=root, train=True, download=True, transform=None)

    n_total = len(base_train)
    n_val = int(n_total * val_ratio)
    n_calib = int(n_total * calib_ratio)
    n_train = n_total - n_val - n_calib

    if n_train <= 0:
        raise ValueError("Train split became empty. Reduce val_ratio/calib_ratio.")

    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(n_total, generator=generator).tolist()

    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:n_train + n_val]
    calib_indices = all_indices[n_train + n_val:]

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    torch.save(
        {
            "dataset_name": dataset_name,
            "root": root,
            "seed": seed,
            "val_ratio": val_ratio,
            "calib_ratio": calib_ratio,
            "n_total": n_total,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "calib_indices": calib_indices,
        },
        split_path,
    )
    print(f"\nSaved split info to: {split_path}")


def load_split_info(root, split_path, dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name not in ["cifar10", "cifar100"]:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")
    
    train_transform, eval_transform, _ = get_transforms(dataset_name, augment=False)

    if dataset_name == "cifar10":
        base_train = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
        num_classes = 10
    else:
        base_train = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
        num_classes = 100

    n_total = len(base_train)
    if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

    split_info = torch.load(split_path)

    if split_info["dataset_name"] != dataset_name:
        raise ValueError(
            f"Split file dataset_name={split_info['dataset_name']} "
            f"does not match requested dataset_name={dataset_name}"
        )

    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]
    calib_indices = split_info["calib_indices"]

    saved_n_total = split_info["n_total"]
    if saved_n_total != n_total:
        raise ValueError(
            f"Saved split expected n_total={saved_n_total}, but current dataset has n_total={n_total}"
        )

    print(f"\nLoaded existing split from: {split_path}")

    train_dataset = TransformSubset(base_train, train_indices, transform=train_transform)
    val_dataset = TransformSubset(base_train, val_indices, transform=eval_transform)

    calib_dataset = None
    if len(calib_indices) > 0:
        calib_dataset = TransformSubset(base_train, calib_indices, transform=eval_transform)

    print(f"\nPrepared {dataset_name.upper()}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Calib size: {0 if calib_dataset is None else len(calib_dataset)}")

    return train_dataset, val_dataset, calib_dataset, num_classes


def main():
    prepare_cifar_data(
        dataset_name="cifar10",
        root="./data",
        val_ratio=0.1,
        calib_ratio=0.1,
        seed=66,
        split_path="./data/cifar10_split_info.pt",
    )

    prepare_cifar_data(
        dataset_name="cifar100",
        root="./data",
        val_ratio=0.1,
        calib_ratio=0.1,
        seed=66,
        split_path="./data/cifar100_split_info.pt",
    )

if __name__ == "__main__":
    main()

