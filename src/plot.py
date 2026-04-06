import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
from collections import defaultdict
import re

from sklearn.metrics import roc_curve, roc_auc_score


def extract_p(stem: str):
    """
    Extract p value from filename.
    Example: sparse_gaussian_std0.1_p0.25 -> 0.25
    """
    match = re.search(r"_p([0-9.]+)", stem)
    if match:
        return float(match.group(1))
    return -1  # for cases like 'clean' or no p

def compute_ece(confidence, correct, n_bins=10):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        if mask.sum() == 0:
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        bin_weight = mask.mean()
        ece += np.abs(bin_acc - bin_conf) * bin_weight

    return ece


def get_family_name(stem: str) -> str:
    """
    Extract dataset family from file stem.

    Examples:
    - sparse_gaussian_std0.1_p0.1 -> sparse_gaussian
    - salt_pepper_p0.25 -> salt_pepper
    - MC_clean_p0.1 -> MC_clean
    - test_aug_clean_embedding -> test_aug_clean
    - clean -> clean
    """
    if stem.startswith("sparse_gaussian"):
        return "sparse_gaussian"
    if stem.startswith("salt_pepper"):
        return "salt_pepper"
    if stem.startswith("MC_clean"):
        return "MC_clean"
    if stem.startswith("test_aug_clean"):
        return "test_aug_clean"
    if stem.startswith("clean"):
        return "clean"
    return stem


def get_measure_map(df, csv_name):
    """
    Return dict of measure_name -> confidence column
    """
    measures = {}

    if "embedding" in csv_name:
        if "knn_confidence" in df.columns:
            measures["knn"] = "knn_confidence"
    else:
        if "prob" in df.columns:
            measures["prob"] = "prob"
        if "normalized_entropy" in df.columns:
            measures["entropy"] = "normalized_entropy"
        if "coherence" in df.columns:
            measures["coherence"] = "coherence"

    return measures


def plot_reliability_on_axis(ax, df, confidence_key="prob", correct_key="correct", n_bins=10, title=""):
    confidence = df[confidence_key].values
    correct = df[correct_key].values
    ece = compute_ece(confidence, correct, n_bins)

    confidence = np.asarray(confidence)
    correct = np.asarray(correct)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        if mask.sum() == 0:
            continue

        bin_centers.append((left + right) / 2)
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confidence[mask].mean())

    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.bar(
        bin_centers,
        bin_accs,
        width=1 / n_bins,
        alpha=0.6,
        edgecolor="black",
        label="Accuracy",
        align="center",
    )
    ax.plot(bin_centers, bin_confs, marker="o", label="Mean confidence")

    ax.text(
        0.05, 0.95, f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top"
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ece


def plot_roc_on_axis(ax, df, confidence_key="prob", correct_key="correct", title=""):
    confidence = df[confidence_key].values
    correct = df[correct_key].values

    fpr, tpr, _ = roc_curve(correct, confidence)
    auc = roc_auc_score(correct, confidence)

    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=9)

    return auc


def save_family_subplot(family_name, items, plot_type="roc", save_root=Path("./figures_cnn"), n_bins=20):
    """
    items: list of dicts with keys:
        - stem
        - df
        - measure_name
        - confidence_key
    """
    if len(items) == 0:
        return {}

    n = len(items)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    score_map = {}

    items = sorted(
        items,
        key=lambda x: (extract_p(x["stem"]), x["measure_name"])
    )

    for i, item in enumerate(items):
        ax = axes[i]
        title = f"{item['stem']} ({item['measure_name']})"

        if plot_type == "roc":
            score = plot_roc_on_axis(
                ax=ax,
                df=item["df"],
                confidence_key=item["confidence_key"],
                correct_key="correct",
                title=title,
            )
            score_map[f"{item['stem']}_{item['measure_name']}"] = np.round(score, 2)
        else:
            score = plot_reliability_on_axis(
                ax=ax,
                df=item["df"],
                confidence_key=item["confidence_key"],
                correct_key="correct",
                n_bins=n_bins,
                title=title,
            )
            score_map[f"{item['stem']}_{item['measure_name']}"] = np.round(score, 2)

    # hide unused axes
    for j in range(len(items), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{family_name} - {plot_type.upper()}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = save_root / f"{family_name}_{plot_type}_subplot.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return score_map


def main():
    csv_files = list(Path("./evaluated_resnet18").glob("*.csv"))
    save_root = Path("./figures_resnet18")
    os.makedirs(save_root, exist_ok=True)

    grouped = defaultdict(list)
    ece_map = {}
    auc_map = {}

    # Step 1: collect all plot jobs by family
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        stem = csv_file.stem
        family = get_family_name(stem)
        measure_map = get_measure_map(df, csv_file.name)

        for measure_name, confidence_key in measure_map.items():
            grouped[family].append(
                {
                    "stem": stem,
                    "df": df,
                    "measure_name": measure_name,
                    "confidence_key": confidence_key,
                }
            )

    # Step 2: save one ROC subplot figure and one ECE subplot figure per family
    for family, items in grouped.items():
        auc_scores = save_family_subplot(
            family_name=family,
            items=items,
            plot_type="roc",
            save_root=save_root,
        )
        auc_map.update(auc_scores)

        ece_scores = save_family_subplot(
            family_name=family,
            items=items,
            plot_type="ece",
            save_root=save_root,
            n_bins=20,
        )
        ece_map.update(ece_scores)

    # Step 3: save summaries
    ece_df = pd.DataFrame(list(ece_map.items()), columns=["metric", "ece"])
    ece_df.to_csv(save_root / "ece_summary.csv", index=False)

    auc_df = pd.DataFrame(list(auc_map.items()), columns=["metric", "auc"])
    auc_df.to_csv(save_root / "auc_summary.csv", index=False)


if __name__ == "__main__":
    main()