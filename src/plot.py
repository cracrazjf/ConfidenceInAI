import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math

from sklearn.metrics import roc_curve, roc_auc_score

def compute_ece(confidence, correct, n_bins=10):
    """
    confidence: array-like, shape [N], values in [0,1]
    correct: array-like, shape [N], values 0 or 1
    """
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


def plot_reliability_diagram(df, confidence_key="prob", correct_key="correct", n_bins=10, title="Reliability Diagram", save_path=None):
    eces = []
    run_names = df["model_path"].unique()
    n_cols = 5
    n_rows = math.ceil(len(run_names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    for i, run_name in enumerate(run_names):
        ax = axes[i]
        df_run = df[df["model_path"] == run_name]
        confidence = df_run[confidence_key].values
        correct = df_run[correct_key].values
        ece = compute_ece(confidence, correct, n_bins)
        eces.append(ece)
        confidence = np.asarray(confidence)
        correct = np.asarray(correct)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_confs = []
        bin_counts = []

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
            bin_counts.append(mask.sum())

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
            fontsize=12,
            verticalalignment="top"
        )

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{run_name}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.9)
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return np.mean(eces)


def plot_roc_and_compute_auc(df, confidence_key="prob", correct_key="correct", title="ROC Curve", save_path=None):
    aucs = []
    run_names = df["model_path"].unique()
    n_cols = 5
    n_rows = math.ceil(len(run_names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    for i, run_name in enumerate(run_names):
        ax = axes[i]
        df_run = df[df["model_path"] == run_name]
        confidence = df_run[confidence_key].values
        correct = df_run[correct_key].values

        fpr, tpr, _ = roc_curve(correct, confidence)
        auc = roc_auc_score(correct, confidence)

        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(run_name)
        ax.legend()
        aucs.append(auc)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return np.mean(aucs)

def main():
    csv_files = list(Path("./evaluated").glob("*.csv"))
    os.makedirs("./figures", exist_ok=True)
    ece_map = {}
    auc_map = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "embedding" in csv_file.name:
            knn_ece = plot_reliability_diagram(df=df, confidence_key="knn_confidence", correct_key="correct", n_bins=10, title=f"{csv_file.name} - KNN Reliability", save_path=Path("./figures") / f"{csv_file.stem}_knn_reliability.png")
            ece_map[f"{csv_file.stem}_knn"] = np.round(knn_ece, 2)
            mahalanobis_ece = plot_reliability_diagram(df=df, confidence_key="maha_confidence", correct_key="correct", n_bins=10, title=f"{csv_file.name} - Mahalanobis Reliability", save_path=Path("./figures") / f"{csv_file.stem}_maha_reliability.png")
            ece_map[f"{csv_file.stem}_maha"] = np.round(mahalanobis_ece, 2)

            knn_auc = plot_roc_and_compute_auc(df=df, confidence_key="knn_confidence", correct_key="correct", title=f"{csv_file.name} - KNN ROC", save_path=Path("./figures") / f"{csv_file.stem}_knn_roc.png")
            auc_map[f"{csv_file.stem}_knn"] = np.round(knn_auc, 2)
            mahalanobis_auc = plot_roc_and_compute_auc(df=df, confidence_key="maha_confidence", correct_key="correct", title=f"{csv_file.name} - Mahalanobis ROC", save_path=Path("./figures") / f"{csv_file.stem}_maha_roc.png")
            auc_map[f"{csv_file.stem}_maha"] = np.round(mahalanobis_auc, 2)
        else:
            prob_ece = plot_reliability_diagram(df=df, confidence_key="prob", correct_key="correct", n_bins=10, title=f"{csv_file.name} - Probability Reliability", save_path=Path("./figures") / f"{csv_file.stem}_prob_reliability.png")
            ece_map[f"{csv_file.stem}_prob"] = np.round(prob_ece, 2)
            entropy_ece = plot_reliability_diagram(df=df, confidence_key="normalized_entropy", correct_key="correct", n_bins=10, title=f"{csv_file.name} - Entropy Reliability", save_path=Path("./figures") / f"{csv_file.stem}_entropy_reliability.png")
            ece_map[f"{csv_file.stem}_entropy"] = np.round(entropy_ece, 2)

            prob_auc = plot_roc_and_compute_auc(df=df, confidence_key="prob", correct_key="correct", title=f"{csv_file.name} - Probability ROC", save_path=Path("./figures") / f"{csv_file.stem}_prob_roc.png")
            auc_map[f"{csv_file.stem}_prob"] = np.round(prob_auc, 2)
            entropy_auc = plot_roc_and_compute_auc(df=df, confidence_key="normalized_entropy", correct_key="correct", title=f"{csv_file.name} - Entropy ROC", save_path=Path("./figures") / f"{csv_file.stem}_entropy_roc.png")
            auc_map[f"{csv_file.stem}_entropy"] = np.round(entropy_auc, 2)

    ece_df = pd.DataFrame(list(ece_map.items()), columns=["metric", "ece"])
    ece_df.to_csv("./figures/ece_summary.csv", index=False)

    auc_df = pd.DataFrame(list(auc_map.items()), columns=["metric", "auc"])
    auc_df.to_csv("./figures/auc_summary.csv", index=False)
if __name__ == "__main__":
    main()