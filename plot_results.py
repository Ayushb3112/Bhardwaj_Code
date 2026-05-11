import os
import joblib
import numpy as np
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/Users/ayushb3112/Documents/tdt4241/.mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


RESULTS_FILE = "/Users/ayushb3112/Documents/tdt4241/models/baseline_results.joblib"
OUTPUT_DIR = "/Users/ayushb3112/Documents/tdt4241/analysis_outputs"
DISTIL_SEEDS = {
    42: {"acc": 0.7339, "f1": 0.7338},
    43: {"acc": 0.7416, "f1": 0.7411},
    44: {"acc": 0.7335, "f1": 0.7331},
}


def plot_metric_comparison(results: dict) -> str:
    models = ["Majority", "TF-IDF + LR"]
    acc = [results["majority"]["acc"], results["tfidf_lr"]["acc"]]
    f1 = [results["majority"]["f1"], results["tfidf_lr"]["f1"]]

    x = np.arange(len(models))
    width = 0.36

    plt.figure(figsize=(8, 5))
    b1 = plt.bar(x - width / 2, acc, width, label="Accuracy", color="#1f77b4")
    b2 = plt.bar(x + width / 2, f1, width, label="Macro-F1", color="#ff7f0e")

    plt.ylim(0, 1.0)
    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "results_model_comparison.png")
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_main_comparison_with_distil(results: dict) -> str:
    majority_acc = results["majority"]["acc"]
    majority_f1 = results["majority"]["f1"]
    tfidf_acc = results["tfidf_lr"]["acc"]
    tfidf_f1 = results["tfidf_lr"]["f1"]
    distil_acc_vals = np.array([v["acc"] for v in DISTIL_SEEDS.values()])
    distil_f1_vals = np.array([v["f1"] for v in DISTIL_SEEDS.values()])
    distil_acc_mean = float(distil_acc_vals.mean())
    distil_acc_std = float(distil_acc_vals.std(ddof=1))
    distil_f1_mean = float(distil_f1_vals.mean())
    distil_f1_std = float(distil_f1_vals.std(ddof=1))

    models = ["Majority", "TF-IDF + LR", "DistilRoBERTa\n(3-seed mean)"]
    acc = [majority_acc, tfidf_acc, distil_acc_mean]
    acc_err = [0.0, 0.0, distil_acc_std]
    f1 = [majority_f1, tfidf_f1, distil_f1_mean]
    f1_err = [0.0, 0.0, distil_f1_std]

    x = np.arange(len(models))
    width = 0.36

    plt.figure(figsize=(10, 5.5))
    b1 = plt.bar(
        x - width / 2,
        acc,
        width,
        label="Accuracy",
        color="#1f77b4",
        yerr=acc_err,
        capsize=5,
    )

    b2 = plt.bar(
        x + width / 2,
        f1,
        width,
        label="Macro-F1",
        color="#ff7f0e",
        yerr=f1_err,
        capsize=5,
    )

    plt.ylim(0, 1.0)
    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Main Test-Set Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for i, bar in enumerate(b1):
        h = bar.get_height()
        label = f"{h:.4f}" if acc_err[i] == 0 else f"{h:.4f}\n±{acc_err[i]:.4f}"
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, label, ha="center", va="bottom", fontsize=9)

    for i, bar in enumerate(b2):
        h = bar.get_height()
        label = f"{h:.4f}" if f1_err[i] == 0 else f"{h:.4f}\n±{f1_err[i]:.4f}"
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "results_main_comparison_with_distil.png")
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_distil_seedwise() -> str:
    seeds = sorted(DISTIL_SEEDS.keys())
    acc = [DISTIL_SEEDS[s]["acc"] for s in seeds]
    f1 = [DISTIL_SEEDS[s]["f1"] for s in seeds]
    x = np.arange(len(seeds))
    width = 0.36

    plt.figure(figsize=(8.5, 5))
    b1 = plt.bar(x - width / 2, acc, width, label="Accuracy", color="#4daf4a")
    b2 = plt.bar(x + width / 2, f1, width, label="Macro-F1", color="#984ea3")
    plt.ylim(0.70, 0.75)
    plt.xticks(x, [str(s) for s in seeds])
    plt.xlabel("Random seed")
    plt.ylabel("Score")
    plt.title("DistilRoBERTa Test Results Across Seeds")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.0004, f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "results_distil_seedwise.png")
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_tfidf_confusion(results: dict) -> str:
    y_true = np.array(results["tfidf_lr"]["y_true"])
    y_pred = np.array(results["tfidf_lr"]["y_pred"])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("TF-IDF + LR Confusion Matrix")
    plt.colorbar()
    ticks = [0, 1]
    labels = ["User OK (0)", "User Fault (1)"]
    plt.xticks(ticks, labels, rotation=20)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "results_tfidf_confusion_matrix.png")
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = joblib.load(RESULTS_FILE)

    out1 = plot_metric_comparison(results)
    out2 = plot_tfidf_confusion(results)
    out3 = plot_main_comparison_with_distil(results)
    out4 = plot_distil_seedwise()

    print("Generated result plots:")
    print(f"- {out1}")
    print(f"- {out2}")
    print(f"- {out3}")
    print(f"- {out4}")


if __name__ == "__main__":
    main()
