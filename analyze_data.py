import os
import pandas as pd
import matplotlib


os.environ.setdefault("MPLCONFIGDIR", "/Users/ayushb3112/Documents/tdt4241/.mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_FILE = "/Users/ayushb3112/Documents/tdt4241/aita_verdicts_unique_6000.csv"
OUTPUT_DIR = "/Users/ayushb3112/Documents/tdt4241/analysis_outputs"


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    # Basic cleanup aligned with training pipeline.
    df["title"] = df["title"].fillna("")
    df["full post"] = df["full post"].fillna("")
    df["text"] = (df["title"] + " " + df["full post"]).str.strip()
    df["text_len_chars"] = df["text"].str.len()
    df["text_len_words"] = df["text"].str.split().str.len()

    # Normalize verdict names for clearer plots.
    verdict_map = {
        "user_is_fault": "User is at fault",
        "user_ok": "User not at fault",
    }
    df["verdict_pretty"] = df["verdict"].map(verdict_map).fillna(df["verdict"])

    # 1) Class distribution plot.
    counts = df["verdict_pretty"].value_counts()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=["#d95f02", "#1b9e77"])
    plt.title("AITA Dataset: Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of posts")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    class_plot = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(class_plot, dpi=200)
    plt.close()

    # 2) Text length distribution (words).
    plt.figure(figsize=(8, 5))
    plt.hist(df["text_len_words"], bins=40, color="#7570b3", alpha=0.9, edgecolor="white")
    plt.title("AITA Dataset: Post Length Distribution")
    plt.xlabel("Words per post (title + body)")
    plt.ylabel("Number of posts")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    length_plot = os.path.join(OUTPUT_DIR, "post_length_distribution_words.png")
    plt.savefig(length_plot, dpi=200)
    plt.close()

    # 3) Boxplot of post length by class.
    grouped = [
        df.loc[df["verdict_pretty"] == label, "text_len_words"].values
        for label in counts.index
    ]
    plt.figure(figsize=(8, 5))
    box = plt.boxplot(grouped, labels=counts.index, patch_artist=True, showfliers=False)
    colors = ["#e7298a", "#66a61e"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    plt.title("Post Length by Class")
    plt.ylabel("Words per post")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    box_plot = os.path.join(OUTPUT_DIR, "post_length_by_class_boxplot.png")
    plt.savefig(box_plot, dpi=200)
    plt.close()

    # Save quick numeric summary for the report/slides.
    summary = {
        "num_rows": len(df),
        "class_counts": counts.to_dict(),
        "mean_words": float(df["text_len_words"].mean()),
        "median_words": float(df["text_len_words"].median()),
        "p95_words": float(df["text_len_words"].quantile(0.95)),
    }
    summary_path = os.path.join(OUTPUT_DIR, "data_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("Analysis complete. Generated files:")
    print(f"- {class_plot}")
    print(f"- {length_plot}")
    print(f"- {box_plot}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
