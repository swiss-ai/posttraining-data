import os
from datasets import load_from_disk, Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser


def generate_model_frequency_histograms(
    data_dir, output_dir, single_file=False, title_suffix=""
):
    if single_file:
        files_to_process = [data_dir]
    else:
        files_to_process = sorted(
            f for f in os.listdir(data_dir)
        )
    # print(files_to_process)
    for fname in files_to_process:
        print("Processing file:", fname)
        data_path = os.path.join(data_dir, fname) if not single_file else fname

        # Support both HF dataset dirs and individual arrow files
        if os.path.isfile(data_path) and data_path.endswith(".arrow"):
            dataset = Dataset.from_file(data_path)
        else:
            dataset = load_from_disk(data_path)
            if hasattr(dataset, "keys"):
                split = "train_split" if "train_split" in dataset else list(dataset.keys())[0]
                dataset = dataset[split]
            elif isinstance(dataset, dict):
                dataset = dataset["train"] if "train" in dataset else dataset["train_split"] if "train_split" in dataset else list(dataset.values())[0]
        chosen_models = dataset["chosen_model"]
        rejected_models = dataset["rejected_model"]
        # all_chosen_scores = dataset["chosen_score"]
        # all_rejected_scores = dataset["rejected_score"]

        chosen_model_counter = defaultdict(int)
        rejected_model_counter = defaultdict(int)
        # chosen_scores = []
        # rejected_scores = []
        identical_counter = 0
        for cm, rm in zip(chosen_models, rejected_models):
            if cm == rm:
                identical_counter += 1
                continue
            chosen_model_counter[cm] += 1
            rejected_model_counter[rm] += 1
            # chosen_scores.append(cs)
            # rejected_scores.append(rs)

        # print("Average chosen score:", sum(chosen_scores) / len(chosen_scores))
        # print("Average rejected score:", sum(rejected_scores) / len(rejected_scores))
        # print(
        #     "Average score difference:",
        #     (sum(chosen_scores) / len(chosen_scores))
        #     - (sum(rejected_scores) / len(rejected_scores)),
        # )
        print("Number of identical chosen and rejected models:", identical_counter)

        chosen_model_counter_sorted = dict(
            sorted(chosen_model_counter.items(), key=lambda item: item[1], reverse=True)
        )
        rejected_model_counter_sorted = dict(
            sorted(
                rejected_model_counter.items(), key=lambda item: item[1], reverse=True
            )
        )
        # calculating percentages
        chosen_model_counter_sorted = {
            k: v / sum(chosen_model_counter_sorted.values()) * 100
            for k, v in chosen_model_counter_sorted.items()
        }
        rejected_model_counter_sorted = {
            k: v / sum(rejected_model_counter_sorted.values()) * 100
            for k, v in rejected_model_counter_sorted.items()
        }

        def plot_model_distributions_separate_y_axes(
            chosen_counter_sorted,
            rejected_counter_sorted,
            filename,
            plot_title="Model Selection Distribution"
            + (" " + title_suffix if title_suffix else ""),
        ):
            chosen_models = list(chosen_counter_sorted.keys())
            chosen_values = list(chosen_counter_sorted.values())
            rejected_models = list(rejected_counter_sorted.keys())
            rejected_values = list(rejected_counter_sorted.values())

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(28, max(6, len(chosen_models), len(rejected_models)) * 0.4),
            )
            fig.suptitle(plot_title, fontsize=20)  # <-- Add this line

            # Chosen models
            bars1 = axes[0].barh(chosen_models, chosen_values, color="green", alpha=0.8)
            axes[0].set_xlabel("Percentage (%)", fontsize=14)
            axes[0].set_title("Chosen Model Distribution", fontsize=16)
            axes[0].invert_yaxis()
            for bar, value in zip(bars1, chosen_values):
                axes[0].text(
                    value + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}%",
                    va="center",
                    fontsize=12,
                )
            axes[0].grid(axis="x", linestyle="--", alpha=0.5)

            # Rejected models
            bars2 = axes[1].barh(
                rejected_models, rejected_values, color="red", alpha=0.8
            )
            axes[1].set_xlabel("Percentage (%)", fontsize=14)
            axes[1].set_title("Rejected Model Distribution", fontsize=16)
            axes[1].invert_yaxis()
            for bar, value in zip(bars2, rejected_values):
                axes[1].text(
                    value + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}%",
                    va="center",
                    fontsize=12,
                )
            axes[1].grid(axis="x", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        os.makedirs(output_dir, exist_ok=True)
        plot_model_distributions_separate_y_axes(
            chosen_model_counter_sorted,
            rejected_model_counter_sorted,
            # f"distribution_{os.path.basename(data_path)}.png",
            os.path.join(
                output_dir,
                f"distribution_{os.path.basename(data_path)}.png",
            ),
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate model frequency histograms from dataset directory."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing dataset files.",
    )
    parser.add_argument(
        "--single_file",
        action="store_true",
        help="Indicates if the data_dir is a single file instead of a directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output histograms.",
    )
    parser.add_argument(
        "--title_suffix",
        type=str,
        default="",
        help="Suffix to add to the plot titles.",
    )
    args = parser.parse_args()
    generate_model_frequency_histograms(
        args.data_dir, args.output_dir, args.single_file, args.title_suffix
    )
