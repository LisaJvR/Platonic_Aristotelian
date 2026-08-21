# image-text, image-speech, text-speech experiment repeat from paltonian paper

# get text embeddings and image embeddings and calculate mKNN and plot
import torch
from pna_models import get_models, pretty_model_name, pretty_text_name, get_size, text_family
from pna_data import load_embeddings
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def compute_nearest_neighbors(feats, topk=1):
    """
    From: https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn

def mutual_knn(feats_A, feats_B, topk):
        """
        From: https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0
        
        acc = (lvm_mask * llm_mask).sum(dim=1) / topk
        
        return acc.mean().item()

# def plot_results(results):
#     for family in ["dinov2", "clip", "mae", "augreg", "data2vec-vision"]:
#         plt.figure(figsize=(10, 6))
#         for image_model in results:
#             if family in image_model:
#                 text_models = results[image_model]
#                 text_model_names = list(text_models.keys())
#                 accuracies = [text_models[text_model] for text_model in text_model_names]

#                 plt.plot(
#                     text_model_names,
#                     accuracies,
#                     marker="o",
#                     label=image_model,
#                 )
#                 plt.savefig(f"../plots/mutual_knn_{family}.png")

def plot_results(results):
    os.makedirs("../plots", exist_ok=True)

    # ------------------------------------------------------------
    # Styling taken from the reference figure
    # ------------------------------------------------------------
    text_color = "#304766"

    viridis = plt.colormaps.get_cmap("viridis")
    

    size_colors = {
        "small": viridis(0.90),
        "base": viridis(0.65),
        "large": viridis(0.35),
        "giant": viridis(0.02),
        "tiny": viridis(0.98),
        "huge": viridis(0.15),
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 14,
        "legend.fontsize": 11,

        "axes.labelcolor": text_color,
        "axes.edgecolor": "#c5cbd1",

        "xtick.color": text_color,
        "ytick.color": text_color,

        "axes.linewidth": 2.0,

        "grid.color": "#dddddd",
        "grid.linewidth": 1.5,
        "grid.alpha": 0.8,
    })



    # ------------------------------------------------------------
    # Plot each vision family separately
    # ------------------------------------------------------------
    families = [
        "dinov2",
        "clip",
        "mae",
        "augreg",
        "data2vec-vision",
    ]

    for family in families:

        family_models = [
            model for model in results
            if family in model.lower()
        ]

        if not family_models:
            continue

        fig, ax = plt.subplots(figsize=(4.7, 4.7), dpi=100)

        # same light background as reference
        ax.set_facecolor("#f3f3f3")
        fig.patch.set_facecolor("white")

        # --------------------------------------------------------
        # Establish common x ordering
        # --------------------------------------------------------
        first_model = family_models[0]

        text_models = list(results[first_model].keys())

        x = np.arange(len(text_models))

        # --------------------------------------------------------
        # Draw each image model
        # --------------------------------------------------------
        for image_model in family_models:

            values = [
                results[image_model][text_model]
                for text_model in text_models
            ]

            size = get_size(image_model)
            color = size_colors[size]

            ax.plot(
                x,
                values,
                color=color,
                linewidth=3.0,
                marker="o",
                markersize=9,
                markeredgewidth=0,
                label=size,
                zorder=3,
            )

        # --------------------------------------------------------
        # X tick labels: 560m, 1b1, ...
        # --------------------------------------------------------
        ax.set_xticks(x)

        ax.set_xticklabels(
            [pretty_text_name(m) for m in text_models],
            rotation=50,
            ha="right",
            rotation_mode="anchor",
            color=text_color,
        )

        # --------------------------------------------------------
        # Y axis
        # --------------------------------------------------------
        ax.set_ylabel(
            f"Alignment to {family.upper()}",
            fontsize=22,
            color=text_color,
        )

        # reference has ticks pointing outward
        ax.tick_params(
            axis="both",
            direction="out",
            length=8,
            width=2,
            colors=text_color,
        )

        # --------------------------------------------------------
        # Grid / spines
        # --------------------------------------------------------
        ax.grid(True, axis="both")
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#c3c9ce")
        ax.spines["bottom"].set_color("#c3c9ce")

        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

        # --------------------------------------------------------
        # Legend
        # --------------------------------------------------------
        handles, labels = ax.get_legend_handles_labels()

        # remove repeated labels if necessary
        unique = {}
        for handle, label in zip(handles, labels):
            unique[label] = handle

        desired_order = [
            size for size in ["small", "base", "large", "giant"]
            if size in unique
        ]

        legend = ax.legend(
            [unique[s] for s in desired_order],
            desired_order,
            loc="upper left",
            frameon=True,
            fancybox=False,
            shadow=True,
            facecolor="white",
            edgecolor="#b8b8b8",
            framealpha=1.0,
            handlelength=0.8,
            handletextpad=0.6,
            borderpad=0.4,
            labelspacing=0.3,
        )

        # --------------------------------------------------------
        # Add BLOOM / OpenLLaMA / LLaMA group names
        # --------------------------------------------------------
        families_x = {}

        for i, model in enumerate(text_models):
            f = text_family(model)

            if f:
                families_x.setdefault(f, []).append(i)

        for text_fam, positions in families_x.items():
            centre = np.mean(positions)

            ax.text(
                centre,
                -0.19,
                text_fam,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=15,
                color=text_color,
            )

        # extra room for the family labels
        plt.subplots_adjust(
            left=0.27,
            right=0.97,
            top=0.97,
            bottom=0.25,
        )

        os.makedirs("../plots/results/", exist_ok=True)
        plt.savefig(
            f"../plots/results/mutual_knn_{family}.png",
            dpi=100,
            bbox_inches="tight",
        )

        plt.close(fig)
            

if __name__ == "__main__":
    test = True

    image_models = get_models("test", "image")
    text_models = get_models("test", "text")

    num_chunks = 10 
    if test == True:
        num_chunks = 2

    results = {}
    # for each image model
    for image_model in image_models:
        # for each text model (in order of platonic paper)
        for image_chunk in range(num_chunks):
            loaded_image_data = load_embeddings(image_model, "image", chunk_num=image_chunk)
            if loaded_image_data is None:
                print(f"Warning: No embeddings found for {image_model} (chunk {image_chunk}). Skipping.")
                continue
            image_feats = loaded_image_data["avg"]
            image_feats_40k = image_feats.repeat_interleave(5, dim=0) # every image x5 captions

            for text_model in text_models:
                 for text_chunk in range(num_chunks):
                    loaded_text_data = load_embeddings(text_model, "text", chunk_num=text_chunk)
                    if loaded_text_data is None:
                        print(f"Warning: No embeddings found for {text_model} (chunk {text_chunk}). Skipping.")
                        continue
                    text_feats = loaded_text_data["avg"]

                    # compute mutual knn
                    acc = mutual_knn(image_feats_40k, text_feats, topk=10)
                    print(f"Mutual KNN accuracy between {image_model} (chunk {image_chunk}) and {text_model} (chunk {text_chunk}): {acc:.4f}")
                    results[(image_model, text_model)] = acc
            # for all data chunks
    
    print("Final results:", results)
    plot_results(results)

