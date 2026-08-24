# image-text, image-speech, text-speech experiment repeat from paltonian paper

# get text embeddings and image embeddings and calculate mKNN and plot
import torch
from pna_models import get_models, pretty_model_name, pretty_text_name, get_size, text_family
from pna_data import load_embeddings, print_meta_info, get_captions_from_index
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import faiss
import numpy as np
import torch

from pna_pipeline import EMB_DIR

# CKA ------------------------------------------------------------------------
def hsic(A, B, unbiased=False):
    '''
        From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
        Eqn 5 from: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    '''

    if unbiased:
        m = A.shape[0]

        # Zero out the diagonal elements of K and L
        A_tilde = A.clone().fill_diagonal_(0)
        B_tilde = B.clone().fill_diagonal_(0)

        # Compute HSIC using the formula in Equation 5
        HSIC_value = (
            (torch.sum(A_tilde * B_tilde.T))
            + (torch.sum(A_tilde) * torch.sum(B_tilde) / ((m - 1) * (m - 2)))
            - (2 * torch.sum(torch.mm(A_tilde, B_tilde)) / (m - 2))
        )

        HSIC_value /= m * (m - 3)
        return HSIC_value
    
    else:
        n = A.shape[0]
        H = torch.eye(n, dtype=A.dtype, device=A.device) - 1 / n
        return torch.trace(A @ H @ B @ H)
            
def compute_cka(feats_A, feats_B, type="linear", rbf_sigma=1.0, u=False):
    '''
    From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
    '''
    if type == "linear":
        kernel_A = torch.mm(feats_A, feats_A.T)
        kernel_B = torch.mm(feats_B, feats_B.T)

    elif type == "rbf":
        kernel_A = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
        kernel_B = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))

    H_AA = hsic(kernel_A, kernel_A, unbiased=u)
    H_BB = hsic(kernel_B, kernel_B, unbiased=u)
    H_AB = hsic(kernel_A, kernel_B, unbiased=u)

    cka_value = H_AB / (torch.sqrt(H_AA * H_BB) + 1e-6)  
    return cka_value.item()



# mutual KNN ------------------------------------------------------------------

def compute_knn_for_all_layers(embeddings, k):
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    # sklearn needs CPU NumPy arrays
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().float().cpu().numpy()

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    print("Computing KNN for embeddings with shape:", embeddings.shape)

    n_samples, n_layers, dim = embeddings.shape

    all_indices = torch.empty(
        (n_samples, n_layers, k),
        dtype=torch.long,
    )

    clean_indices = np.empty((n_samples, k), dtype=np.int64)

    for layer in range(n_layers):
        layer_embeddings = embeddings[:, layer, :].copy()

        faiss.normalize_L2(layer_embeddings)
        index = faiss.IndexFlatL2(dim)

        # XXX
        if use_gpu and faiss.get_num_gpus() > 0:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources,0,index,) #move to GPU

        index.add(layer_embeddings)

        _, indices = index.search(layer_embeddings, k + 1)

        for i in range(n_samples):
            neighbours = indices[i][indices[i] != i]
            clean_indices[i] = neighbours[:k]

        all_indices[:, layer, :] = torch.from_numpy(clean_indices.copy())

    return all_indices

def compute_mutual_knn_layer(knn_A, knn_B):
    '''
    mKNN(l,l) = 1/N sum_i^N (|KNN_A(i,l) intersect KNN_B(i,l)| / k)
    '''
    print("knn_A shape:", knn_A.shape)
    print("knn_B shape:", knn_B.shape)

    assert knn_A.shape == knn_B.shape

    k = knn_A.shape[1]

    matches = knn_A.unsqueeze(2) == knn_B.unsqueeze(1)

    overlap = matches.any(dim=2).sum(dim=1)

    per_sample_score = overlap.float() / k

    return per_sample_score.mean().item()

def mutual_knn_all_layers(feats_A, feats_B, topk):

    knnA_indices = compute_knn_for_all_layers(feats_A, topk)
    knnB_indices = compute_knn_for_all_layers(feats_B, topk)

    n_layers_A = knnA_indices.shape[1]
    n_layers_B = knnB_indices.shape[1]

    scores = torch.empty((n_layers_A, n_layers_B), dtype=torch.float32)

    for layer_A in range(n_layers_A):
        for layer_B in range(n_layers_B):

            knn_A = knnA_indices[:, layer_A, :]
            knn_B = knnB_indices[:, layer_B, :]

            score = compute_mutual_knn_layer(knn_A,knn_B,)

            scores[layer_A, layer_B] = score

            print(
                f"Layer {layer_A} vs layer {layer_B}: "
                f"{score:.4f}"
            )

    return scores

def mutual_knn_one_to_one(feats_A, feats_B, topk):
    "calculate the mutualknn accuarcy, but only for one caption per image, then do it over all 5 captions and get mean and std"
    print("feats_A shape:", feats_A.shape)
    print("feats_B shape:", feats_B.shape)
    for i in range(5):
        feats_A_one_caption = feats_A[i::5]  # take every 5th caption starting from i
        feats_B_one_caption = feats_B[i::5]  # take every 5th caption starting from i

        scores = mutual_knn_all_layers(feats_A_one_caption, feats_B_one_caption, topk)

        if i == 0:
            all_scores = scores.unsqueeze(0)
        else:
            all_scores = torch.cat((all_scores, scores.unsqueeze(0)), dim=0)

    return all_scores.mean(dim=0)

def plot_results(results):
    '''
    Results contains mutual KNN and std for each image-text model pair, for each layer combination.
    This function plots the mean mutual KNN for each image-text model pair, with error bars
    representing the standard deviation across the 5 captions per image.
    '''
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
    size_colors = {
        "base":  viridis(0.95),  # yellow
        "large": viridis(0.55),  # teal/green
        "huge":  viridis(0.05),  # purple
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

        family_models = list({
            image_model
            for (image_model, text_model), score in results.items()
            if family in image_model
        })

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

        text_models = [
            text_model
            for (image_model, text_model) in results.keys()
            if image_model == first_model
        ]

        x = np.arange(len(text_models))

        # --------------------------------------------------------
        # Draw each image model
        # --------------------------------------------------------
        for image_model in family_models:

            values = [
                results[(image_model, text_model)]
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
            size for size in ["tiny", "small", "base", "large", "huge", "giant"]
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

def load_results(file_path):
    results = {}
    with open(file_path, "r") as f:
        next(f)  # skip header
        for line in f:
            image_model, text_model, mean_score = line.strip().split(",")
            results[(image_model, text_model)] = [float(mean_score)]
    return results


if __name__ == "__main__":
    test = True

    image_models = get_models("test", "image")
    text_models = get_models("test", "text")

    num_chunks = 10 
    if test == True:
        num_chunks = 1

    file_path = "../plots/results/mutual_knn_results.txt"

    if os.path.exists(file_path):
        results = load_results(file_path)
        plot_results(results)

    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    results = {}
    # for each image model
    for image_model in image_models:
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
                    # print_meta_info(text_model, "text", text_chunk)
                    if loaded_text_data is None:
                        print(f"Warning: No embeddings found for {text_model} (chunk {text_chunk}). Skipping.")
                        continue
                    text_feats = loaded_text_data["avg"]

                    print(f"Loaded text embeddings for {text_model} (chunk {text_chunk}) with shape: {text_feats.shape}")
                    print(f"Loaded image embeddings for {image_model} (chunk {image_chunk}) with shape: {image_feats_40k.shape}")
                    
                    scores = mutual_knn_one_to_one(image_feats_40k, text_feats, topk=10)
                    print(f"Mutual KNN accuracy between {image_model} (chunk {image_chunk}) and {text_model} (chunk {text_chunk}): {scores.max().item() :.4f}")
                    results[(image_model, text_model)] = [scores.max().item()]  

    
    with open(file_path, "w") as f:
        f.write("Image_Model,Text_Model,Mean_Score,Std_Score\n")
        for (image_model, text_model), (max_score) in results.items():
            f.write(f"{image_model},{text_model},{max_score:.4f}\n")

    print("Final results:", results)
    plot_results(results)

