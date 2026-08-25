# image-text, image-speech, text-speech experiment repeat from paltonian paper

# get text embeddings and image embeddings and calculate mKNN and plot

import torch
from pna_models import get_models, pretty_model_name, pretty_text_name, get_size, text_family
from pna_data import load_embeddings, print_meta_info, get_captions_from_index, load_all_chunks
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import numpy as np

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
            
def compute_cka_layer(feats_A, feats_B, type="linear", rbf_sigma=1.0, u=False):
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

def cka_all_layers(feats_A, feats_B, type="linear", rbf_sigma=1.0):
    n_layers_A = feats_A.shape[1]
    n_layers_B = feats_B.shape[1]

    scores = torch.empty((n_layers_A, n_layers_B), dtype=torch.float32)

    for layer_A in range(n_layers_A):
        for layer_B in range(n_layers_B):
            score = compute_cka_layer(feats_A[:, layer_A, :], feats_B[:, layer_B, :], type=type, rbf_sigma=rbf_sigma)
            scores[layer_A, layer_B] = score

    return scores
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

    n_samples, n_layers, dim = embeddings.shape

    all_indices = torch.empty(
        (n_samples, n_layers, k),
        dtype=torch.long,
    )

    
    for layer in range(n_layers):
        layer_embeddings = embeddings[:, layer, :].copy()

        # XXX
        if use_gpu :
            import faiss
            if faiss.get_num_gpus() > 0:
                faiss.normalize_L2(layer_embeddings)
                index = faiss.IndexFlatL2(dim)
                
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(resources,0,index,) #move to GPU
                index.add(layer_embeddings)
                _, indices = index.search(layer_embeddings, k + 1)
        else: 
            """ Pure AI implementation for CPU (no faiss) """
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import normalize

            layer_embeddings = normalize(
                layer_embeddings,
                norm="l2",
                axis=1
            )

            if k >= n_samples:# XXX
                k = n_samples - 1
                print(f"Warning: k ({k}) is greater than or equal to the number of samples ({n_samples}). Adjusting k to {k}.")

            index = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
            index.fit(layer_embeddings)

            _, indices = index.kneighbors(layer_embeddings)

        clean_indices = np.empty((n_samples, k), dtype=np.int64)
        for i in range(n_samples):
            neighbours = indices[i][indices[i] != i]
            clean_indices[i] = neighbours[:k]

        all_indices[:, layer, :] = torch.from_numpy(clean_indices.copy())

    return all_indices

def compute_mutual_knn_layer(knn_A, knn_B):
    '''
    mKNN(l,l) = 1/N sum_i^N (|KNN_A(i,l) intersect KNN_B(i,l)| / k)
    '''

    assert knn_A.shape == knn_B.shape

    k = knn_A.shape[1]

    matches = knn_A.unsqueeze(2) == knn_B.unsqueeze(1)

    overlap = matches.any(dim=2).sum(dim=1)

    per_sample_score = overlap.float() / k

    return per_sample_score.mean().item()

def mutual_knn_all_layers(knnA_indices, knnB_indices,num_image_layers, num_text_layers, topk):
    n_layers_A = num_image_layers
    n_layers_B = num_text_layers

    scores = torch.empty((n_layers_A, n_layers_B), dtype=torch.float32)

    for layer_A in range(n_layers_A):
        for layer_B in range(n_layers_B):

            knn_A = knnA_indices[:, layer_A, :]
            knn_B = knnB_indices[:, layer_B, :]

            score = compute_mutual_knn_layer(knn_A,knn_B,)

            scores[layer_A, layer_B] = score

    return scores

def scores_one_to_one(feats_image, text_model, topk, type="knn",subtype="linear", rbf_sigma=1.0, num_chunks=10):
    "calculate the mutualknn accuarcy, but only for one caption per image, then do it over all 5 captions and get mean and std"
    if type == "knn":
        image_knn = compute_knn_for_all_layers(feats_image, topk)

    for i in range(5):
        feats_text = load_all_chunks(text_model, "text", num_chunks=num_chunks, caption_number=i)

        if type == "knn":
            text_knn = compute_knn_for_all_layers(feats_text, topk)
            scores = mutual_knn_all_layers(image_knn, text_knn,num_image_layers=feats_image.shape[1], num_text_layers=feats_text.shape[1], topk=topk)
        elif type == "cka":
            scores = cka_all_layers(feats_image, feats_text, subtype, rbf_sigma)

        if i == 0:
            all_scores = scores.unsqueeze(0)
        else:
            all_scores = torch.cat((all_scores, scores.unsqueeze(0)), dim=0)

    return all_scores.mean(dim=0)

def run_experiment(image_models, text_models, file_path, num_chunks=10, topk=10, type="knn", cka_type=None, rbf_sigma=1.0):
    
    results = {}

    if len(image_models) == 0:
        raise ValueError("No image models were provided. Check your modelset configuration.")

    if len(text_models) == 0:
        raise ValueError(
            "No text models were provided. Your modelset has an empty text list, so no kNN pairs can be computed. "
            "Update MODELSETS in pna_models.py."
        )

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        results = load_results(file_path)
        plot_results(results)
        print(f"Results already exist in {file_path}. Loaded and plotted existing results.")
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("Image_Model,Text_Model,Max_Score\n")


    for image_model in tqdm(image_models, desc="Processing image models"):

        image_feats = load_all_chunks(image_model, "image", num_chunks=num_chunks)

        if image_feats is None:
            print(f"Warning: No embeddings found for {image_model} Skipping.")
            continue

        for text_model in text_models:
            if image_model in results and text_model in results:
                continue

            # type should indicate it what to do
            scores = scores_one_to_one(image_feats, text_model, topk=topk, type=type, subtype=cka_type, rbf_sigma=rbf_sigma, num_chunks=num_chunks)

            results[(image_model, text_model)] = scores.max().item()

            with open(file_path, "a") as f:
                f.write(f"{image_model},{text_model},{scores.max().item():.2f}\n")

    if len(results) == 0:
        print(
            "Warning: No results were computed. "
            "Check that embeddings exist for the selected models/chunks and that both model lists are non-empty."
        )

    print(f"Final results {type}:", results)
    plot_results(results)
    return None

def belongs_to_family(image_model, family):
    if family == "imagenet21k":
        return (
            "augreg" in image_model
        )
    if family == "clip":
        return (
            "_clip_" in image_model
            and "_ft_in12k" not in image_model
        )

    if family == "clip (12K ft)":
        return (
            "_clip_" in image_model
            and "_ft_in12k" in image_model
        )

    return family in image_model

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

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
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
        "clip (12K ft)",
        "mae",
        "imagenet21k",
        "data2vec-vision",
    ]

    for family in families:

        family_models = list({
            image_model
            for (image_model, text_model), score in results.items()
            if belongs_to_family(image_model, family)
        })

        if not family_models:
            continue

        fig, ax = plt.subplots(figsize=(4.7, 5.5), dpi=300)

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

    # mKNN
    run_experiment(image_models, text_models, num_chunks=num_chunks, topk=10,type="knn", file_path="../plots/results/knn/mutual_knn_results.txt")
    # CKA linear & RBF
    # run_experiment(image_models, text_models, num_chunks=num_chunks, type="cka", cka_type="linear", file_path="../plots/results/cka/cka_linear_results.txt")
    # run_experiment(image_models, text_models, num_chunks=num_chunks, type="cka", cka_type="rbf", rbf_sigma=1.0, file_path="../plots/results/cka/cka_rbf_results.txt")

    

