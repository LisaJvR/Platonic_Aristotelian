from pna_data import get_captions_from_index, get_image_files,load_embeddings, normalize_speakers
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from PIL import Image, ImageOps
from pna_metrics import knn_layers
from pna_models import pretty_model_name, get_family_name
from pna_data import clamp_tensor_outliers

rng = np.random.default_rng(5)  # For reproducibility

PLOTS_DIR = "../plots"

def get_data(model_name, modality, chunk_number=0, chunk_size=800):
    print(f"Loading data for model: {model_name}, modality: {modality}, chunk_number: {chunk_number}")
    data = load_embeddings(model_name, modality, chunk_num=chunk_number)
    mapped_df = None  # Initialize mapped_df to None

    if modality in ["text", "speech"]:
        mapped_df = get_captions_from_index(modality)
        mapped_df = mapped_df[chunk_number * chunk_size:(chunk_number + 1) * chunk_size]
        # print("Mapped dataframe shape:", mapped_df.shape)

    if modality == "image":
        index_df = pd.read_csv(f"../embeddings/{modality}/dataset_index.csv")
        mapped_df = index_df[chunk_number * chunk_size:(chunk_number + 1) * chunk_size]

    print("Model:", data["metadata"]["model_name"])
    print("Modality:", data["metadata"]["modality"])
    print("Avg shape:", data["avg"].shape)
    print("Avg dtype:", data["avg"].dtype)

    return data, mapped_df

def check_for_nans(embeddings):
    nan_mask = ~np.isfinite(embeddings)
    per_sample_has_nan = nan_mask.any(axis=(1, 2))
    n_bad = per_sample_has_nan.sum()
    print(f"Samples with any NaN/Inf: {n_bad} / {embeddings.shape[0]}")

    # show first few bad indices
    bad_idx = np.where(per_sample_has_nan)[0]
    print("First bad indices:", bad_idx[:20])


def compute_knn_image(df, index, k):
    embeddings = df["avg"].cpu().numpy()
    image_ids = index["image"].tolist()
    print(len(image_ids), embeddings.shape)

    indices = knn_layers(embeddings, k=k)

    query_with_knns = []
    num_layers = embeddings.shape[1]
    for layer in range(num_layers):

        for i in range(len(image_ids)):
            query_image = image_ids[i]
            knns = indices[i, layer]
            knn_images = [image_ids[idx] for idx in knns]

            query_with_knns.append({
                "query_image": query_image,
                "layer": layer,
                "knn_images": knn_images
            })
    return query_with_knns

def compute_knn_caption(df, index, k, modality="text"):
    print(index)
    embeddings = df["avg"].cpu().numpy()
    image_ids = index["image"].tolist()
    captions = index["caption"].tolist()
    
    print(len(captions), len(image_ids), embeddings.shape)

    indices = knn_layers(embeddings, k=k)

    model_chunk_avg = []

    for layer in range(indices.shape[1]):

        correct_matches = 0
        correct_speaker_matches = 0

        for i in range(len(captions)):
            sample_image = image_ids[i]
            # print(f"Sample image: {sample_image}, Caption: {captions[i]}")

            neighbor_indices = indices[i, layer]
            # print(f"Layer {layer}, Sample {i}: Neighbor indices: {neighbor_indices}")
            neighbor_images = [image_ids[idx] for idx in neighbor_indices]

            correct_matches += sum(
                1 for image in neighbor_images
                if image == sample_image
            )
            correct_captions = [captions[idx] for idx in neighbor_indices]
            # print(f"Neighbor images: {neighbor_images}, Correct matches so far: {correct_matches}, Correct captions: {correct_captions}")
            # check speakers of captions of nearest neighbors

            neighbor_speakers = [
                index.iloc[int(idx)]["speaker"]
                for idx in neighbor_indices
            ]
            # cast to printable numbers
            neighbor_speakers = [int(speaker) for speaker in neighbor_speakers]
            # print(f"Neighbor speakers: {neighbor_speakers}")
            correct_speaker_matches += sum(
                1 for speaker in neighbor_speakers
                if speaker == index.iloc[i]["speaker"]
            )
        total = len(captions) * k

        
        print(
            f"Layer {layer}: "
            f"{correct_matches}/{total} same-image neighbours "
            f"({correct_matches / total:.4f})"
            f" Neigbor speakers: {correct_speaker_matches}/{total} same-speaker neighbours "
        )
        model_chunk_avg.append(correct_matches / total)

    return None, indices, model_chunk_avg

def load_square_image(path, size=256):
    """Center-crop to square for visualization only."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = ImageOps.fit(
            img, (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5)
        )
        return np.asarray(img)

def plot_speaker_mean_embeddings(speaker_means, model_name, type="speaker_means"):
    # Find all layers
    layers = sorted(set(layer for speaker, layer in speaker_means.keys()))
    transformed_embeddings = None  # Initialize transformed_embeddings to None

    plt.figure(figsize=(8, 6))
    avg_pc1_explained = []
    avg_pc2_explained = []
    avg_pc1_explained_ratio = []
    avg_pc2_explained_ratio = []

    for layer in layers:

        # Get all speakers for this layer
        speakers = []
        embeddings = []

        for (speaker, speaker_layer), mean_embedding in speaker_means.items():

            if speaker_layer == layer:
                speakers.append(speaker)

                embeddings.append(
                    mean_embedding.detach().cpu().numpy()
                )

        embeddings = np.stack(embeddings)

        print(
            f"Layer {layer}: "
            f"{len(speakers)} speakers, "
            f"shape = {embeddings.shape}"
        )

        # PCA directly on speaker mean embeddings
        pca = sklearn.decomposition.PCA(n_components=2)

        transformed_embeddings = pca.fit_transform(
            embeddings
        )


        avg_pc1_explained_ratio.append(pca.explained_variance_ratio_[0])
        avg_pc2_explained_ratio.append(pca.explained_variance_ratio_[1])

        avg_pc1_explained.append(pca.explained_variance_[0])
        avg_pc2_explained.append(pca.explained_variance_[1])

        plt.scatter(
            transformed_embeddings[:, 0],
            transformed_embeddings[:, 1],
            alpha=0.7,
            c=np.array(speakers, dtype=float),
        )
    plt.colorbar(label="Speaker number")

    pretty_model_name = get_family_name(modality="speech", model_name=model_name) + " " + pretty_model_name(model_name)
    plt.title(
            f"PCA of Speaker Mean Embeddings ({type})\n"
            f"{(model_name)} - All layers"
        )

    plt.xlabel(
            f"PC1 ({np.mean(avg_pc1_explained_ratio):.1%})"
        )

    plt.ylabel(
            f"PC2 ({np.mean(avg_pc2_explained_ratio):.1%})"
        )

    plt.grid()

    plt.savefig(
            f"{PLOTS_DIR}/knn/"
            f"pca_speaker_means_{model_name}_layer_all_{type}.png"
        )
    print("Mean PC1 explained variance:", np.mean(avg_pc1_explained))
    print("Mean PC2 explained variance:", np.mean(avg_pc2_explained))

    print("Mean norm:", np.linalg.norm(embeddings, axis=1).mean())
    print("Max norm:", np.linalg.norm(embeddings, axis=1).max())

    plt.close()

def compute_speaker_knn(indices, index):
    speakers = index["speaker"].to_numpy()

    layer_scores = []

    for layer in range(indices.shape[1]):
        same_speaker = 0
        total = 0

        for i in range(len(speakers)):
            neighbor_indices = indices[i, layer]

            same_speaker += np.sum(
                speakers[neighbor_indices] == speakers[i]
            )

            total += len(neighbor_indices)

        score = same_speaker / total
        layer_scores.append(score)

        print(
            f"Layer {layer}: "
            f"same-speaker KNN = {score:.4f}"
        )

    return layer_scores

def compare_clipped_to_non_clipped():
    from pna_experiment import experiment_driver, run_experiment

    print("Running experiments for clipped and non-clipped embeddings...")
    run_experiment(
        ["vit_base_patch16_224.mae"],
        ["bert-base-uncased"],
        modalities=["image", "text"],
        num_chunks=10,
        file_path="../results/cka_image_text_not_clipped.csv",
        type="cka",
        cka_type="linear",
        rbf_sigma=1.0,
        biased=False,
        clip=False,
        exact=False,
        q=0.9
    )

    print("Running experiments for clipped embeddings...")
    run_experiment(
        ["vit_base_patch16_224.mae"],
        ["bert-base-uncased"],
        modalities=["image", "text"],
        num_chunks=10,
        file_path="../results/cka_image_text_clipped.csv",
        type="cka",
        cka_type="linear",
        rbf_sigma=1.0,
        biased=False,
        clip=True,
        exact=False,
        q=0.9
    )

    print("Running experiments for mknn with clipped embeddings...")
    run_experiment(
        ["vit_base_patch16_224.mae"],
        ["bert-base-uncased"],
        modalities=["image", "text"],
        num_chunks=10,
        file_path="../results/mknn_image_text_clipped.csv",
        type="mknn",
        rbf_sigma=1.0,
        biased=False,
        clip=True,
        exact=False,
        q=0.9
    )

    print("Running experiments for mknn with non-clipped embeddings...")
    run_experiment(
        ["vit_base_patch16_224.mae"],
        ["bert-base-uncased"],
        modalities=["image", "text"],
        num_chunks=10,
        file_path="../results/mknn_image_text_not_clipped.csv",
        type="mknn",
        rbf_sigma=1.0,
        biased=False,
        clip=False,
        exact=False,
        q=0.9
    )

    # read the results and compare
    cka_clipped = pd.read_csv("../results/cka_image_text_clipped.csv")
    cka_not_clipped = pd.read_csv("../results/cka_image_text_not_clipped.csv")
    mknn_clipped = pd.read_csv("../results/mknn_image_text_clipped.csv")
    mknn_not_clipped = pd.read_csv("../results/mknn_image_text_not_clipped.csv")

    print("CKA results comparison:")
    print(cka_clipped)
    print(cka_not_clipped) 
    print("Difference in CKA results (clipped - not clipped):")
    print(cka_clipped.sub(cka_not_clipped))

    print("MKNN results comparison:")
    print(mknn_clipped)
    print(mknn_not_clipped)
    print("Difference in MKNN results (clipped - not clipped):")
    print(mknn_clipped.sub(mknn_not_clipped))

    return None

if __name__ == "__main__":
    compare_clipped_to_non_clipped()

    parser = argparse.ArgumentParser(description="Sanity checks for embeddings")
    parser.add_argument("--model_name",default="vit_base_patch16_224.mae", type=str, help="Model name")
    parser.add_argument("--modality",default="image", type=str, choices=["image", "text", "speech"], help="Modality")
    args = parser.parse_args()

    model_name = args.model_name
    modality = args.modality

    chunk_numbers = np.arange(0,1)
    
    os.makedirs(f"{PLOTS_DIR}/knn", exist_ok=True)
    mapped_df = None
    # save KNN image
    if modality in ["speech", "text"]:
        plt.figure(figsize=(10, 6))
        plt.title(f'KNN Performance per Layer for {model_name} Chunk {chunk_numbers}')
        for chunk_number in chunk_numbers:
            data, mapped_df = get_data(model_name, modality, chunk_number=chunk_number)
            check_for_nans(data["avg"].cpu().numpy())
            norm_layer_scores = None
            if modality == "speech":
               
                print(f"Normalize speakers...")
                norm_data, speaker_means, centralized_means = normalize_speakers(data, mapped_df)

                plot_speaker_mean_embeddings(speaker_means, model_name, type="speaker_means")
                plot_speaker_mean_embeddings(centralized_means, model_name, type="centralized_means")
                _, norm_indices, model_chunk_avg_norm = compute_knn_caption(df=norm_data, index=mapped_df, k=4, modality=modality)
                plt.plot(range(norm_indices.shape[1]), model_chunk_avg_norm, marker='x', label=f'Chunk {chunk_number} (n)')
                  # You can pass a DataFrame if you have one, or keep it None to use the default loading mechanism
                print(f"Model chunk average: {np.mean(model_chunk_avg_norm):.4f}")

                norm_layer_scores = compute_speaker_knn(norm_indices, mapped_df)
                
            _, indices, model_chunk_avg = compute_knn_caption(df=data, index=mapped_df, k=4, modality=modality)
            layer_scores = compute_speaker_knn(indices, mapped_df)
            for layer, score in enumerate(layer_scores):
                if norm_layer_scores is not None:
                    print(f"Layer {layer}: same-speaker KNN = {score:.4f}, centralized = {norm_layer_scores[layer]:.4f}")
            print(f"Model chunk average: {np.mean(model_chunk_avg):.4f}")

            plt.plot(range(indices.shape[1]), model_chunk_avg, marker='o', label=f'Chunk {chunk_number}')
            plt.xticks(range(indices.shape[1]))
        plt.legend()
        plt.xlabel('Layer')
        plt.ylabel('Fraction of Same-Image Neighbours')
        plt.grid()
        plt.savefig(f"{PLOTS_DIR}/knn/performance_layer_model_chunk_{modality}_{model_name}_{chunk_numbers[0]}_{chunk_numbers[-1]}.png")
    
    if modality == "image":
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        chunk_number = 0
        data, mapped_df = get_data(
            model_name,
            modality,
            chunk_number=chunk_number
        )

        embeddings = data["avg"].cpu().numpy()
        check_for_nans(embeddings)

        num_layers = embeddings.shape[1]

        layers_to_show = [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1
        ]

        q_knns = compute_knn_image(
            df=data,
            index=mapped_df,
            k=4
        )

        # Pick one query image from layer 0
        layer0_queries = [
            q for q in q_knns
            if q["layer"] == 0
        ]

        selected = rng.choice(layer0_queries)
        query_image = selected["query_image"]

        # Retrieve the same query for every layer
        query_per_layer = []

        for layer in layers_to_show:
            matches = [
                q for q in q_knns
                if q["layer"] == layer
                and q["query_image"] == query_image
            ]

            if not matches:
                raise ValueError(
                    f"No match for query {query_image} "
                    f"at layer {layer}"
                )

            query_per_layer.append(matches[0])

        n_cols = 5

        # Slightly wider than your original version
        fig, axes = plt.subplots(
            len(layers_to_show),
            n_cols,
            figsize=(8.5, 1.45 * len(layers_to_show)),
            squeeze=False,
            dpi=150
        )

        # -------------------------
        # Column headings
        # -------------------------
        column_titles = [
            "Query",
            "NN 1",
            "NN 2",
            "NN 3",
            "NN 4",
        ]

        for col, title in enumerate(column_titles):
            axes[0, col].set_title(
            title,
            fontsize=10,
            fontweight="semibold",
            color="0.5",      # grey
            pad=8
        )

        # -------------------------
        # Plot every layer
        # -------------------------
        for row, query_result in enumerate(query_per_layer):

            layer = query_result["layer"]

            image_ids = [
                query_result["query_image"],
                *query_result["knn_images"][:4]
            ]

            image_paths = get_image_files(image_ids)

            for col, image_path in enumerate(image_paths):
                img = load_square_image(image_path, size=256)

                ax = axes[row, col]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")


                for spine in ax.spines.values():
                    spine.set_visible(False)

            axes[row, 0].set_ylabel(
                f"Layer {layer}",
                fontsize=10,          # increase from 8
                fontweight="medium",
                color="0.4",          # grey
                rotation=0,
                labelpad=32,
                va="center",
                ha="right"
            )

        # -------------------------
        # Layout
        # -------------------------
        fig.subplots_adjust(
    left=0.11,
    right=0.995,
    bottom=0.02,
    top=0.92,
    wspace=0.055,
    hspace=0.10
)

        # Smaller, cleaner title
        clean_model_name = get_family_name(modality, model_name) + " " + pretty_model_name(model_name)

        fig.suptitle(
            f"Layer-wise nearest-neighbour retrieval — {clean_model_name}",
            fontsize=11,
            fontweight="semibold",
            y=0.99
        )

        # -------------------------
        # Vertical separator
        # between query and NN columns
        # -------------------------
        query_right = axes[0, 0].get_position().x1
        nn_left = axes[0, 1].get_position().x0

        separator_x = (
            query_right + nn_left
        ) / 2

        line = plt.Line2D(
            [separator_x, separator_x],
            [0.02, 0.955],
            transform=fig.transFigure,
            linewidth=1.5,
            color="0.75"
        )

        fig.add_artist(line)

        output_base = (
            f"{PLOTS_DIR}/knn/"
            f"nearest_neighbour_image_retrieval_"
            f"{modality}_{model_name}_chunk{chunk_number}"
        )

        fig.savefig(
            output_base + ".png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white"
        )

        plt.close(fig)

