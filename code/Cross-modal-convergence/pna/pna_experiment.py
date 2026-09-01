import torch
from pna_models import get_models, pretty_text_name, get_size, text_family
from pna_data import load_all_chunks
from pna_metrics import knn_layers, mutual_knn_layers, cka_layers
from platonic_plot_estimates import get_platonic_trend
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib import cm
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import numpy as np

def calculate_score(feats_image, text_model, modalities, topk, type="mknn",subtype="linear", rbf_sigma=1.0, num_chunks=10, biased=False):
    "calculate the mutualknn accuarcy, but only for one caption per image, then do it over all 5 captions and get mean and std"
    if type == "mknn":
        image_knn = knn_layers(feats_image, topk)

    for i in range(5):
        feats_text = load_all_chunks(text_model, modalities[1], num_chunks=num_chunks, caption_number=i)

        if type == "mknn":
            text_knn = knn_layers(feats_text, topk)
            scores = mutual_knn_layers(image_knn, text_knn,num_image_layers=feats_image.shape[1], num_text_layers=feats_text.shape[1], topk=topk)
        elif type == "cka":
            scores = cka_layers(feats_image, feats_text, subtype, rbf_sigma, biased=biased)

        if i == 0:
            all_scores = scores.unsqueeze(0)
        else:
            all_scores = torch.cat((all_scores, scores.unsqueeze(0)), dim=0)

    return all_scores.mean(dim=0)

def run_experiment(image_models, text_models,modalities, file_path, num_chunks=10, topk=10, type="mknn", cka_type=None, rbf_sigma=1.0, biased=False):
    
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
        plot_results(results, f"{type}{'_' + cka_type if cka_type else ''}{'_sigma' + str(rbf_sigma) if type == 'cka' and cka_type == 'rbf' else ''}{'_biased' if biased else ''}")
        print(f"Results already exist in {file_path}. Loaded and plotted existing results.")
        # return None
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("Image_Model,Text_Model,Max_Score\n")


    for image_model in tqdm(image_models, desc=f"{type}: Processing image model:", leave=False):
        image_feats = load_all_chunks(image_model, modalities[0], num_chunks=num_chunks)

        if image_feats is None:
            print(f"Warning: No embeddings found for {image_model} Skipping.")
            continue

        for text_model in tqdm(text_models, desc=f"{type}: Processing text model:", leave=False):
            if (image_model, text_model) in results:
                print(f"Skipping {image_model} and {text_model} as results already exist.")
                continue

            scores = calculate_score(image_feats, text_model,modalities=modalities, topk=topk, type=type, subtype=cka_type, rbf_sigma=rbf_sigma, biased=biased, num_chunks=num_chunks)
            results[(image_model, text_model)] = scores.max().item()

            with open(file_path, "a") as f:
                f.write(f"{image_model},{text_model},{scores.max().item():.4f}\n")

    if len(results) == 0:
        print(
            "Warning: No results were computed. "
            "Check that embeddings exist for the selected models/chunks and that both model lists are non-empty."
        )

    print(f"Final results {type}:", results)
    plot_results(results, f"{modalities[0]}_{modalities[1]}_{type}{'_' + cka_type if cka_type else ''}{'_sigma' + str(rbf_sigma) if type == 'cka' and cka_type == 'rbf' else ''}{'_biased' if biased else ''} ")
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

def plot_results(results, type, modalities=["image", "text"]):
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
        all_data =[]
        for image_model in family_models:

            values = [
                results[(image_model, text_model)]
                for text_model in text_models
                # if (image_model, text_model) in results
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

            all_data.append((x, values))

        all_x = np.concatenate([data[0] for data in all_data])
        all_y = np.concatenate([data[1] for data in all_data])
        coeff, intercept = get_reg_coeffs(all_x.reshape(-1, 1), all_y.reshape(-1, 1))
        x_trend = np.sort(all_x)
        y_trend = coeff[0][0] * x_trend + intercept[0]

        
        all_coeffs, avg_family_coeffs = get_platonic_trend(x,"Platonic", metric=type)

        print(f"Family: {family}, Image Model: {size}, Coefficients:{all_coeffs[(family, size)]} , Average Family Coefficient: {avg_family_coeffs[family]}")

        plat_y  = all_coeffs[(family, size)] * x_trend + intercept[0]

        print(f"All coeefs: {all_coeffs}")

        ax.fill_between(
            x_trend,
            plat_y,
            y_trend,
            color="lightgrey",
            alpha=0.5,
            zorder=1,
        )
        ax.plot(
            x_trend,
            plat_y,
            color="grey",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label="expected = {:.4f}x ".format(avg_family_coeffs[family])
        )

        ax.plot(
            x_trend,
            y_trend,
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label="observed = {:.4f}x ".format(coeff[0][0])
            )   
        print("Plat y: ", plat_y)
        print("Trend y: ", y_trend)

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

        # Show y-axis ticks with exactly 2 decimals.
        # ax.yaxis.set_major_locator(MultipleLocator(0.005))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

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
        desired_order.append("observed = {:.4f}x ".format(coeff[0][0]))
        desired_order.append("expected = {:.4f}x ".format(avg_family_coeffs[family]))

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
            f"../plots/results/{type}_{family}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

def get_reg_coeffs(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_, model.intercept_

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
    speech_models = get_models("test", "speech")

    num_chunks = 10 
    if test == True:
        num_chunks = 1


    # BASELINES
    # image-text convergence
    # mKNN
    # run_experiment(image_models, text_models,modalities=["image", "text"], num_chunks=num_chunks, topk=10,type="mknn", file_path="../plots/text_image/knn/mutual_knn_results.txt")
    # CKA linear & RBF
    # run_experiment(image_models,modalities=["image", "text"], num_chunks=num_chunks, type="cka", cka_type="linear", biased=False, file_path="../plots/text_image/cka/cka_linear_results_unbiased.txt")
    # run_experiment(image_models, text_models,modalities=["image", "text"], num_chunks=num_chunks, type="cka", cka_type="rbf", rbf_sigma=1.0, biased=False, file_path="../plots/results/cka/cka_rbf_results_unbiased.txt")

    

    # image-speech convergence
    run_experiment(image_models, speech_models,modalities=["image", "speech"], num_chunks=num_chunks, topk=10,type="mknn", file_path="../plots/image_speech/knn/mutual_knn_results.txt")
    # CKA linear & RBF
    # run_experiment(image_models, speech_models,modalities=["image", "speech"], num_chunks=num_chunks, type="cka", cka_type="linear", biased=False, file_path="../plots/image_speech/cka/cka_linear_results_unbiased.txt")
    # run_experiment(image_models, speech_models,modalities=["image", "speech"], num_chunks=num_chunks, type="cka", cka_type="rbf", rbf_sigma=1.0, biased=False, file_path="../plots/results/cka/cka_rbf_results_unbiased.txt")


    # image-text convergence
