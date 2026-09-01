import os
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pna_models import get_size, image_family, pretty_model_name, text_family, speech_family

from platonic_plot_estimates import get_platonic_trend

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

image_families = [
        "dinov2",
        "clip",
        "clip (12K ft)",
        "mae",
        "imagenet21k",
        "data2vec-vision",
    ]

speech_families = [
        "wav2vec2",
        "hubert",
        "wavlm",
        "unispeech"
    ]

def get_reg_coeffs(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_, model.intercept_

def belongs_to_family(image_model, family, modalities):
    if modalities[0] == "image":
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
    if modalities[0] == "speech":
        return family in image_model


def plot_results(results, type, modalities):
    os.makedirs("../plots", exist_ok=True)

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

    families = []
    if modalities[0] == "image":
        families = image_families
    elif modalities[0] == "speech":
        families = speech_families
    else:
        print(f"Warning: Unknown modality {modalities[0]}. No plots will be generated.")

    
    
    for family  in families:
        family_models = list({
            y_model
            for (y_model, x_model), score in results.items()
            if belongs_to_family(y_model, family, modalities)
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

        x_models = [
            x_model
            for (y_model, x_model) in results.keys()
            if y_model == first_model
        ]

        x = np.arange(len(x_models))

        all_data = []
        for y_model in family_models:

            values = [
                results[(y_model, x_model)]
                for x_model in x_models
            ]

            size = get_size(y_model)
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
        print(f"all_x: {all_x}, all_y: {all_y}")
        coeff, intercept = get_reg_coeffs(all_x.reshape(-1, 1), all_y.reshape(-1, 1))
        x_trend = np.sort(all_x)
        y_trend = coeff[0][0] * x_trend + intercept[0]

        if (modalities[0] == "image") & (modalities[1] == "text"):    
            all_coeffs, avg_family_coeffs = get_platonic_trend(x,"Platonic", metric=type)
            plat_y  = all_coeffs[(family, size)] * x_trend + intercept[0]

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

        ax.set_xticks(x)
        ax.set_xticklabels(
                [pretty_model_name(m) for m in x_models],
                rotation=50,
                ha="right",
                rotation_mode="anchor",
                color=text_color,
            )
        ax.set_ylabel(
        f"Alignment to {family.upper()}",
        fontsize=22,
        color=text_color,
        )

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    
        ax.tick_params(
                axis="both",
                direction="out",
                length=8,
                width=2,
                colors=text_color,
            )

        ax.grid(True, axis="both")
        ax.set_axisbelow(True)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        ax.spines["left"].set_color("#c3c9ce")
        ax.spines["bottom"].set_color("#c3c9ce")
        
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

        handles, labels = ax.get_legend_handles_labels()

        # remove repeated labels if necessary
        unique = {}
        for handle, label in zip(handles, labels):
                unique[label] = handle

        desired_order = []
        if modalities[0] == "image":
            desired_order = [
                size for size in ["tiny", "small", "base", "large", "huge", "giant"] 
                if size in unique
            ]
            if (modalities[0] == "image") & (modalities[1] == "text"):
                desired_order.append("expected = {:.4f}x ".format(avg_family_coeffs[family]))
        elif modalities[0] == "speech":
            desired_order = ["base", "large", "xlarge"]

        desired_order.append("observed = {:.4f}x ".format(coeff[0][0]))
        print(f"Desired order: {desired_order}, unique: {unique}")
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

        #--------------------------------------------------------
        # Add  group names
        #--------------------------------------------------------
        families_x = {}
        f = None
        for i, model in enumerate(x_models):
            if modalities[1] == "text":
                f = text_family(model)
            elif modalities[1] == "speech":
                f = speech_family(model)

            if f:
                families_x.setdefault(f, []).append(i)
            # print(f"families_x: {families_x}, {modalities}, {x_models}, {model}, {f}")

        for x_fam, positions in families_x.items():
                centre = np.mean(positions)

                ax.text(
                    centre,
                    -0.19,
                    x_fam,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=15,
                    color=text_color,
                )
        plt.subplots_adjust(
                        left=0.27,
                        right=0.97,
                        top=0.97,
                        bottom=0.25,
                    )
            
        os.makedirs("../plots/results/", exist_ok=True)
        plt.savefig(
                        f"../plots/results/{type}_{family}_{modalities[0]}_{modalities[1]}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
            
        plt.close(fig)