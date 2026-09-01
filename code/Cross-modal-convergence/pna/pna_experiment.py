import torch
from pna_models import get_models
from pna_data import load_all_chunks
from pna_metrics import knn_layers, mutual_knn_layers, cka_layers
from pna_plotting_code import plot_image_text_results, plot_results, get_reg_coeffs
import os
from tqdm import tqdm


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
        plot_results(results, f"{type}{'_' + cka_type if cka_type else ''}{'_sigma' + str(rbf_sigma) if type == 'cka' and cka_type == 'rbf' else ''}{'_biased' if biased else ''}", modalities)
        print(f"Results already exist in {file_path}. Loaded and plotted existing results.")
        return None
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
    plot_results(results, f"{modalities[0]}_{modalities[1]}_{type}{'_' + cka_type if cka_type else ''}{'_sigma' + str(rbf_sigma) if type == 'cka' and cka_type == 'rbf' else ''}{'_biased' if biased else ''} ", modalities)
    return None

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
    # run_experiment(image_models, text_models,modalities=["image", "text"], num_chunks=num_chunks, topk=10,type="mknn", file_path="../plots/results/knn/mutual_knn_results_image_text.txt")
    # CKA linear & RBF
    # run_experiment(image_models,modalities=["image", "text"], num_chunks=num_chunks, type="cka", cka_type="linear", biased=False, file_path="../plots/results/cka/cka_linear_results_unbiased_image_text.txt")
    # run_experiment(image_models, text_models,modalities=["image", "text"], num_chunks=num_chunks, type="cka", cka_type="rbf", rbf_sigma=1.0, biased=False, file_path="../plots/results/cka/cka_rbf_results_unbiased_image_text.txt")

    
    modalities = ["image", "speech"]
    # image-speech convergence
    run_experiment(image_models, speech_models,modalities, num_chunks=num_chunks, topk=10,type="mknn", file_path=f"../plots/results/knn/mutual_knn_results_{modalities[0]}_{modalities[1]}.txt")
    # CKA linear & RBF
    # run_experiment(image_models, speech_models,modalities, num_chunks=num_chunks, type="cka", cka_type="linear", biased=False, file_path=f"../plots/results/cka/cka_linear_results_unbiased_{modalities[0]}_{modalities[1]}.txt")
    # run_experiment(image_models, speech_models,modalities, num_chunks=num_chunks, type="cka", cka_type="rbf", rbf_sigma=1.0, biased=False, file_path=f"../plots/results/cka/cka_rbf_results_unbiased_{modalities[0]}_{modalities[1]}.txt")


    # image-text convergence
