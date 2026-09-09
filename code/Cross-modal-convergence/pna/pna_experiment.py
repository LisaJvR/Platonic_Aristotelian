import torch
from pna_models import get_models
from pna_data import load_all_chunks
from pna_metrics import  knn_layers, compare_layers, compute_cka
from pna_plotting_code import  plot_results, get_reg_coeffs
import os
from tqdm import tqdm
from calibrated_similarity import calibrate, calibrate_layers
from metrics_config import METRICS
import csv
import os


RESULT_COLUMNS = [
    "modality_a",
    "modality_b",
    "model_a",
    "model_b",
    "metric",
    "raw_score",
    "calibrated_score",
    "p_value",
    "threshold",
]

def load_results(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=RESULT_COLUMNS,
            )
            writer.writeheader()

        return []

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_result(file_path, result):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            key: result.get(key)
            for key in RESULT_COLUMNS
        })

def result_exists(
    results,
    model_a,
    model_b,
    metric,
    require_calibrated=False,
):
    for row in results:

        if (
            row["model_a"] != model_a
            or row["model_b"] != model_b
            or row["metric"] != metric
        ):
            continue

        if require_calibrated:
            value = row.get("calibrated_score")

            return value not in (
                None,
                "",
                "None",
            )

        return True

    return False


def load_aligned_features(model,modality, n_sets,num_chunks,clip=False,exact=False,q=0.9,):

    return [
        load_all_chunks(
            model,
            modality,
            num_chunks=num_chunks,
            caption_number=i,
            clip=clip,
            exact=exact,
            q=q,
        )
        for i in range(n_sets)
    ] # XXX could be an issue

def compute_raw_scores(feats_A, feats_B_list, metric_fn, metric_kwargs):
    # matrix of scores
    matrices = [
        compare_layers(
            feats_A,
            feats_B,
            metric_fn,
            metric_kwargs,
        )
        for feats_B in feats_B_list
    ]

    # [n_sets, L_A, L_B] 
    matrices = torch.stack(matrices)

    # image-text: mean of 5
    # speech-text: mean of 1
    mean_matrix = matrices.mean(dim=0)

    return mean_matrix

def compute_calibrated_score(
    feats_A,
    feats_B_list,
    metric_fn,
    metric_kwargs,
    K=200,
):
    n_layers_A = feats_A.shape[1]
    n_layers_B = feats_B_list[0].shape[1]
    n_sets = len(feats_B_list)

    X_layers = [
        feats_A[:, i, :]
        for i in range(n_layers_A)
    ]

    Y_layers = [
        feats_B[:, j, :]
        for feats_B in feats_B_list
        for j in range(n_layers_B)
    ]

    def similarity(X, Y):
        return metric_fn(
            X,
            Y,
            **metric_kwargs,
        )

    def aggregate(S):

        S = S.reshape(
            n_layers_A,
            n_sets,
            n_layers_B,
        )

        # Same aggregation as raw experiment
        S = S.mean(dim=1)

        return S.max()

    return calibrate_layers(
        X_layers,
        Y_layers,
        similarity,
        agg=aggregate,
        K=K,
    )

def evaluate_pair(
    feats_A,
    feats_B_list,
    metric_config,
    calibrate=False,
):
    # new calculate score code

    metric_fn = metric_config["fn"]
    kwargs = metric_config["kwargs"]

    layer_scores = compute_raw_scores(
        feats_A,
        feats_B_list,
        metric_fn,
        kwargs,
    )

    # XXX issue 
    # raw_score: model_name, model_name, max_score
    # layer_scores: model_name, model_name, [L_A, L_B]
    print(f"Layer scores: {layer_scores}")
    print(f"Layer scores max: {layer_scores.max()}")
    print(f"Layer scores max item: {layer_scores.max().item()}")

    
    result = {
        "raw_score": layer_scores.max().item(),
        "layer_scores": layer_scores,
    }

    if calibrate:

        calibrated, p, threshold = compute_calibrated_score(
            feats_A,
            feats_B_list,
            metric_fn,
            kwargs,
        )

        # XXX issue
        # raw_score: model_name, model_name, max_score
        # layer_scores: model_name, model_name, [L_A, L_B]
        result.update({
            "calibrated_score": calibrated.item(),
            "p_value": p.item(),
            "threshold": threshold.item(),
        })
    else:
        result.update({
            "calibrated_score": None,
            "p_value": None,
            "threshold": None,
        })

    return result

def run_experiment(
    experiment_name,
    metric_name,
    EXPERIMENTS,
    model_set="all",
    num_chunks=10,
    clip=False,
    exact=False,
    q=0.9,
    calibrate=False,
    calibration_K=200,
    results_file="../plots/results/results.csv",
):

    experiment = EXPERIMENTS[experiment_name]
    metric_config = METRICS[metric_name]

    modality_a, modality_b = experiment["modalities"]
    n_sets = experiment["n_sets"]

    models_A = get_models(model_set, modality_a)
    models_B = get_models(model_set, modality_b)

    existing_results = load_results(results_file)

    for model_A in tqdm(
        models_A,
        desc=f"{metric_name}: {modality_a}",
    ):

        # Load A once because it is reused against every B model.
        feats_A = load_all_chunks(
            model_A,
            modality_a,
            num_chunks=num_chunks,
            clip=clip,
            exact=exact,
            q=q,
        )

        if feats_A is None:
            print(f"Missing embeddings: {model_A}")
            continue

        for model_B in models_B:

            if result_exists(
                existing_results,
                model_A,
                model_B,
                metric_name,
                require_calibrated=calibrate,
            ):
                continue

            # load all chunks
            feats_B_list = load_aligned_features(
                model=model_B,
                modality=modality_b,
                n_sets=n_sets,
                num_chunks=num_chunks,
                clip=clip,
                exact=exact,
                q=q,
            )

            if any(feats is None for feats in feats_B_list):
                print(f"Missing embeddings: {model_B}")
                continue

            scores = evaluate_pair(
                feats_A,
                feats_B_list,
                metric_config,
                calibrate=calibrate,
                # calibration_K=calibration_K,
            )

            result = {
                "modality_a": modality_a,
                "modality_b": modality_b,
                "model_a": model_A,
                "model_b": model_B,
                "metric": metric_name,
                "raw_score": scores["raw_score"],
                "calibrated_score": scores["calibrated_score"],
                "p_value": scores["p_value"],
                "threshold": scores["threshold"],
            }

            save_result(
                results_file,
                result,
            )

            # Add immediately so duplicate checks remain correct
            # during this run.
            existing_results.append(result)
            # existing_results.append({
            #     key: "" if value is None else str(value)
            #     for key, value in result.items()
            # })


# def run_experiment_old(image_models, text_models, modalities, file_path, num_chunks=10, topk=10, type="mknn", cka_type=None, rbf_sigma=1.0, biased=False, exact=False, q=0.9, clip=False, _calibrate=False):
#     results = {}
#     c_results = {}

#     if len(image_models) == 0:
#         raise ValueError("No image models were provided. Check your modelset configuration.")

#     if len(text_models) == 0:
#         raise ValueError(
#             "No text models were provided. Your modelset has an empty text list, so no kNN pairs can be computed. "
#             "Update MODELSETS in pna_models.py."
#         )

#     if os.path.exists(file_path.replace(".txt", "_calibrated.txt")) and os.path.getsize(file_path.replace(".txt", "_calibrated.txt")) > 0:
#         c_results = load_results(file_path.replace(".txt", "_calibrated.txt"))
#         print(f"Calibrated results already exist in {file_path.replace('.txt', '_calibrated.txt')}. Loaded existing results.")
#         _calibrate = False  # Disable calibration if results already exist

#     if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
#         results = load_results(file_path)
#         file_name = f"{type}{'_' + cka_type if cka_type else ''}{'_sigma' + str(rbf_sigma) if type == 'cka' and cka_type == 'rbf' else ''}{'_biased' if biased else ''}"
#         plot_results(results, c_results, type, modalities, file_name)
#         print(f"Results already exist in {file_path}. Loaded and plotted existing results.")

#         # return None
#     else:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "w") as f:
#             f.write("Image_Model,Text_Model,Max_Score\n")

#     for image_model in tqdm(image_models, desc=f"{type}: Processing {modalities[0]} model:", leave=False):
#         image_feats = load_all_chunks(image_model, modalities[0], num_chunks=num_chunks, clip=clip, exact=exact, q=q)

#         if image_feats is None:
#             print(f"Warning: No embeddings found for {image_model} Skipping.")
#             continue

#         for text_model in tqdm(text_models, desc=f"{type}: Processing {modalities[1]} model:", leave=False):
#             if (image_model, text_model) in results:
#                 print(f"Skipping {image_model} and {text_model} as results already exist.")
#                 continue

#             if _calibrate:
#                 scores = calculate_score(image_feats, text_model,modalities=modalities, topk=topk, type=type, subtype=cka_type, rbf_sigma=rbf_sigma, biased=biased, num_chunks=num_chunks, clip=clip, exact=exact, q=q,_calibrate=_calibrate)
#                 c_results[(image_model, text_model)] = scores.item()

#                 with open(file_path.replace(".txt", "_calibrated.txt"), "a") as f:
#                         f.write(f"{image_model},{text_model},{scores.item():.4f}\n")

#             scores = calculate_score(image_feats, text_model,modalities=modalities, topk=topk, type=type, subtype=cka_type, rbf_sigma=rbf_sigma, biased=biased, num_chunks=num_chunks, clip=clip, exact=exact, q=q)
#             results[(image_model, text_model)] = scores.max().item()

#             with open(file_path, "a") as f:
#                 f.write(f"{image_model},{text_model},{scores.max().item():.4f}\n")
            
#     if len(results) == 0:
#         print(
#             "Warning: No results were computed. "
#             "Check that embeddings exist for the selected models/chunks and that both model lists are non-empty."
#         )

#     file_name = f"{modalities[0]}_{modalities[1]}_{type}{'_' + cka_type if cka_type else ''}{'_sigma' + str(rbf_sigma) if type == 'cka' and cka_type == 'rbf' else ''}{'_biased' if biased else ''} {f'_clipped' if clip else ''}"
#     plot_results(results,c_results, type, modalities, file_name)
    
#     return None

# def load_results(file_path):
#     results = {}
#     if not os.path.exists(file_path):
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         print(f"Results file {file_path} does not exist. Created an empty results dictionary.")
#     else:
#         with open(file_path, "r") as f:
#             next(f)  # skip header
#             for line in f:
#                 image_model, text_model, mean_score = line.strip().split(",")
#                 results[(image_model, text_model)] = [float(mean_score)]
#     return results

def experiment_driver(
    experiment_names,
    EXPERIMENTS,
    model_set="all",
    num_chunks=10,
    clip=False,
    exact=False,
    q=0.9,
    calibrate=False,
    calibration_K=200,
    plot=True,
    results_file="../plots/results/results.csv",
):

    for experiment_name in experiment_names:

        experiment = EXPERIMENTS[experiment_name]

        print(f"\nExperiment: {experiment_name}")

        for metric_name in experiment["metrics"]:

            print(f"  Metric: {metric_name}")

            run_experiment(
                experiment_name=experiment_name,
                EXPERIMENTS=EXPERIMENTS,
                metric_name=metric_name,
                model_set=model_set,
                num_chunks=num_chunks,
                clip=clip,
                exact=exact,
                q=q,
                calibrate=calibrate,
                calibration_K=calibration_K,
            )

            # Plot after the metric has finished.
            if plot:
                plot_experiment_results(
                    results_file=results_file,
                    experiment_name=experiment_name,
                    EXPERIMENTS=EXPERIMENTS,
                    metric_name=metric_name,
                    calibrated=calibrate,
                )


# def calculate_score(feats_image, text_model, modalities, topk, type="mknn",subtype="linear", rbf_sigma=1.0, num_chunks=10, biased=False, exact=False, q=0.9, clip=False, _calibrate=False):
#     "calculate the mutualknn accuarcy, but only for one caption per image, then do it over all 5 captions and get mean and std"
#     if type == "mknn":
#         image_knn = knn_layers(feats_image, topk)

#     if (modalities[0] == "speech") and (modalities[1] == "text"):
#         feats_text = load_all_chunks(text_model, modalities[1], num_chunks=num_chunks, caption_number=0, clip=clip, exact=exact, q=q)
#         if _calibrate:

#             def sim(X, Y):
#                 # replaced by compute raw scores
#                 if type == "cka":
#                     return compute_cka(X, Y, subtype, rbf_sigma, u=not biased)
#                 elif type == "mknn":
#                     image_knn = knn_layers(X, topk)
#                     text_knn = knn_layers(Y, topk)
#                     return mutual_knn_layers(image_knn, text_knn,num_image_layers=X.shape[1], num_text_layers=Y.shape[1], topk=topk)

#             print(f"Calibrating {text_model} with {feats_image.shape[1]} image layers and {feats_text.shape[1]} text layers.")
#             print(f"Image features shape: {feats_image.shape}, Text features shape: {feats_text.shape}")
#             X_layers = [
#                             feats_image[:, l, :]
#                             for l in range(feats_image.shape[1])
#                         ]
            
#             Y_layers = [
#                             feats_text[:, l, :]
#                             for l in range(feats_text.shape[1])
#                         ]
#             calibrated_score, _, _ = calibrate_layers(X_layers, Y_layers, sim)
#             return calibrated_score

#         else:
#             # replaced by compute raw scores
#             if type == "mknn":
#                 text_knn = knn_layers(feats_text, topk)
#                 scores = mutual_knn_layers(image_knn, text_knn,num_image_layers=feats_image.shape[1], num_text_layers=feats_text.shape[1], topk=topk)
#             elif type == "cka":
#                 scores = cka_layers(feats_image, feats_text, subtype, rbf_sigma, biased=biased)
#             return scores.unsqueeze(0)
        
#     else:
#         all_text_feats = []
#         for i in range(5):
#             feats_text = load_all_chunks(text_model, modalities[1], num_chunks=num_chunks, caption_number=i, clip=clip, exact=exact, q=q)
#             all_text_feats.append(feats_text)
#             if type == "mknn":
#                 text_knn = knn_layers(feats_text, topk)
#                 scores = mutual_knn_layers(image_knn, text_knn,num_image_layers=feats_image.shape[1], num_text_layers=feats_text.shape[1], topk=topk)
#             elif type == "cka":
#                 scores = cka_layers(feats_image, feats_text, subtype, rbf_sigma, biased=biased)

#             if i == 0:
#                 all_scores = scores.unsqueeze(0)
#             else:
#                 all_scores = torch.cat((all_scores, scores.unsqueeze(0)), dim=0)

#         if not _calibrate:
#             return all_scores.mean(dim=0)
#         else:

#             def sim(X, Y):
#                 if type == "cka":
#                     return compute_cka(X, Y, subtype, rbf_sigma, u=not biased)
#                 elif type == "mknn":
#                     image_knn = knn_layers(X.unsqueeze(1), topk)
#                     text_knn = knn_layers(Y.unsqueeze(1), topk)
#                     return mutual_knn_layers(image_knn, text_knn,num_image_layers=X.shape[1], num_text_layers=Y.shape[1], topk=topk)

#             n_image_layers = feats_image.shape[1]
#             n_text_layers = feats_text.shape[1]
#             n_captions = 5

#             def aggregate(S):
#                 # calibrate_layers creates:
#                 #
#                 # [L_image, 5 * L_text]

#                 S = S.reshape(
#                     n_image_layers,
#                     n_captions,
#                     n_text_layers
#                 )

#                 # Same operation as your normal experiment
#                 mean_scores = S.mean(dim=1)

#                 # Same final layer selection
#                 return mean_scores.max()

#             print(f"Calibrating {text_model} with {feats_image.shape[1]} image layers and {feats_text.shape[1]} text layers.")
#             print(f"Image features shape: {feats_image.shape}, Text features shape: {feats_text.shape}")
  
#             X_layers = [
#                 feats_image[:, l, :]
#                 for l in range(feats_image.shape[1])
#             ]

#             Y_layers = [
#                 feats_text[:, l, :]
#                 for l in range(feats_text.shape[1])
#             ]
#             calibrated_score, _, _ = calibrate_layers(X_layers, Y_layers, sim, aggregate=aggregate)
#             print("IT WORKED!")
#             return calibrated_score

def plot_experiment_results(
    results_file,
    experiment_name,
    EXPERIMENTS,
    metric_name,
    calibrated=False,
):
    experiment = EXPERIMENTS[experiment_name]
    modalities = experiment["modalities"]

    rows = load_results(results_file)

    raw_results = results_to_dict(
        rows,
        modalities=modalities,
        metric=metric_name,
        calibrated=False,
    )

    calibrated_results = {}

    if calibrated:
        calibrated_results = results_to_dict(
            rows,
            modalities=modalities,
            metric=metric_name,
            calibrated=True,
        )

    file_name = make_plot_filename(
        modalities=modalities,
        metric_name=metric_name,
        calibrated=calibrated,
        plot_type="scaling",
    )

    plot_results(
        raw_results,
        calibrated_results,
        metric_name,
        modalities,
        file_name,
    )

def make_plot_filename(
    modalities,
    metric_name,
    calibrated=False,
    plot_type="scaling",
):
    modality_name = "-".join(modalities)

    calibration_name = (
        "raw-vs-calibrated"
        if calibrated
        else "raw"
    )

    return (
        f"{modality_name}"
        f"__{metric_name}"
        f"__{calibration_name}"
        f"__{plot_type}"
    )

def results_to_dict(
    rows,
    modalities,
    metric,
    calibrated=False,
):
    modality_a, modality_b = modalities

    score_column = (
        "calibrated_score"
        if calibrated
        else "raw_score"
    )

    results = {}

    for row in rows:

        if (
            row["modality_a"] != modality_a
            or row["modality_b"] != modality_b
            or row["metric"] != metric
        ):
            continue

        value = row[score_column]

        if value in ("", None):
            continue

        results[
            (row["model_a"], row["model_b"])
        ] = float(value)

    return results

if __name__ == "__main__":
     EXPERIMENTS = {
    "image_text": {
        "modalities": ("image", "text"),
        "n_sets": 5,
        "metrics": [
            "mknn_k10",
            "cka_linear_unbiased",
        ],
    },

    "speech_text": {
        "modalities": ("speech", "text"),
        "n_sets": 1,
        "metrics": [
            "mknn_k10",
            "cka_linear_unbiased",
        ],
    },

    "image_speech": {
        "modalities": ("image", "speech"),
        "n_sets": 5,
        "metrics": [
            "mknn_k10",
            "cka_linear_unbiased",
        ],
    },
}
    # data lenth // samples in a chunk
     number_of_chunks = 34380 // 4000

     experiment_driver(
        experiment_names=[
            "image_text",
            # "speech_text",
            # "speech_image",
        ],
        EXPERIMENTS=EXPERIMENTS,
        model_set="test",
        num_chunks= number_of_chunks,
        clip=False,
        exact=False,
        q=0.9,
        calibrate=False,
        calibration_K=200,
        results_file="../plots/results/results.csv",
    )