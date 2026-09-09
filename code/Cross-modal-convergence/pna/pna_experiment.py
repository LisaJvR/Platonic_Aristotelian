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
    if not results or len(results) == 0:
        return False
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
    
    print(f"Calibrating with K={K}...")

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
    calibration_K=200,
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
            K=calibration_K,
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
    results_file,
    model_set="all",
    num_chunks=10,
    clip=False,
    exact=False,
    q=0.9,
    calibrate=False,
    calibration_K=200,
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
                calibration_K=calibration_K,
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

            existing_results.append(result)

def experiment_driver(
    experiment_names,
    EXPERIMENTS,
    results_file,
    model_set="all",
    num_chunks=10,
    clip=False,
    exact=False,
    q=0.9,
    calibrate=False,
    calibration_K=200,
    plot=True
):

    for experiment_name in experiment_names:

        experiment = EXPERIMENTS[experiment_name]

        print(f"\nExperiment: {experiment_name}")

        for metric_name in experiment["metrics"]:

            print(f"  Metric: {metric_name}")

            run_experiment(
                experiment_name=experiment_name,
                EXPERIMENTS=EXPERIMENTS,
                results_file=results_file,
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
    #  number_of_chunks = 34380 // 4000
    number_of_chunks =1

    results_files = "../results/metrics"
    os.mkdir(results_files) if not os.path.exists("../results/metrics") else None
    
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
        calibrate=True,
        calibration_K=10,# XXX 200
        plot=True,
        results_file=f"{results_files}/results.csv",
    )

    # plot_experiment_results(
    #     results_file=f"{results_files}/results.csv",
    #     experiment_name="image_text",
    #     EXPERIMENTS=EXPERIMENTS,
    #     metric_name="mknn_k10",
    #     calibrated=False,
    # )