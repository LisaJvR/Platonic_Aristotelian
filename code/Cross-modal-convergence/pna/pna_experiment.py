import torch
from pna_models import get_models
from pna_data import load_all_chunks
from pna_metrics import  knn_layer, compare_layers, compute_cka, compute_cka_kernel
from pna_plotting_code import  plot_results, get_reg_coeffs, plot_results_ordered
import os
from tqdm import tqdm
from calibrated_similarity import calibrate, calibrate_layers
from metrics_config import METRICS
import csv
import os
import sys


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
    # check to see if the result already exists in the file and overwrite it if it does
    if file_exists:
        rows = load_results(file_path)
        for i, row in enumerate(rows):
            if (
                row["model_a"] == result["model_a"]
                and row["model_b"] == result["model_b"]
                and row["metric"] == result["metric"]
            ):
                rows[i] = {
                    key: result.get(key)
                    for key in RESULT_COLUMNS
                }
                with open(file_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=RESULT_COLUMNS,
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                return

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

    matrices = [
        compare_layers(
            feats_A,
            feats_B,
            metric_fn,
            metric_kwargs,
        )
        for feats_B in tqdm(feats_B_list, desc="Comparing layers", leave=False)
    ]

    matrices = torch.stack(matrices)
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



    def similarity(X, Y):
        return metric_fn(X,Y, **metric_kwargs,)

    def aggregate(S):
        S = S.reshape(n_layers_A,n_sets,n_layers_B,)
        # Same aggregation as raw experiment
        S = S.mean(dim=1)

        return S.max()
    
    print(f"Calibrating with K={K} for {metric_fn.__name__} with kwargs {metric_kwargs}")
    if metric_fn == compute_cka: # cka takes kernels as input
        X_layers = [
                compute_cka_kernel(
                    feats_A[:, i, :],
                    **metric_kwargs
                )
                for i in range(n_layers_A)
            ]

        Y_layers = [
                compute_cka_kernel(
                    feats_B[:, j, :],
                    **metric_kwargs
                )
                for feats_B in feats_B_list
                for j in range(n_layers_B)
            ]
    else:
        X_layers = [
        feats_A[:, i, :]
        for i in range(n_layers_A)
        ]

        Y_layers = [
            feats_B[:, j, :]
            for feats_B in feats_B_list
            for j in range(n_layers_B)
        ]

    # calibrate layers
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
    non_calibrated=False,
    calibrate=False,
    calibration_K=200,
    existing_results=None,
):

    metric_fn = metric_config["fn"]
    kwargs = metric_config["kwargs"]

    if existing_results is None:
        layer_scores = compute_raw_scores(
            feats_A,
            feats_B_list,
            metric_fn,
            kwargs,
        )

        result = {
            "raw_score": layer_scores.max().item(),
            "layer_scores": layer_scores,
        }
    else:
        result = {
            "raw_score": float(existing_results["raw_score"]),
            "layer_scores": None,
        }

    if calibrate:

        calibrated, p, threshold = compute_calibrated_score(
            feats_A,
            feats_B_list,
            metric_fn,
            kwargs,
            K=calibration_K,
        )

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

def interleave_images(feats_A, num_captions=5):
    '''
    feats_A: [N, L, D]
    num_captions: number of captions per image
    '''
    n_samples, n_layers, n_dim = feats_A.shape

    # if n_samples % num_captions != 0:
    #     raise ValueError(f"Number of samples {n_samples} is not divisible by number of captions {num_captions}")

    n_images = n_samples * num_captions

    interleaved_feats = torch.empty(
        (n_images, n_layers, n_dim),
        dtype=feats_A.dtype,
        device=feats_A.device,
    )

    for i in range(n_images):
        interleaved_feats[i] = feats_A[i*num_captions:(i+1)*num_captions].mean(dim=0)

    return interleaved_feats

def interleave_captions(feats_B_list):
    feats_B = torch.stack(feats_B_list, dim=1)
    N, C, L, D = feats_B.shape
    feats_B = feats_B.reshape(N * C, L, D)
    return [feats_B]


def run_caption_density_experiment(
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
    caption_density=5,
):
    experiment = EXPERIMENTS[experiment_name]
    metric_config = METRICS[metric_name]

    modality_a, modality_b = experiment["modalities"]
    n_sets = experiment["n_sets"]

    models_A = get_models(model_set, modality_a)
    models_B = get_models(model_set, modality_b)

    if not (experiment_name == "caption_density_it" or experiment_name == "caption_density_is"):
        print(f"Skipping {experiment_name} for caption density experiment (not supported)")
        return

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


        feats_A = interleave_images(feats_A, num_captions=5)

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
                # print(
                #     f"Skipping {model_A} vs {model_B} for {metric_name} "
                #     f"(already exists in results file)"
                # )
                continue

            # load all chunks (for this experiment)
            feats_B_list = load_aligned_features(
                model=model_B,
                modality=modality_b,
                n_sets=caption_density,
                num_chunks=num_chunks,
                clip=clip,
                exact=exact,
                q=q,
            )

            # interleave images for 5 captions
            feats_B_list = interleave_captions(feats_B_list)
            
            if any(feats is None for feats in feats_B_list):
                print(f"Missing embeddings: {model_B}")
                continue

            matching_result = next(
                (
                    row for row in existing_results
                    if row["modality_a"] == modality_a
                    and row["modality_b"] == modality_b
                    and row["model_a"] == model_A
                    and row["model_b"] == model_B
                    and row["metric"] == metric_name
                ),
                None,
            )

            # print(f"Shape of feats_A: {feats_A.shape}, feats_B_list: {feats_B_list.shape}")
            scores = evaluate_pair(
                feats_A,
                feats_B_list,
                metric_config,
                calibrate=calibrate,
                calibration_K=calibration_K,
                existing_results=matching_result
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

    # just do linear cka for now since rbf is often unstable.
    if metric_name in ("cka_linear_biased", "cka_linear_unbiased", "svcca") and (calibrate==True):
        calibrate = True
    else:
        calibrate = False
    

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
                print(
                    f"Skipping {model_A} vs {model_B} for {metric_name} (cal:{calibrate}) "
                )
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

            matching_result = next(
                (
                    row for row in existing_results
                    if row["modality_a"] == modality_a
                    and row["modality_b"] == modality_b
                    and row["model_a"] == model_A
                    and row["model_b"] == model_B
                    and row["metric"] == metric_name
                ),
                None,
            )
            scores = evaluate_pair(
                feats_A,
                feats_B_list,
                metric_config,
                calibrate=calibrate,
                calibration_K=calibration_K,
                existing_results=matching_result
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

        if experiment_name in ("caption_density_it", "caption_density_is"):
            new_results_file = results_file.replace(".csv", f"__caption_density.csv")
            print(f"\nExperiment: {experiment_name} (caption density) - results will be saved to {new_results_file}")

        print(f"\nExperiment: {experiment_name}")

        for metric_name in experiment["metrics"]:

            print(f"  Metric: {metric_name}")

            if experiment_name in ("caption_density_it", "caption_density_is"):
                run_caption_density_experiment(
                    experiment_name=experiment_name,
                    EXPERIMENTS=EXPERIMENTS,
                    results_file=new_results_file,
                    metric_name=metric_name,
                    model_set=model_set,
                    num_chunks=num_chunks,
                    clip=clip,
                    exact=exact,
                    q=q,
                    calibrate=calibrate,
                    calibration_K=calibration_K,
                )
            else:
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

    # plot_results(
    #     raw_results,
    #     calibrated_results,
    #     metric_name,
    #     modalities,
    #     file_name,
    # )
    plot_results_ordered(
        raw_results,
        calibrated_results,
        metric_name,
        modalities,
        file_name ,
        order_by="size",
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
            "n_sets": 1,
            "metrics": [
                "mknn_k10",
                "cka_linear_biased",
                "cka_rbf_biased",
                "cka_linear_unbiased",
                "cka_rbf_unbiased",
                "svcca",
                
            ],
        },
      
        "speech_text": {
            "modalities": ("speech", "text"),
            "n_sets": 1,
            "metrics": [
                "mknn_k10",
                "cka_linear_biased",
                "cka_rbf_biased",
                "cka_linear_unbiased",
                "cka_rbf_unbiased",
                "svcca",
            ],
        },

        "image_speech": {
            "modalities": ("image", "speech"),
            "n_sets": 1,
            "metrics": [
                "mknn_k10",
                "cka_linear_biased",
                "cka_rbf_biased",
                "cka_linear_unbiased",
                "cka_rbf_unbiased",
                "svcca",
            ],
        },

        "caption_density_it": {
            "modalities": ("image", "text"),
            "n_sets": 5,
            "metrics": [
                "mknn_k10",
                "cka_linear_biased",
                "cka_rbf_biased"
            ]
        },

        "caption_density_is": {
            "modalities": ("image", "speech"),
            "n_sets": 5,
            "metrics": [
                "mknn_k10",
                "cka_linear_biased",
                "cka_rbf_biased"
            ]
        },
    }

    number_of_chunks = 9
    plot_dir = "../results/plots"

    results_files = "../results/metrics"
    os.mkdir("../results/") if not os.path.exists("../results/") else None
    os.mkdir(results_files) if not os.path.exists("../results/metrics") else None

    # get experiment names form args
    if len(sys.argv) > 1:
        experiment_names = [sys.argv[1]]
        calibrate = sys.argv[2] == "calibrate" if len(sys.argv) > 2 else False
        print(f"Running experiment: {experiment_names} with calibrate={calibrate}")
    else:
        experiment_names = [
            # "caption_density_it",
            # "caption_density_is",
            "image_speech",
            "speech_text",
            "image_text",
        ]
        calibrate = True
    
    # first run all non-calibrated experiments, then run all calibrated experiments
    experiment_driver(
        experiment_names=experiment_names,
        EXPERIMENTS=EXPERIMENTS,
        model_set="test",
        num_chunks= number_of_chunks,
        clip=True,
        exact=False,
        q=0.9,
        calibrate=False,
        calibration_K=200,# XXX 200
        plot=False,
        results_file=f"{results_files}/results.csv",
    )
    if calibrate:
        print(f"\n\nRunning calibration for {experiment_names} with K={200}")
        experiment_driver(
            experiment_names=experiment_names,
            EXPERIMENTS=EXPERIMENTS,
            model_set="test",
            num_chunks= number_of_chunks,
            clip=True,
            exact=False,
            q=0.9,
            calibrate=True,
            calibration_K=200,# XXX 200
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