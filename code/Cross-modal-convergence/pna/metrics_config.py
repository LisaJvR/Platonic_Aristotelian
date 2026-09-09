from pna_metrics import (
    compute_cka,
    compute_mutual_knn,
    compute_rsa,
    compute_svcca,
    compute_pwcca,
    compute_cknna,
)


METRICS = {
    "mknn_k10": {
        "fn": compute_mutual_knn,
        "kwargs": {
            "topk": 10,
        },
        "higher_is_better": True,
    },

    "cka_linear_unbiased": {
        "fn": compute_cka,
        "kwargs": {
            "kernel": "linear",
            "unbiased": True,
        },
        "higher_is_better": True,
    },

    "cka_rbf_unbiased": {
        "fn": compute_cka,
        "kwargs": {
            "kernel": "rbf",
            "rbf_sigma": 1.0,
            "unbiased": True,
        },
        "higher_is_better": True,
    },

    "rsa": {
        "fn": compute_rsa,
        "kwargs": {},
        "higher_is_better": True,
    },

    "svcca": {
        "fn": compute_svcca,
        "kwargs": {},
        "higher_is_better": True,
    },

    "pwcca": {
        "fn": compute_pwcca,
        "kwargs": {},
        "higher_is_better": True,
    },

    "cknna_k10": {
        "fn": compute_cknna,
        "kwargs": {
            "k": 10,
        },
        "higher_is_better": True,
    },
}