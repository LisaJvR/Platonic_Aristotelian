# Platonic Aristotelian

This repository contains the experimental code and reporting materials for a research project on cross-modal convergence between self-supervised models across text, image, and speech modalities.

The main focus is to compare learned representations from large pretrained foundation models and explore how their internal representations align across modalities, with emphasis on speech and vision-language structure.

## Project overview

The codebase is organised around the experimental pipeline in:

- `code/Cross-modal-convergence/pna/` — data loading, model extraction, metric computation, and plotting
- `code/data/` — generated dataset files and experiment metadata
- `code/results/` — result CSVs, plots, and notebooks
- `report/` — thesis/LaTeX documentation for the project

The workflow is built around the Flickr8K dataset and combines image, text, and audio modalities to extract embeddings from large self-supervised models, compare them with similarity metrics such as KNN agreement and CKA, and inspect convergence patterns.

## Repository structure

```text
.
├── README.md
├── code/
│   ├── requirements.txt
│   ├── data/
│   ├── scripts/
│   └── Cross-modal-convergence/
│       ├── pna/
│       ├── plots/
│       └── results/
└── report_files/
```

## Requirements

- Python 3.10+
- PyTorch and related ML dependencies
- Optional but recommended: CUDA-capable GPU for large model extraction
- Access to Hugging Face model checkpoints and Kaggle-hosted datasets

## Environment setup

From the project root:

```bash
cd /path/to/Platonic_Aristotelian
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

If you are using Hugging Face models, authenticate before running the extraction pipeline:

```bash
hf auth login
```

Some models may require additional access approvals (for example, Gemma-family checkpoints).

## Data setup

The pipeline downloads and prepares the Flickr8K multimodal dataset using Kaggle-backed metadata and audio assets. The scripts expect Kaggle access to be configured for the current environment.

The relevant preparation logic lives in:

- `code/Cross-modal-convergence/pna/pna_data.py`

This script creates the merged dataset file and filtered version used for experiments, including audio and caption metadata. By default it writes dataset artefacts under the project data folders and uses the external embeddings directory configured in the script.

## Running the extraction pipeline

The main extraction pipeline is:

- `code/Cross-modal-convergence/pna/pna_pipeline.py`

Run it from the `pna` directory:

```bash
cd code/Cross-modal-convergence/pna
python pna_pipeline.py
```

This script:

1. Builds the multimodal Flickr8K dataset
2. Loops over the model sets and modalities
3. Extracts hidden-state embeddings for text, image, and audio models
4. Saves features to a structured embeddings directory

The model families and test selections are defined in `pna_models.py`. In the main script, you can switch between `modelset = "test"` and other configured sets and choose the `modalities` list.

## Running the experiments and metrics

The experiment logic is in:

- `code/Cross-modal-convergence/pna/pna_experiment.py`

This file loads stored embeddings, computes similarity scores, and writes results to CSVs in the results folder. It is the stage where comparisons such as KNN metrics and CKA are evaluated.

The plotting and analysis utilities are in:

- `code/Cross-modal-convergence/pna/pna_plotting_code.py`
- `code/Cross-modal-convergence/pna/metrics_config.py`

## Analysis notebooks and reporting

For interactive exploration and quick visual analysis, use:

- `code/Cross-modal-convergence/results/metrics/quick_analysis.ipynb`

This notebook is intended for inspecting results, checking calibration patterns, and generating quick summaries before preparing larger plots or thesis figures.

## Important notes

- The project is computationally expensive. Large language and vision models can require substantial memory and GPU time.
- Model checkpoints are cached in the Hugging Face cache; if needed, clear the cache manually when switching models or rerunning experiments.
- The project contains hard-coded local paths for embeddings and offload folders in `pna_data.py` and related scripts; these may need adjustment on a new machine.
- On Google Colab, the cache can be cleared with:

```bash
rm -rf /root/.cache/huggingface/hub/*
```


## Typical workflow

```bash
cd code/Cross-modal-convergence/pna
python pna_pipeline.py
# inspect generated embeddings and results
jupyter notebook ../results/metrics/quick_analysis.ipynb
```

This is a research codebase rather than a polished production application, so setup and execution paths may vary depending on hardware, dataset access, and available model checkpoints.
