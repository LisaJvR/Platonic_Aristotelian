# Platonic_Aristotelian
ML&amp;AI MSc Project focussing on Cross-modal convergence between self-supervised speech models.


## Running the pipeline on a system other than local / colab
- the models are saved to a bin direcoty in `pna_pipeline.py` (local) and to content/offload when cuda is set to true,
this can be manually changed at the top of the file at OFF_LOAD_FOLDER_<device> at the top of the file.