# Platonic_Aristotelian
ML&amp;AI MSc Project focussing on Cross-modal convergence between self-supervised speech models.


## Running the pipeline on a system other than local / colab
- before running the pipeline, log into huggingface to get acces: `hf auth login` , some models require seperate login and agreenl=ments (gemma)

- the models are saved to a bin direcoty in `pna_pipeline.py` (local) and to content/offload when cuda is set to true,
this can be manually changed at the top of the file at OFF_LOAD_FOLDER_COLAB/LOCAL at the top of the file.

- (running on colab): delete the cache of HF models: `!rm -rf /root/.cache/huggingface/hub/*`
