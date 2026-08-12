MODELSETS = {
    "all": {
        "text": [
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-1b7",
            "bigscience/bloom-3b",
            "bigscience/bloom-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b", # large
            "NousResearch/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-70B", #largest
            "google/gemma-2b",
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
    
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-7B-hf",
            "facebook/data2vec-text-base", # have data2vec audio later
        ],

        "image": [
            "vit_tiny_patch16_224.augreg_in21k",#imagenet
            "vit_small_patch16_224.augreg_in21k",
            "vit_base_patch16_224.augreg_in21k",
            "vit_large_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_large_patch14_dinov2.lvd142m",
            "vit_giant_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",
            "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "vit_large_patch14_clip_224.laion2b_ft_in12k",
            "vit_huge_patch14_clip_224.laion2b_ft_in12k",

            "facebook/data2vec-vision-base", # also have text and audio
            "facebook/data2vec-vision-large",
        ],

        "speech": [
            "facebook/wav2vec2-base",#self-supervised
            "facebook/wav2vec2-large",
            "facebook/wav2vec2-large-robust", # Same family/size, different pretraining data
            "facebook/wav2vec2-large-lv60", # Same family/size, different pretraining data

            "facebook/wav2vec2-xls-r-300m", # cross lingual (larger pretrained dataset), not fine tuned, just checkpoints
            "facebook/wav2vec2-xls-r-1b",

            "facebook/hubert-base-ls960",
            "facebook/hubert-large-ll60k",
            "facebook/hubert-xlarge-ll60k",

            "facebook/data2vec-audio-base",
            "facebook/data2vec-audio-large",

            "microsoft/wavlm-base",
            "microsoft/wavlm-base-plus",
            "microsoft/wavlm-large",

            "microsoft/unispeech-sat-base",
            "microsoft/unispeech-sat-large",

            # "facebook/w2v-bert-2.0", # conformer

            # "openai/whisper-tiny", # only use encodings
            # "openai/whisper-base",
            # "openai/whisper-small",
            # "openai/whisper-medium",
            # "openai/whisper-large",
        ],
    },
    
    "test": {
        "text": [
            "bigscience/bloom-560m",
            "openlm-research/open_llama_3b",
            "huggyllama/llama-7b",
            "NousResearch/Meta-Llama-3-8B",
            "google/gemma-2b",
            "mistralai/Mistral-7B-v0.1",
            "allenai/OLMo-1B-hf",
            "facebook/data2vec-text-base",
        ],
        "image": [
            "vit_base_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
            "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "facebook/data2vec-vision-base",
        ],
        "speech": [
            "facebook/wav2vec2-base",#self-supervised
            "facebook/hubert-base-ls960",
            "facebook/data2vec-audio-base",
            "microsoft/wavlm-base",
            "microsoft/unispeech-sat-base",
        ],
    }
}

'''
Input the @modelset: "main" and @modality: "text", "image", or "speech" to get the list of models for that modality.
'''
def get_models(modelset, modality):
    return MODELSETS[modelset][modality]