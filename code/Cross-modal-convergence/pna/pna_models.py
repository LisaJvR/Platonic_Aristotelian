MODELSETS = {
    "all": {
        # 0-1B : 4 ; 2-3B: 4 ; 7-8B: 8 ; 10+B: 1 30+B: 3
        "text": [
            # Platonic model family (Fig 3 - compare image to 3 family of text models)
            "bigscience/bloom-560m",# P
            "bigscience/bloom-1b1",# P
            "bigscience/bloom-1b7",# P
            "bigscience/bloom-3b",# P
            "bigscience/bloom-7b1",# P
            "openlm-research/open_llama_3b",# P
            "openlm-research/open_llama_7b",# P
            "openlm-research/open_llama_13b",# P
            "huggyllama/llama-7b",# P 
            "huggyllama/llama-13b",# P
            "huggyllama/llama-30b",# P
            "huggyllama/llama-65b", # P 

            # Platonic 
            "google/gemma-2b",# P
            "google/gemma-7b",# P
            "mistralai/Mistral-7B-v0.1",# P
            "mistralai/Mixtral-8x7B-v0.1",# P
            "allenai/OLMo-1B-hf",# P
            "allenai/OLMo-7B-hf",# P

            "NousResearch/Meta-Llama-3-8B",##
            "NousResearch/Meta-Llama-3-70B", #largest #

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
            "bigscience/bloom-560m",#done
            "bigscience/bloom-1b1",#done
            "bigscience/bloom-1b7",#done
            "bigscience/bloom-3b", #done
            "bigscience/bloom-7b1",

            "openlm-research/open_llama_3b", #done
            "openlm-research/open_llama_7b",
            # "openlm-research/open_llama_13b",# skipped
            
            # "huggyllama/llama-7b",
            # "huggyllama/llama-13b",# skipped
            
            # Platonic 
            # "google/gemma-2b",
            # "google/gemma-7b",# P
            # "mistralai/Mistral-7B-v0.1",# P
            # "mistralai/Mixtral-8x7B-v0.1",# skipped
            # "allenai/OLMo-1B-hf",# P
            # "allenai/OLMo-7B-hf",# P
    
            # "NousResearch/Meta-Llama-3-8B",##         
            # "facebook/data2vec-text-base", # have data2vec audio later
        ],
        "image": [
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            # "vit_base_patch16_224.augreg_in21k",
            # "vit_large_patch16_224.augreg_in21k",
            # "vit_base_patch14_dinov2.lvd142m",
            # "vit_base_patch16_clip_224.laion2b",
            # "vit_base_patch16_clip_224.laion2b_ft_in12k",
            # "facebook/data2vec-vision-base",
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

import re

def pretty_model_name(model_name: str) -> str:
    name = model_name.lower()

    # ---------- Vision ----------
    if "dinov2" in name:
        size = re.search(r"vit_(tiny|small|base|large|giant)", name)
        size = size.group(1) if size else ""
        return f"DINOv2 ({size})"

    if ".mae" in name:
        size = re.search(r"vit_(tiny|small|base|large|huge)", name)
        size = size.group(1) if size else ""
        return f"MAE ({size})"

    if "clip" in name:
        size = re.search(r"vit_(tiny|small|base|large|huge)", name)
        size = size.group(1) if size else ""

        if "_ft_in12k" in name:
            return f"CLIP ({size}, IN12K-ft)"

        return f"CLIP ({size})"

    if "augreg" in name:
        size = re.search(r"vit_(tiny|small|base|large)", name)
        size = size.group(1) if size else ""
        return f"ViT ({size})"

    if "data2vec-vision" in name:
        size = model_name.split("-")[-1]
        return f"data2vec Vision ({size})"

    # ---------- Text ----------
    if "bloom" in name:
        size = model_name.split("-")[-1]
        return f"BLOOM ({size.upper()})"

    if "open_llama" in name:
        size = model_name.split("_")[-1]
        return f"OpenLLaMA ({size.upper()})"

    if "huggyllama/llama" in name:
        size = model_name.split("-")[-1]
        return f"LLaMA ({size.upper()})"

    if "meta-llama-3" in name:
        size = model_name.split("-")[-1]
        return f"LLaMA 3 ({size.upper()})"

    if "gemma" in name:
        size = model_name.split("-")[-1]
        return f"Gemma ({size.upper()})"

    if "mistral-" in name:
        size = re.search(r"(\d+b)", name)
        size = size.group(1).upper() if size else ""
        return f"Mistral ({size})"

    if "mixtral" in name:
        size = re.search(r"(\d+x\d+b)", name)
        size = size.group(1).upper() if size else ""
        return f"Mixtral ({size})"

    if "olmo" in name:
        size = re.search(r"(\d+b)", name)
        size = size.group(1).upper() if size else ""
        return f"OLMo ({size})"

    if "data2vec-text" in name:
        size = model_name.split("-")[-1]
        return f"data2vec Text ({size})"

    # ---------- Speech ----------
    if "wav2vec2-xls-r" in name:
        size = model_name.split("-")[-1]
        return f"XLS-R ({size.upper()})"

    if "wav2vec2" in name:
        if "large-robust" in name:
            return "wav2vec 2.0 (large, robust)"
        if "large-lv60" in name:
            return "wav2vec 2.0 (large, LV60)"
        if "large" in name:
            return "wav2vec 2.0 (large)"
        if "base" in name:
            return "wav2vec 2.0 (base)"

    if "hubert" in name:
        if "xlarge" in name:
            return "HuBERT (xlarge)"
        if "large" in name:
            return "HuBERT (large)"
        if "base" in name:
            return "HuBERT (base)"

    if "data2vec-audio" in name:
        size = model_name.split("-")[-1]
        return f"data2vec Audio ({size})"

    if "wavlm" in name:
        if "base-plus" in name:
            return "WavLM (base+)"
        size = model_name.split("-")[-1]
        return f"WavLM ({size})"

    if "unispeech-sat" in name:
        size = model_name.split("-")[-1]
        return f"UniSpeech-SAT ({size})"

    # fallback
    return model_name

def get_size(model_name):
        name = model_name.lower()

        if "tiny" in name:
            return "tiny"
        if "small" in name:
            return "small"
        if "base" in name:
            return "base"
        if "large" in name:
            return "large"
        if "huge" in name:
            return "huge"
        if "giant" in name:
            return "giant"

        return "base"

def pretty_text_name(model_name):
        """Only the model size displayed on the tick."""
        name = model_name.lower()

        if "bloom-" in name:
            return model_name.split("-")[-1]

        if "open_llama_" in name:
            return model_name.split("_")[-1]

        if "huggyllama/llama-" in name:
            return model_name.split("-")[-1]

        return model_name.split("/")[-1]

def text_family(model_name):
        name = model_name.lower()

        if "bloom" in name:
            return "BLOOM"
        if "open_llama" in name:
            return "OpenLLaMA"
        if "huggyllama" in name:
            return "LLaMA"

        return ""