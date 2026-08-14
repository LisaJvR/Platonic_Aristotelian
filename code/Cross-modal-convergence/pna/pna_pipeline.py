from unittest import case

from tqdm import trange
import torch
import gc
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, ViTImageProcessor, ViTModel
import os
from pna_data import build_flikr8k_text_audio_image, get_image_files
from pna_models import get_models
import timm
from timm.data import resolve_data_config, create_transform
from torchvision.models.feature_extraction import create_feature_extractor

EMB_DIR = "../embeddings"
OFF_LOAD_FOLDER_COLAB = "/content/offload" #XXX change for other system
OFF_LOAD_FOLDER_LOCAL = "../../bin/offload"

def check_bfloat16_support():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 7

def save_to_dir(avg_features,meta_data):
    '''
    Save to directory with the following structure: ../embeddings/{modality}/{model_name}/features.pt
    '''

    safe_model_name = meta_data["model_name"].replace("/", "__")

    dir_path = os.path.join(
        EMB_DIR,
        meta_data["modality"], safe_model_name
    )

    os.makedirs(dir_path, exist_ok=True)

    output = {
        "avg": avg_features.cpu(),
        "metadata": meta_data,
    }
    
    torch.save(output,os.path.join(dir_path, "features.pt"))

    print(f"Saved features to: {dir_path}/features.pt")

def save_dataset_index(df, modality):
    if modality == "text":
        index_df = (
        df[["image", "caption_number"]]
        .sort_values(["image", "caption_number"])
        .reset_index(drop=True)
    )

        index_df.to_csv(
            f"{EMB_DIR}/text/dataset_index.csv",
            index=False
        )
        print(f"Saved text dataset index to: {EMB_DIR}/text/dataset_index.csv")

    elif modality == "image":
        index_df = (
        df[["image"]]
        .sort_values(["image"])
        .reset_index(drop=True)
    )

        index_df.to_csv(
            f"{EMB_DIR}/image/dataset_index.csv",
            index=False
        )
        print(f"Saved image dataset index to: {EMB_DIR}/image/dataset_index.csv")

    elif modality == "speech":
        index_df = (
        df[["image", "caption_number"]]
        .sort_values(["image", "caption_number"])
        .reset_index(drop=True)
    )

        index_df.to_csv(
            f"{EMB_DIR}/speech/dataset_index.csv",
            index=False
        )
        print(f"Saved speech dataset index to: {EMB_DIR}/speech/dataset_index.csv")

def check_prior_extraction(safe_model_name, modality):
    # check if the directory with modality and model has files in it return boolean
    save_path = f"{EMB_DIR}/{modality}/{safe_model_name}/features.pt"

    if os.path.exists(save_path):
        print(f"Features already extracted for {safe_model_name} in {modality}. Skipping extraction.")
        return True
    return False

def clean_up(model, tok):
    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Image
# -----------------------

def load_img_models(model_name, device):
    proc = ViTImageProcessor.from_pretrained(model_name)
    vision_model = ViTModel.from_pretrained(model_name)
    return proc,vision_model

def extract_image(df, model_name, device, batch_size, cuda=True, test=False):
    images = df["image"].tolist()

    proc,vision_model = load_img_models(model_name, device)

    num_params = sum(p.numel() for p in vision_model.parameters())
    feats_all = []

    if test == True:
        images = images[:batch_size]  # Only process the first batch for testing

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]

        inputs = proc(images=batch_images, return_tensors='pt')

        inputs = { k: v.to(device)
        for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = vision_model(**inputs, output_hidden_states=True)
            cls_layers = [h[:, 0, :] for h in outputs.hidden_states[1:] ]
            features = torch.stack(cls_layers, dim=1)

        feats_all.append(features.cpu())

    meta_data = {
        "model_name": model_name,
        "modality": "image",
        "num_params": num_params,
        "num_samples": len(images),
        "num_layers": features.shape[1],
        "hidden_dim": features.shape[2],
        "dtype": str(features.dtype),
    }

    return torch.cat(feats_all, dim=0), meta_data

def run_extraction(model_names, df, device,modality, batch_size=1, test=False):
    save_dataset_index(df, modality=modality)

    for model_name in model_names:
        print(f"\n Extracting features for model: {model_name}")
        safe_model_name = model_name.replace("/", "__")
        if check_prior_extraction(safe_model_name, modality):
            print(f"Skipping {model_name} as features already extracted.")
            continue

        match modality:
           case "image":
                avg_feats, metadata = extract_image(df, model_name, device=device, batch_size=batch_size, cuda=torch.cuda.is_available(), test=test)
           case "text":
                avg_feats, metadata = extract_text(df, model_name,device=device, batch_size=batch_size, max_length=64, cuda=torch.cuda.is_available(), test=test)
           case "speech":
                print("Speech extraction not implemented yet.")
                continue
    
        print(f"Model: {model_name}, Avg Feats Shape: {avg_feats.shape}, Num Params: {metadata['num_params']}")
        
        save_to_dir(avg_feats,metadata)

    print(f"--- {modality} extraction test completed for all models.--- ")

# Text
# -----------------------

def load_text_model(model_name, cuda=True):
    ''' 
    Adapted from Koepke: https://github.com/akoepke/cave_umwelten/blob/main/extract_features.py#L163

    This function takes in the huggingface model name and returns the model along with its tokenizer.
    Tokenizers are set to pad on the left and use eos tokens when tokens are not available.
    It uses device, and returns hidden states for all layers.
    '''

    # need this to run locally, no issue on cuda
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if "huggyllama" in model_name:
        tokenizer.pad_token = "[PAD]"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if cuda == True:
        offload_folder = OFF_LOAD_FOLDER_COLAB
        os.makedirs(offload_folder, exist_ok=True)
    else:
        offload_folder = OFF_LOAD_FOLDER_LOCAL
        os.makedirs(offload_folder, exist_ok=True)

    dtype = torch.float16 #if check_bfloat16_support() else torch.float32 XXX cast back to 32 for analysis

    print(f"Loading model: {model_name} with dtype: {dtype} and offload folder: {offload_folder}")

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True, # want all layers
        dtype=dtype,#XXX double check if this is the best option for memory usage, could use float16 or bfloat16
        device_map="auto",
    )

    model.eval()
    # print(model.hf_device_map)

    return tokenizer, model

def extract_text(text_data, model_name, batch_size, max_length=512, test=True, cuda=True):
    '''
    This functions takes in the entire text corpus along with model name, batch_size, and the max_length (in tokens).
    return attention = True returns the attions mask for each token - metadata to see which tokens were used.
    It sends the texts to the model in batches but claculates the tokens of the entire text corpus.

    it returns the average pooling layer and the last layer of the hidden states for each token
    and the number of parameters in the model.
    '''
        # texts data: image caption_number caption
    text = text_data["caption"].tolist()
    tok, model = load_text_model(model_name, cuda=cuda) # returns eval model
    num_params = sum(p.numel() for p in model.parameters())

    # tokenize all the texts at once
    tokenized = tok(
        text, padding="longest",
        truncation=True, return_tensors="pt",
        max_length=max_length # not essential for all to have same lenth for pooling.
    )

    device = next(model.parameters()).device

    all_avg_feats = []
    # all_last_toks = []

    print(f"Extracting features for {len(text)} texts in batches of {batch_size}...")
    for i in trange(0, len(text), batch_size):

        # automatically handles different representations and additional fields
        batch = {k: v[i:i + batch_size].to(device) for k, v in tokenized.items()} 
        with torch.no_grad():
            outputs = model(**batch) #XXX check if this breaks, could use batch['input_ids'] and batch['attention_mask'] instead of **batch

            # pooled average
            feats = torch.stack(outputs["hidden_states"]).permute(1, 0, 2, 3)  # (B, L, T, D) - Batch, Layer, Token, Dim
            mask = batch["attention_mask"].unsqueeze(-1).unsqueeze(1)      # (B, 1, T, 1) 
            feats_avg = (feats * mask).sum(2) / mask.sum(2)

            # toks_last = feats[:, :, -1, :]  # (B, L, [T],  D) # last layer token (captures all the info from previous tokens)
            all_avg_feats.append(feats_avg.cpu())
            # all_last_toks.append(toks_last.cpu())

            if test == True:
                break  # Only process the first batch for testing

    print(f"Avg Feats shape: {torch.stack(all_avg_feats).shape}")
    all_avg_feats = torch.cat(all_avg_feats, dim=0) #avg embedding over all layers, 1 sample
    # all_last_toks = torch.cat(all_last_toks, dim=0)

    clean_up(model, tok)

    meta_data = {
        "model_name": model_name,
        "modality": "text",
        "num_params": num_params,
        "num_samples": all_avg_feats.shape[0],
        "num_layers": all_avg_feats.shape[1],
        "hidden_dim": all_avg_feats.shape[2],
        "dtype": str(all_avg_feats.dtype),
    }

    return all_avg_feats, meta_data, #  all_last_toks,


if __name__ == "__main__":

    df = build_flikr8k_text_audio_image()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0))
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_vram:.1f} GB")

    print(df.head())

    modelset = "test"
    modalities = ["text", "image", "speech"]

    for modality in modalities:
        print(f"Running {modelset} {modality}: --------------------------------")
        text_models = get_models(modelset, modality)
        print(f"Text models: {text_models}")

        df_copy = df[["image", "caption_number", "caption"]].copy()#XXX not best use of storage
        
        run_extraction(text_models, df_copy, device=device, modality=modality, batch_size=24, test=False)





