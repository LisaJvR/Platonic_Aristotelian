from PIL import Image
from unittest import case
from tqdm import tqdm, trange
import torch
import gc
import torchvision
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, ViTImageProcessor, ViTModel
import os
from pna_data import build_flikr8k_text_audio_image, get_image_files
from pna_models import get_models
import timm
from timm.data import resolve_data_config, create_transform
from torchvision.models.feature_extraction import create_feature_extractor

EMB_DIR = "../embeddings"
OFF_LOAD_FOLDER_COLAB = "/content/offload" #XXX change for other system
OFF_LOAD_FOLDER_LOCAL = "../../bin/offload"

import os
import shutil

def delete_hf_cached_model(model_name):
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")

    cache_name = "models--" + model_name.replace("/", "--")
    model_cache_path = os.path.join(cache_root, cache_name)

    if os.path.exists(model_cache_path):
        print(f"Deleting cached model: {model_cache_path}")
        shutil.rmtree(model_cache_path)
        print("Deleted.")
    else:
        print(f"Cache not found for {model_name}")

def save_to_dir(avg_features,meta_data, batch_num=None):
    '''
    Save to directory with the following structure: ../embeddings/{modality}/{model_name}/features.pt
    '''

    safe_model_name = meta_data["model_name"].replace("/", "__")

    dir_path = os.path.join(
        EMB_DIR, meta_data["modality"], safe_model_name
    )

    os.makedirs(dir_path, exist_ok=True)

    output = {
        "avg": avg_features.cpu(),
        "metadata": meta_data,
    }

    if batch_num is not None:
        torch.save(output,os.path.join(dir_path, f"features_{batch_num}.pt"))
        print(f"Saved features to: {dir_path}/features_{batch_num}.pt")
    else:
        torch.save(output,os.path.join(dir_path, "features_all.pt"))
        print(f"Saved features to: {dir_path}/features_all.pt")

def save_dataset_index(df, modality):
    if f"{EMB_DIR}/{modality}" not in os.listdir(EMB_DIR):
        os.makedirs(f"{EMB_DIR}/{modality}", exist_ok=True)
    if f"{EMB_DIR}/{modality}/dataset_index.csv" in os.listdir(f"{EMB_DIR}/{modality}"):
        print(f"Dataset index for {modality} already exists. Skipping save.")
        return
    
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

def check_prior_extraction(safe_model_name, modality, chunk):
    # check if the directory with modality and model has files in it return boolean
    save_path = f"{EMB_DIR}/{modality}/{safe_model_name}/features_{chunk}.pt"

    if os.path.exists(save_path):
        print(f"Features already extracted for {safe_model_name} in {modality}. Skipping extraction.")
        return True
    return False

def check_device():
    if  torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")

    return device

# Image
# -----------------------

def load_img_models(model_name):
    device = check_device()
    # dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32
    dtype = torch.float32

    model = timm.create_model( model_name, pretrained=True)

    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model),
        is_training=False
    )

    if device.type == "cuda":
        offload_folder = OFF_LOAD_FOLDER_COLAB
        os.makedirs(offload_folder, exist_ok=True)
    else:
        offload_folder = OFF_LOAD_FOLDER_LOCAL
        os.makedirs(offload_folder, exist_ok=True)

    print(f"Loading model: {model_name} with dtype: {dtype} and offload folder: {offload_folder}")

    model = model.to(device, dtype=dtype).eval()
    return transform, model, dtype

def check_vision_hidden_states(model, outputs):
    all_hs = outputs.hidden_states
    block_hs = all_hs[1:]

    print("Architecture:", model.__class__.__name__)
    print("Config model_type:", model.config.model_type)
    print(
        "Configured layers:",
        getattr(model.config, "num_hidden_layers", None)
    )
    print("Total returned hidden states:", len(all_hs))
    print("Block hidden states:", len(block_hs))

    for i, h in enumerate(all_hs):
        print(f"Hidden state {i}: {tuple(h.shape)}")

        if h.ndim != 3:
            return False

    expected = getattr(model.config, "num_hidden_layers", None)

    if expected is not None:
        assert len(block_hs) == expected, (
            f"Expected {expected} transformer block outputs, "
            f"got {len(block_hs)}"
        )

    return True

'''
Adapted from Koepke: https://github.com/akoepke/cave_umwelten/blob/main/extract_features.py
'''
def extract_image(df, model_name, device, batch_size, cuda=True, test=False):
    # unique images
    image_paths = get_image_files(df["image"].unique().tolist())

    transform, vision_model, dtype = load_img_models(model_name)

    num_blocks = len(vision_model.blocks)
    num_params = sum(p.numel() for p in vision_model.parameters())

    block_ids = list(range(num_blocks))
    return_nodes = [f"blocks.{i}.add_1" for i in block_ids]

    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

    chunk_size = max(1, 800 // batch_size) 

    chunk_feats = []
    c_index = 0

    if test == True:
        image_paths = image_paths[:batch_size]

    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting {model_name}", unit="batch"):

        batch_image_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(img_path).convert("RGB") for img_path in batch_image_paths]

        inputs = torch.stack([transform(img) for img in batch_images]).to(device=device, dtype=dtype)

        with torch.no_grad():
            outputs = vision_model(inputs)

            cls_layers = [h[:, 0, :] for h in outputs.values()] 
            features = torch.stack(cls_layers).permute(1, 0, 2) # B, L, D

            chunk_feats.append(features.cpu())
            del outputs, features, inputs

            if len(chunk_feats) == chunk_size:
                chunk_tensor = torch.cat(chunk_feats, dim=0)
            
                save_to_dir(avg_features=chunk_tensor, meta_data={
                    "model_name": model_name,
                    "modality": "image",
                    "num_params": num_params,
                    "chunk_number": c_index,
                    "dtype": str(chunk_tensor.dtype),
                } , batch_num=c_index)

                c_index +=1
                del chunk_feats, chunk_tensor
                chunk_feats = []

            if test == True: break 

    if len(chunk_feats) > 0:
            chunk_tensor = torch.cat(chunk_feats, dim=0)
            save_to_dir(avg_features=chunk_tensor, meta_data={
                "model_name": model_name,
                "modality": "image",
                "num_params": num_params,
                "chunk_number": c_index,
                "dtype": str(chunk_tensor.dtype),
            },batch_num=c_index)

    del chunk_feats, num_params, vision_model
    delete_hf_cached_model(model_name)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None

def run_extraction(model_names, df,modality, batch_size=1, test=False):
    save_dataset_index(df, modality=modality)

    for model_name in model_names:
        print(f"\n Extracting features for model: {model_name}")
        safe_model_name = model_name.replace("/", "__")

        device = check_device()

        if test == True:
            chunk = 0
        else:
            chunk = 9

        if check_prior_extraction(safe_model_name, modality, chunk):
            print(f"Skipping {model_name} (chunk {chunk}) as features already extracted.")
            continue

        match modality:
           case "image":
                extract_image(df, model_name, device=device, batch_size=batch_size, cuda=torch.cuda.is_available(), test=test)
           case "text":
                extract_text(df, model_name, device=device, batch_size=batch_size, max_length=64, cuda=torch.cuda.is_available(), test=test)
           case "speech":
                print("Speech extraction not implemented yet.")
                continue
    
        print(f"Model: {model_name} done.")

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

    device = check_device()
    dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

    print(f"Loading model: {model_name} with dtype: {dtype} and offload folder: {offload_folder}")

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True, # want all layers
        dtype=dtype,#XXX double check if this is the best option for memory usage, could use float16 or bfloat16
        device_map="auto",
    )

    model.eval()

    return tokenizer, model

def extract_text(text_data, model_name,device, batch_size,test, max_length=512, cuda=True):
    '''
    This functions takes in the entire text corpus along with model name, batch_size, and the max_length (in tokens).
    return attention = True returns the attions mask for each token - metadata to see which tokens were used.
    It sends the texts to the model in batches but claculates the tokens of the entire text corpus.

    it returns the average pooling layer and the last layer of the hidden states for each token
    and the number of parameters in the model.
    '''
    text = text_data["caption"].tolist()
    tok, model = load_text_model(model_name, cuda=cuda) # returns eval model

    num_params = sum(p.numel() for p in model.parameters())

    # tokenize all the texts at once
    tokenized = tok(
        text, padding="longest",
        truncation=True, return_tensors="pt",
        max_length=max_length # not essential for all to have same lenth for pooling.
    )

    device = next(model.parameters()).device #XXX redundant, but ensures model and tokens are on same device

    chunk_size = max(1, 4000 // batch_size)

    chunk_feats = []
    c_index = 0

    for i in tqdm(range(0, len(text), batch_size), desc=f"Extracting {model_name}",
    unit="batch"):

        # automatically handles different representations and additional fields
        batch = {k: v[i:i + batch_size].to(device) for k, v in tokenized.items()} 
        with torch.no_grad():
            outputs = model(**batch) #XXX check if this breaks, could use batch['input_ids'] and batch['attention_mask'] instead of **batch

            feats = torch.stack(outputs["hidden_states"]).permute(1, 0, 2, 3)  # (B, L, T, D) 
            mask = batch["attention_mask"].unsqueeze(-1).unsqueeze(1)  # (B, 1, T, 1) 
            feats_avg = (feats * mask).sum(2) / mask.sum(2)

            if torch.isnan(feats_avg).any():
                print(f"NaNs in pooled features at batch {i // batch_size}")
                return None

            chunk_feats.append(feats_avg.cpu())
            del outputs, feats, feats_avg, batch, mask

            if len(chunk_feats) == chunk_size:
                chunk_tensor = torch.cat(chunk_feats, dim=0)

                save_to_dir(avg_features=chunk_tensor, meta_data={
                    "model_name": model_name,
                    "modality": "text",
                    "num_params": num_params,
                    "chunk_number": c_index,
                    "dtype": str(chunk_tensor.dtype),
                } , batch_num=c_index)

                c_index +=1
                del chunk_feats, chunk_tensor
                chunk_feats = []

            if test == True: break 

    if len(chunk_feats) > 0:
        chunk_tensor = torch.cat(chunk_feats, dim=0)
        save_to_dir(avg_features=chunk_tensor, meta_data={
            "model_name": model_name,
            "modality": "text",
            "num_params": num_params,
            "chunk_number": c_index,
            "dtype": str(chunk_tensor.dtype),
        },batch_num=c_index)

    del model, tok
    delete_hf_cached_model(model_name)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None

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
    modalities = ["image"]

    for modality in modalities:
        print(f"Running {modelset} {modality}: --------------------------------")
        models = get_models(modelset, modality)
        print(f"Models: {models}")

        df_copy = df[["image", "caption_number", "caption"]].copy()#XXX not best use of storage
        run_extraction(models, df_copy, modality=modality, batch_size=32, test=True)