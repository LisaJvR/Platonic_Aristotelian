from tqdm import trange
import torch
import gc
from transformers import AutoModel, AutoTokenizer
import os

def save_to_dir(avg_features,final_features,meta_data):
    '''
    Save to directory with the following structure: ../embeddings/{modality}/{model_name}/features.pt
    '''

    safe_model_name = meta_data["model_name"].replace("/", "__")

    dir_path = os.path.join(
        "..",
        "embeddings",
        meta_data["modality"],
        safe_model_name
    )

    os.makedirs(dir_path, exist_ok=True)

    output = {
                "avg": avg_features.cpu(),
                "final": final_features.cpu(),
                "metadata": meta_data
            }
    
    torch.save(
                output,
                os.path.join(dir_path, "features.pt")
            )

    print(f"Saved features to: {dir_path}/features.pt")

def load_img_models(model_name):
    return None

def extract_image(images, model_name, device, batch_size):
    return None

def load_text_model(model_name, colab=True):
    ''' 
    Adapted from Koepke: https://github.com/akoepke/cave_umwelten/blob/main/extract_features.py#L163

    This function takes in the huggingface model name and returns the model along with its tokenizer.
    Tokenizers are set to pad on the left and use eos tokens when tokens are not available.
    It uses device, and returns hidden states for all layers.
    '''
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if colab == True:
        offload_folder = "/content/offload" # for colab
        os.makedirs(offload_folder, exist_ok=True)

    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True, # want all layers
        torch_dtype="auto",#XXX double check if this is the best option for memory usage, could use float16 or bfloat16
        device_map="auto",
    )

    model.eval()
    print(model.hf_device_map)

    return tokenizer, model

# max char lengths is 199, this max length is token- each model measures it differently.
def extract_text(text_data, model_name, batch_size, max_length=512, test=True):
    '''
    This functions takes in the entire text corpus along with model name, batch_size, and the max_length (in tokens).
    return attention = True returns the attions mask for each token - metadata to see which tokens were used.
    It sends the texts to the model in batches but claculates the tokens of the entire text corpus.

    it returns the average pooling layer and the last layer of the hidden states for each token
    and the number of parameters in the model.
    '''
        # texts data: image caption_number caption
    text = text_data["caption"].tolist()

    if test == True:
        text = text[:1]  # Use only the first text to avoid memory issues

    tok, model = load_text_model(model_name)
    model.eval()

    

    num_params = sum(p.numel() for p in model.parameters())

    # tokenize all the texts at once
    tokenized = tok(
        text,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=max_length # not essential for all to have same lenth for pooling.
    )

    device = next(model.parameters()).device

    all_avg_feats = []
    all_last_toks = []

    for i in trange(0, len(text), batch_size):

        # automatically handles different representations and additional fields
        batch = {k: v[i:i + batch_size].to(device) for k, v in tokenized.items()} 
        with torch.no_grad():
            outputs = model(**batch) #XXX check if this breaks, could use batch['input_ids'] and batch['attention_mask'] instead of **batch

            # pooled average
            feats = torch.stack(outputs["hidden_states"]).permute(1, 0, 2, 3)  # (B, L, T, D) - Batch, Layer, Token, Dim
            mask = batch["attention_mask"].unsqueeze(-1).unsqueeze(1)      # (B, 1, T, 1) 
            feats_avg = (feats * mask).sum(2) / mask.sum(2)
            # last jayer token
            toks_last = feats[:, :, -1, :]  # (B, L, [T],  D) # last layer token (captures all the info from previous tokens)

            all_avg_feats.append(feats_avg.cpu())
            all_last_toks.append(toks_last.cpu())

    all_avg_feats = torch.cat(all_avg_feats, dim=0) #avg embedding over all layers, 1 sample
    all_last_toks = torch.cat(all_last_toks, dim=0)

    #  save space ------- 
    del model
    del tok
    gc.collect()
    # -------------------

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta_data = {
        "num_params": num_params,
        "mask": tokenized["attention_mask"].cpu(),
        # Alignment metadata specific to Flickr8k
        "image_ids": text_data["image"].tolist(),
        "caption_numbers": text_data["caption_number"].tolist(),
        "model_name": model_name,
        "modality": "text",
    }
    # output formal (B,L,D)
    return all_avg_feats, all_last_toks, meta_data


def test_text_extraction(model_names, texts_df, batch_size=1, max_length=512):
    '''
    This function tests the text extraction for a list of model names and a list of texts.
    It prints the shape of the average and last layer features for each model.
    '''

    for model_name in model_names:
        print(f"\n Extracting features for model: {model_name}")
        avg_feats, last_feats, num_params, metadata = extract_text(text_data, model_name, batch_size=batch_size, max_length=max_length)
        print(f"Model: {model_name}, Avg Feats Shape: {avg_feats.shape}, Last Feats Shape: {last_feats.shape}, Num Params: {num_params}")
        
        save_to_dir(avg_feats,last_feats, metadata)

    print("--- Text extraction test completed for all models.--- ")


from pna_data import get_flickr8k_dataset, get_text_only, build_ids, build_flikr8k_text_audio_image
from pna_models import get_models

if __name__ == "__main__":
    df = get_flickr8k_dataset()

    print(torch.cuda.get_device_name(0))


    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {total_vram:.1f} GB")
    # Example usage
    dataset, texts_df = get_text_only()

    print(texts_df.head())

    text_models = get_models("test", "text")
    print(f"Text models: {text_models}")

    text_df = df[["image", "caption_number", "caption"]].copy()#XXX not best use of storage

    # test extract_text function for each model but only one line of text to avoid memory issues
    test_text_extraction(text_models, text_df, batch_size=1, max_length=512)




