from tqdm import trange
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_img_models(model_name):
    return None

def extract_image(images, model_name, device, batch_size):
    return None


def load_llm(model_name):
    '''
    Loads the model from huggingface and returns the model. 
    It uses device_map="auto" to automatically use the available GPU/CPU resources.
    '''
    
    # dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    # print(f"torch_dtype:\t{dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        output_hidden_states=True,
    ).eval()
    return model


def load_text_model(model_name):
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True, # want all layers
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()

    return tokenizer, model

# max char lengths is 199, this max length is token- each model measures it differently.
def extract_text(texts, model_name, batch_size, max_length=512, return_att=False):
    '''
    This functions takes in the entire text corpus along with model name, batch_size, and the max_length (in tokens).
    return attention = True returns the attions mask for each token - metadata to see which tokens were used.
    It sends the texts to the model in batches but claculates the tokens of the entire text corpus.

    it returns the average pooling layer and the last layer of the hidden states for each token
    and the number of parameters in the model.
    '''
    tok, model = load_text_model(model_name)

    num_params = sum(p.numel() for p in model.parameters())
    # tokenize all the texts at once
    tokenized = tok(
        texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=max_length # not essential for all to have same lenth for pooling.
    )

    device = next(model.parameters()).device

    all_avg_feats = []
    all_last_feats = []

    for i in trange(0, len(texts), batch_size):
        batch = {k: v[i:i + batch_size].to(device) for k, v in tokenized.items()} # automatically handles different representations and additional fields
        with torch.no_grad():
            outputs = model(**batch) #XXX check if this breaks, could use batch['input_ids'] and batch['attention_mask'] instead of **batch

            # pooled average
            feats = torch.stack(outputs["hidden_states"]).permute(1, 0, 2, 3)  # (B, L, T, D) - Batch, Layer, Token, Dim
            mask = batch["attention_mask"].unsqueeze(-1).unsqueeze(1)      # (B, 1, T, 1) 
            feats_avg = (feats * mask).sum(2) / mask.sum(2)
            # last jayer token
            feats_last = feats[:, :, -1, :]  # (B, L, [T],  D) # last layer token (captures all the info from previous tokens)

            all_avg_feats.append(feats_avg.cpu())
            all_last_feats.append(feats_last.cpu())

    all_avg_feats = torch.cat(all_avg_feats, dim=0)
    all_last_feats = torch.cat(all_last_feats, dim=0)

    # try save space ---- 
    del model
    del tok
    gc.collect()
    # -------------------

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if return_att == True:
        return all_avg_feats, all_last_feats,num_params, tokenized["attention_mask"] # output formal (B,L,D)
    else:
        # output formal (B,L,D)
        return all_avg_feats, all_last_feats, num_params

def test_text_extraction(model_names, texts, batch_size=1, max_length=512):
    '''
    This function tests the text extraction for a list of model names and a list of texts.
    It prints the shape of the average and last layer features for each model.
    '''
    test_text = texts[:1]  # Use only the first text to avoid memory issues
    #print test text dataype and shape
    print(f"Test text datatype: {type(test_text)}, shape: {len(test_text)}")

    for model_name in model_names:
        print(f"\n Extracting features for model: {model_name}")
        avg_feats, last_feats, num_params = extract_text(test_text, model_name, batch_size=batch_size, max_length=max_length)
        print(f"Model: {model_name}, Avg Feats Shape: {avg_feats.shape}, Last Feats Shape: {last_feats.shape}, Num Params: {num_params}")

    print("--- Text extraction test completed for all models.--- ")


from pna_data import get_flickr8k_dataset_paths, get_text_only
from pna_models import get_models

if __name__ == "__main__":
    # Example usage
    dataset, texts_df = get_text_only()

    print(texts_df.head())

    text_models = get_models("main", "text")
    print(f"Text models: {text_models}")
    # test extract_text function for each model but only one line of text to avoid memory issues
    test_text_extraction(text_models, texts_df["caption"].tolist(), batch_size=1, max_length=512)




