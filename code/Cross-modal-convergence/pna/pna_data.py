import kagglehub
import pandas as pd
import os
import PIL
from PIL import Image


path = kagglehub.dataset_download("adityajn105/flickr8k") # Flickr 8k Dataset
path_audio = kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")
token_text_path = kagglehub.dataset_download("sealeopard/flickr8k-token-txt")

df_path = "../../data/flickr8k_audio_text_image.csv"

print("Path to image files:", path)
print("Path to dataset files:", path_audio)
print("Path to token_file files:", token_text_path)

def get_flickr8k_dataset_paths():
    return path, path_audio

def build_flikr8k_text_audio_image():
    # Get the paths to the Flickr8k dataset files
    path, path_audio = get_flickr8k_dataset_paths()

    # extract token linking file
    token_text_df = pd.read_csv(
        token_text_path + "/Flickr8k.token.txt", sep="\t",
        header=None,names=["image", "caption"])

    token_text_df["caption_number"] = (
        token_text_df["image"].str.extract(r"#(\d+)$")[0].astype(int)
        )

    token_text_df["image"] = (
        token_text_df["image"].str.replace(r"#\d+$", "", regex=True)
    )

    # get audio captions and clean data & caption numbers
    audio_caption_df = pd.read_csv(
        path_audio + "/wav2capt.txt",sep=r"\s+",
        header=None,names=["audio", "image", "caption_number"]
    )
    audio_caption_df["caption_number"] = (
        audio_caption_df["caption_number"].str.replace("#", "", regex=False)
        .astype(int)
    )

    # merge into one refence db
    all_df = pd.merge(
        audio_caption_df,token_text_df,
        on=["image", "caption_number"],how="left",
        validate="one_to_one"
    )

    all_df = all_df[["image", "caption_number", "audio", "caption",]]

    # sort by image and caption_number
    all_df = all_df.sort_values(by=["image", "caption_number"]).reset_index(drop=True)

    # view data
    print("Shape of merged dataframe:", all_df.shape)
    print("Columns in merged dataframe:", all_df.columns.tolist()) 
    print(all_df.head())

    # check for directory else create it
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    all_df.to_csv(df_path, index=False)

    return all_df

def get_image_files(image_ids):
    path, _ = get_flickr8k_dataset_paths()
    image_dir = os.path.join(path, "Images")

    images = []
    for image_id in image_ids:
        img_path = os.path.join(image_dir, image_id)
        try:
            with Image.open(img_path) as img:
                img.verify()
            images.append(img_path)

        except (IOError, SyntaxError, FileNotFoundError) as e:
            print(
                f"Warning: {image_id} is not a valid image file. "
                f"Error: {e}"
            )
    return images

def load_embeddings(model_name, modality, chunk_num=0):
    import torch

    safe_model_name = model_name.replace("/", "__")
    emb_dir = f"../embeddings/{modality}/{safe_model_name}/features_{chunk_num}.pt"
    data = torch.load(emb_dir)

    return data

def print_meta_info(model_name, modality):
    data = load_embeddings(model_name, modality)
    print("Model:", data["metadata"]["model_name"])
    print("Modality:", data["metadata"]["modality"])
    print("Avg shape:", data["avg"].shape)
    print("Avg dtype:", data["avg"].dtype)


def get_captions_from_index(modality):
    index_df = pd.read_csv(f"../embeddings/{modality}/dataset_index.csv")
    dataset = pd.read_csv("../../data/flickr8k_audio_text_image.csv")

    index_df = index_df.reset_index(names="embedding_index")

    mapped_df = index_df.merge(
        dataset[["image", "caption_number", "caption"]],
        on=["image", "caption_number"],
        how="left",
        validate="one_to_one"
    )

    return mapped_df # returns: embedding_index, image, caption_number, caption