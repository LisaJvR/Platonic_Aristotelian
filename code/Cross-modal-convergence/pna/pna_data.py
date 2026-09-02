import kagglehub
import pandas as pd
import os
import PIL
from PIL import Image
import torch
import wave

path = kagglehub.dataset_download("adityajn105/flickr8k") # Flickr 8k Dataset
path_audio = kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")
token_text_path = kagglehub.dataset_download("sealeopard/flickr8k-token-txt")

df_path = "../../data/flickr8k_audio_text_image.csv"

EMB_DIR = "../embeddings"
OFF_LOAD_FOLDER_COLAB = "/content/offload" #XXX change for other system
OFF_LOAD_FOLDER_LOCAL = "../../bin/offload"

print("Path to image files:", path)
print("Path to dataset files:", path_audio)
print("Path to token_file files:", token_text_path)

def get_flickr8k_dataset_paths():
    return path, path_audio

def get_audio_length(df):
    for audio_file in df["audio"].unique():
            audio_path = os.path.join(get_flickr8k_dataset_paths()[1], "flickr_audio/flickr_audio/wavs", audio_file)
            if os.path.exists(audio_path):
                with wave.open(str(audio_path), "rb") as wav_file:
                    num_frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate() # all 16kHz
                    length_in_seconds = num_frames / sample_rate
                    df["audio_length"] = length_in_seconds
            else:
                print(f"Warning: {audio_file} not found in the dataset.")
    return df

def remove_outliers(df, lower_quantile=0.01, upper_quantile=0.99):
    """
    From the original dataset, remove the text outliers (length) and audio (length)
    Then return filtered (the combination of what is left)
    """
    df = df.copy()

    df["caption_length"] = df["caption"].apply(lambda x: len(x.split()))
    df = get_audio_length(df)

    lower_bound_text = df["caption_length"].quantile(lower_quantile)
    upper_bound_text = df["caption_length"].quantile(upper_quantile)
    lower_bound_audio = df["audio_length"].quantile(lower_quantile)
    upper_bound_audio = df["audio_length"].quantile(upper_quantile)

    lower_bound_speaker = df.groupby("speaker").size().quantile(0.1)
    upper_bound_speaker = df.groupby("speaker").size().quantile(0.9)
    
    filtered_df = df[(df["caption_length"] >= lower_bound_text) & (df["caption_length"] <= upper_bound_text) & 
                     (df["audio_length"] >= lower_bound_audio) & (df["audio_length"] <= upper_bound_audio) &
                     (df.groupby("speaker").size() >= lower_bound_speaker) & (df.groupby("speaker").size() <= upper_bound_speaker)]
    
    return filtered_df

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
    speaker_caption_df = pd.read_csv(
        path_audio + "/wav2spk.txt",sep=r"\s+",header=None,names=["audio", "speaker"])

    audio_caption_df = audio_caption_df.merge(
        speaker_caption_df, on="audio", how="left", validate="one_to_one"
    )

    # merge into one refence db
    all_df = pd.merge(
        audio_caption_df,token_text_df,
        on=["image", "caption_number"],how="left",
        validate="one_to_one"
    )

    all_df = all_df[["image", "caption_number", "speaker", "audio", "caption"]]

    # sort by image and caption_number
    all_df = all_df.sort_values(by=["image", "caption_number"]).reset_index(drop=True)

    # view data
    print("Shape of merged dataframe:", all_df.shape)
    print("Columns in merged dataframe:", all_df.columns.tolist()) 
    print(all_df.head())

    # check for directory else create it
    os.makedirs(os.path.dirname(df_path), exist_ok=True)
    all_df.to_csv(df_path, index=False)

    all_df = remove_outliers(all_df, lower_quantile=0.01, upper_quantile=0.99)
    print("Shape of cleaned dataframe:", all_df.shape)
    print(all_df.head())

    save_dataset_index(all_df, "text")
    save_dataset_index(all_df, "image")
    save_dataset_index(all_df, "speech")

    return all_df

def get_audio_files(audio_ids):
    _, path = get_flickr8k_dataset_paths()
    audio_dir = os.path.join(path, "flickr_audio/flickr_audio/wavs")

    audio_files = []
    for audio_id in audio_ids:
        audio_path = os.path.join(audio_dir, audio_id)
        if os.path.exists(audio_path):
            audio_files.append(audio_path)
        else:
            print(f"Warning: {audio_id} not found in {audio_dir}.")
    return audio_files

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
    
    safe_model_name = model_name.replace("/", "__")
    emb_dir = f"{EMB_DIR}/{modality}/{safe_model_name}/features_{chunk_num}.pt"
    if not os.path.exists(emb_dir):
        print(f"Embeddings for {model_name} ({modality}) chunk {chunk_num} not found at {emb_dir}.")
        return None
    data = torch.load(emb_dir)

    return data

def load_all_chunks(model_name, modality, num_chunks, caption_number=0):
    '''
    Load all chunks of embeddings for a given model and modality, and return a concatenated tensor of the embeddings.
    If modality is "text" or "speech", it will select every 5th embedding starting from the specified caption_number.
    '''
    chunks = []

    for chunk_num in range(num_chunks):
        data = load_embeddings(
            model_name,
            modality,
            chunk_num=chunk_num
        )

        if data is None:
            raise ValueError(
                f"Missing {modality} chunk {chunk_num} "
                f"for {model_name}"
            )

        if modality == "text" or modality == "speech":
            feats = data["avg"]

            feats = feats.to(torch.float32)  # Ensure the tensor is of type float32

            if feats.shape[0] % 5 != 0:
                raise ValueError(
                    f"Unexpected shape for {modality} chunk {chunk_num} "
                    f"for {model_name}: {feats.shape}"
                )
            caption_feats = feats[caption_number::5]
            chunks.append(caption_feats)
            del caption_feats, data
        else:
            chunks.append(data["avg"])
            del data

    return torch.cat(chunks, dim=0)

def print_meta_info(model_name, modality, chunk_number):
    data = load_embeddings(model_name, modality, chunk_num=chunk_number)
    print("Model:", data["metadata"]["model_name"])
    print("Modality:", data["metadata"]["modality"])
    print("Avg shape:", data["avg"].shape)
    print("Avg dtype:", data["avg"].dtype)

def normalize_speakers(df, index):
    #XXX TODO
    return None

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


def get_captions_from_index(modality):
    
    index_df = pd.read_csv(f"{EMB_DIR}/{modality}/dataset_index.csv")
    dataset = pd.read_csv("../../data/flickr8k_audio_text_image.csv")

    index_df = index_df.reset_index(names="embedding_index")

    mapped_df = index_df.merge(
        dataset[["image", "caption_number", "caption","speaker"]],
        on=["image", "caption_number"],
        how="left",
        validate="one_to_one"
    )

    return mapped_df # returns: embedding_index, image, caption_number, caption

if __name__ == "__main__":
    all_df = build_flikr8k_text_audio_image()