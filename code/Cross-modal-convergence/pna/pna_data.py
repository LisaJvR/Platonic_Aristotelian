import kagglehub
import pandas as pd
import os


# Flickr8k Image Caption Dataset
# ------------------------------
# The Flickr8k dataset is a benchmark for image captioning and image-text
# retrieval. It contains 8,000 natural images, each paired with five
# independently written captions describing the main objects, actions,
# and events in the scene.
#
# Dataset characteristics:
# - Images: 8,000
# - Captions per image: 5
# - Total captions: 40,000
#
# The images were collected from Flickr and selected from six different
# Flickr groups. They were manually chosen to represent a diverse range
# of everyday scenes while generally avoiding famous people and landmarks.
#
# Common files:
# - Flickr8k.token.txt      : All image-caption pairs.
# - Flickr_8k.trainImages.txt : Training image split.
# - Flickr_8k.devImages.txt   : Validation (development) image split.
# - Flickr_8k.testImages.txt  : Test image split.
#
# The dataset is widely used for evaluating image captioning, cross-modal
# retrieval, and vision-language representation learning models.
path = kagglehub.dataset_download("adityajn105/flickr8k") # Flickr 8k Dataset


# Flickr8k Audio Caption Dataset
# ------------------------------
# This dataset contains 40,000 spoken audio captions (.wav), corresponding to
# the five captions for each image in the Flickr8k dataset across the original
# train, validation, and test splits.
#
# Audio specifications:
# - Format: Microsoft WAVE (.wav)
# - Sample rate: 16,000 Hz
# - Bit depth: 16-bit
#
# Metadata files:
# - wav2capt.txt : Maps each audio file to its corresponding Flickr8k image
#                  (.jpg) and caption index.
# - Flickr8k.token.txt : Contains the text of each caption, allowing the
#                        image and caption index from wav2capt.txt to be
#                        resolved to the original caption text.
# - wav2spk.txt : Maps each audio file to its speaker ID. There are
#                 183 unique speakers in total.
#
# Total audio files: 40,000
# Unique speakers: 183
path_audio = kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")

# print("Path to dataset files:", path)
print("Path to dataset files:", path_audio)


token_text_path = kagglehub.dataset_download("sealeopard/flickr8k-token-txt")

df_path = "../../data/flickr8k_audio_text_image.csv"

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
    # Get the paths to the Flickr8k dataset files
    path, _ = get_flickr8k_dataset_paths()

    # List all image files in the dataset directory
    all_image_files = os.listdir(path)

    # Filter the list to include only the specified image IDs
    filtered_image_files = [
        img_file for img_file in all_image_files if img_file in image_ids
    ]

    return filtered_image_files