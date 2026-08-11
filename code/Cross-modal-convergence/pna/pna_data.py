import kagglehub


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
# print("Path to dataset files:", path_audio)

def get_flickr8k_dataset_paths():
    return path, path_audio

def get_flikr8k_text_image():
    return path

def build_flikr8k_text_audio_image():
    # Get the paths to the Flickr8k dataset files
    path, path_audio = get_flickr8k_dataset_paths()

    # Build the dataset structure
    dataset = {
        "image": {
            "path": path,
            "files": [
                "Flickr8k_Dataset.zip",
                "Flickr8k_text.zip"
            ]
        },
        "audio": {
            "path": path_audio,
            "files": [
                "flickr_audio_captions.zip"
            ]
        }
    }

    return dataset

def get_text_only():
    import pandas as pd
    # Get the paths to the Flickr8k dataset files
    path = get_flikr8k_text_image()
    print("Path to dataset files:", path)

    # Build the dataset structure for text only
    dataset = {
        "text": {
            "path": path,
            "files": [
                "Flickr8k_text.zip"
            ]
        }
    }

    # Load the text data into a DataFrame
    text_path = path + "/captions.txt"
    text_df = pd.read_csv(text_path, sep=",")
    text_df.columns = ["image", "caption"]

    return dataset, text_df

