from consts import *
from datasets import load_dataset, DatasetDict
from feature_extraction import feature_extractor, tokenizer


# download data

common_voice = DatasetDict()

common_voice["train"] = load_dataset(DATASET_PATH, DATASET_NAME, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(DATASET_PATH, DATASET_NAME, split="test", use_auth_token=True)

# Drop unnecessary columns
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# Downsample to 16000 kHz
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=8)
common_voice.save_to_disk(MODEL_PATH)
