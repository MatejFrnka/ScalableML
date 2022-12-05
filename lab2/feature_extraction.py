from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

from consts import *

feature_extractor = WhisperFeatureExtractor.from_pretrained(PRETRAINED_MODEL_PATH)
tokenizer = WhisperTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, language="Swedish", task="transcribe")
processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL_PATH, language="Swedish", task="transcribe")
