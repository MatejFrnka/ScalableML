import gradio as gr
from transformers import pipeline
import os
import deepl
import openai
from PIL import Image
import requests
from io import BytesIO

TARGET_LANG = "EN-GB"

deepl_key = os.environ.get('DEEPL')
openai.api_key = os.environ.get('OPENAI')

whisper = pipeline(model="frnka/whisper-small-SE")  # change to "your-username/the-name-you-picked"
translator = deepl.Translator(deepl_key)


def transcribe(audio):
    text_sv = whisper(audio)["text"]
    print(f"Audio transcribed: {text_sv}")
    text_en = translator.translate_text(text_sv, target_lang=TARGET_LANG).text
    print(f"Text translated: {text_en}")
    res = openai.Image.create(
        prompt=text_en,
        n=1,
        size="512x512"
    )
    img_url = res['data'][0]['url']
    print(f"Image generated: {img_url}")
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    return text_sv, text_en, img


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=[gr.Textbox(label="Transcribed text"),
             gr.Textbox(label="English translation"),
             gr.Image(type="pil", label="Output image")],
    title="Swedish speech to image",
    description="You may have heard of text to image, or speach to text, but have you heard of speech to image? Now you have!",
)

iface.launch()
