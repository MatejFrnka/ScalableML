from transformers import pipeline
import gradio as gr

pipe = pipeline(model="openai/whisper-small")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    print("asdf")
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()