import gradio as gr
from infer import predict_one

def analyze_audio(file):
    if file is None:
        return "Please upload or record an audio file."
    label, prob = predict_one(file)
    return f"🎙 Emotion: {label} (confidence: {prob:.2f})"

iface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="🎧 Record or Upload Audio"),
    outputs="text",
    title="Speech Emotion Recognizer (Day-23)",
    description="Detects emotion (happy / sad / angry / neutral) from voice using Wav2Vec2 + Linear Classifier."
)

if __name__ == "__main__":
    iface.launch()
