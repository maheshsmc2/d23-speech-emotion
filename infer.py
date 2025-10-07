import torch
from transformers import Wav2Vec2FeatureExtractor
from model import EmotionNet
import soundfile as sf

CLASSES = ["neutral", "happy", "sad", "angry"]

def load_ckpt(path=None, base="facebook/wav2vec2-base"):
    model = EmotionNet(base, num_classes=len(CLASSES))
    if path:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval().to("cpu")  # Force CPU for Spaces
    fe = Wav2Vec2FeatureExtractor.from_pretrained(base)
    return model, fe

@torch.no_grad()
def predict_one(wav_path, ckpt=None):
    model, fe = load_ckpt(ckpt)
    # Robust audio loading
    try:
        audio, sr = sf.read(wav_path)
    except Exception:
        import torchaudio
        audio, sr = torchaudio.load(wav_path)
        audio = audio.mean(0).numpy()
    # Ensure mono
    try:
        import numpy as np
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(-1)
    except Exception:
        pass
    # Resample to 16k if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    # Featurize & predict
    inputs = fe(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    logits = model(**inputs)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    idx = int(torch.argmax(probs))
    return CLASSES[idx], float(probs[idx])
