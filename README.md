<div align="center">

# 🎙️ Speech Emotion Recognizer (Day-23)

[![Hugging Face Space](https://img.shields.io/badge/🤗 Open%20in%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/maheshsmc/d23-speech-emotion)
[![GitHub Repo](https://img.shields.io/badge/📂 View%20on%20GitHub-black?logo=github)](https://github.com/maheshsmc2/d23-speech-emotion)
[![Python Version](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44-orange?logo=gradio)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/Transformers-4.44.2-yellow?logo=huggingface)](https://huggingface.co/docs/transformers)

Detects emotion (happy / sad / angry / neutral) from speech using **Wav2Vec2 + Linear Classifier**.  
Part of *Week-3 & 4 GenAI Code Revision Series*.

</div>
# 🎙️ Speech Emotion Recognizer (Day-23)

Detects emotion (happy / sad / angry / neutral) from speech using **Wav2Vec2 + Linear Classifier**.  
Part of Week-3 & 4 GenAI Code Revision Series.

## Features
- Self-supervised **Wav2Vec2** encoder  
- Mean-pooling classifier head  
- Gradio UI for record/upload  
- Hugging Face Space ready  

## Workflow
```
.wav → FeatureExtractor → Wav2Vec2Model → Pool → Linear → Softmax → Emotion label
```

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face
- Create a **Gradio Space** and upload this repo/ZIP.  
- Ensure `app.py` is the entry file.  
- Spaces will auto-install from `requirements.txt`.  

## Base Model
`facebook/wav2vec2-base`

## License
MIT
