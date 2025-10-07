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
