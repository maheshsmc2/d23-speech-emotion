from transformers import Wav2Vec2Model
import torch
import torch.nn as nn

class EmotionNet(nn.Module):
    def __init__(self, base="facebook/wav2vec2-base", num_classes=4, dropout=0.2):
        super().__init__()
        self.enc = Wav2Vec2Model.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h//2, num_classes)
        )

    def forward(self, input_values, attention_mask=None):
        out = self.enc(input_values=input_values, attention_mask=attention_mask)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else torch.ones_like(hs[:,:,0:1])
        x = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(x)
