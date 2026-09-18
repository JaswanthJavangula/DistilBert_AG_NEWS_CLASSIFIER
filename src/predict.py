"""Inference for AG News topic classification."""
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "distilbert"
MAX_LEN = 96

# ── load ONCE at import, not per request ──
_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.eval()                                  # dropout OFF, permanently

_labels = json.load(open(MODEL_DIR / "labels.json"))
_labels = {int(k): v for k, v in _labels.items()}   # JSON keys are strings


def predict_text(text: str) -> dict:
    """Return {'label': str, 'confidence': float, 'scores': {label: prob}}"""

    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    # TODO 1: tokenize — truncation, padding, max_length, return_tensors='pt'
    enc = _tok(text, truncation=True, padding=True , max_length= MAX_LEN, return_tensors='pt')

    # TODO 2: forward pass under torch.no_grad()
    with torch.no_grad():
        logits = _model(**enc).logits

    # TODO 3: logits -> probabilities
    # hint: torch.softmax(logits, dim=-1)[0]
    probs = torch.softmax(logits, dim=-1)[0]

    # TODO 4: pick the winner
    # hint: int(probs.argmax())
    idx = int(probs.argmax())

    return {
        "label": _labels[idx],
        "confidence": round(float(probs[idx]), 4),
        "scores": {_labels[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    for t in [
        "Manchester United signed a new striker for 60 million pounds",
        "Apple reported record quarterly earnings driven by iPhone sales",
        "Scientists discover a new method for carbon capture",
        "Peace talks resume between the two nations in Geneva",
    ]:
        print(t[:50], "→", predict_text(t))

