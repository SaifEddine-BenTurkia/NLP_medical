import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def clinical_vectorizer(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(model.config.hidden_size)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()