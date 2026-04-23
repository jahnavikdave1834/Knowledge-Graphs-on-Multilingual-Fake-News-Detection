# src/model.py
# Deployment-safe lightweight model (NO torch, NO transformers)

import torch
import torch.nn as nn
from transformers import AutoModel

class FakeNewsModel(nn.Module):
    def __init__(self, model_name='distilbert-base-multilingual-cased', use_kg=True):
        super(FakeNewsModel, self).__init__()
        self.use_kg = use_kg
        self.bert = AutoModel.from_pretrained(model_name)
        
        # DistilBERT hidden size is typically 768
        bert_out_dim = self.bert.config.hidden_size
        kg_dim = 3 if use_kg else 0
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_out_dim + kg_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # Simple heuristic keywords (kept for backwards compatibility if needed)
        self.fake_keywords = [
            "secret", "miracle", "cure", "shocking",
            "hoax", "conspiracy", "instantly", "confirmed",
            "scientists claim", "breaking", "guaranteed"
        ]

    def forward(self, input_ids, attention_mask, kg_features=None):
        # Pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # distilbert returns last_hidden_state as the first element
        # we use the representation of the first token ([CLS])
        hidden_state = outputs.last_hidden_state[:, 0, :]
        
        if self.use_kg and kg_features is not None:
            combined = torch.cat((hidden_state, kg_features), dim=1)
        else:
            combined = hidden_state
            
        logits = self.classifier(combined)
        return logits

    def predict_proba(self, text):
        """
        Dummy predict_proba based on original code, in case some inference script still relies on it.
        """
        if not text or not text.strip():
            return 0.5

        text = text.lower()
        score = sum(word in text for word in self.fake_keywords)
        probability = score / len(self.fake_keywords)
        return max(0.05, min(probability, 0.95))
