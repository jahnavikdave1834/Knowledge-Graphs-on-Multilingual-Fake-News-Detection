# Multilingual Fake News Detection using Knowledge Graphs

> **NLP-Powered Misinformation Detection System**
> Detects fake news across multiple languages using transformer embeddings fused with knowledge graph features.

## 🧠 Architecture

```
Input Text → Multilingual Tokenizer → Transformer Embeddings → KG Feature Extraction (NER) → Neural Classifier → Prediction
```

### Components

| Module | File | Purpose |
|--------|------|---------|
| `app` | `app.py` | Streamlit web UI for real-time fake news detection |
| `model` | `model.py` | Neural network with KG feature fusion layer |
| `graph` | `graph.py` | Knowledge Graph construction via Named Entity Recognition |
| `dataset` | `dataset.py` | FakeNewsNet data loading & preprocessing pipeline |
| `train` | `train.py` | Model training loop with evaluation metrics |
| `utils` | `utils.py` | Tokenization, embedding, and helper utilities |
| `debug_model` | `debug_model.py` | Model diagnostics and inspection tools |

---

## 🔍 Key Features

- 🌐 **Multilingual Detection** — supports cross-lingual fake news classification
- 🤗 **Transformer Embeddings** — contextual representations via DistilBERT / XLM-RoBERTa
- 🕸️ **Knowledge Graph Fusion** — NER-based graph features (nodes, edges, density) fused with text embeddings
- 📊 **FakeNewsNet Dataset** — BuzzFeed & PolitiFact real/fake news articles
- ⚠️ **Uncertainty Handling** — confidence-based `UNCERTAIN` prediction class
- 🖥️ **Interactive UI** — clean Streamlit interface for live predictions

---

## 📁 Dataset

| Source | Files |
|--------|-------|
| BuzzFeed | `BuzzFeed_fake_news_content.csv`, `BuzzFeed_real_news_content.csv` |
| PolitiFact | `PolitiFact_fake_news_content.csv`, `PolitiFact_real_news_content.csv` |
| Graph Relations | `*News.txt`, `*NewsUser.txt`, `*UserUser.txt`, `*UserFeature.mat` |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2️⃣ Train the Model

```bash
python train.py
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🔒 Design Principles

1. **Language Agnostic**: XLM-R embeddings generalize across languages without retraining.
2. **Graph-Enhanced**: Knowledge graphs capture entity relationships text alone misses.
3. **Confidence Thresholding**: Low-confidence predictions flagged as `UNCERTAIN` to avoid false certainty.
4. **Modular Pipeline**: Each component (graph, model, dataset) is independently testable.

---

## ⚙️ Model Details

- **Tokenizer**: Multilingual subword tokenizer
- **Backbone**: DistilBERT / XLM-RoBERTa (configurable)
- **KG Features**: Node count, edge count, graph density extracted via spaCy NER
- **Classifier**: Fully connected neural network with dropout regularization
- **Loss**: Binary cross-entropy with class balancing

---

## 👤 Author

**Jahnavi K Dave**
B.Tech Computer Science
[![GitHub](https://img.shields.io/badge/GitHub-jahnavikdave1834-181717?style=flat&logo=github)](https://github.com/jahnavikdave1834)
