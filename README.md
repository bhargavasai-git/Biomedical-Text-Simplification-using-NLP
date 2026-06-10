# Biomedical-Text-Simplification-using-NLP

> Fine-tuning domain-specific large language models to simplify complex biomedical abstracts into plain, readable language — bridging the gap between scientific literature and general audiences.

## Overview

Biomedical literature is dense, jargon-heavy, and inaccessible to most non-expert readers. This project tackles the **automatic simplification of biomedical text** by fine-tuning Microsoft's **BioGPT** (a domain-specific causal language model) on the [PLABA dataset](https://osf.io/rnpmf/) — a benchmark corpus of biomedical abstract–plain-language pairs.

The pipeline covers end-to-end NLP: from preprocessing and NER-based linguistic analysis, through model fine-tuning, to multi-metric evaluation using ROUGE, BERT Score, SARI, and FKGL.

---

## Objectives

- Simplify complex biomedical abstracts into plain language using fine-tuned LLMs
- Analyse linguistic features (NER, dependency parsing, jargon density) that affect simplification quality
- Evaluate simplification quality using both lexical and semantic metrics
- Compare the effect of preprocessing strategies (lemmatization, stopword removal) on generation output

---

## Dataset

**PLABA Dataset** (Plain Language Adaptation of Biomedical Abstracts)

| Split | Description |
|-------|-------------|
| `train.csv` | Biomedical abstract–plain language pairs for fine-tuning |
| `test.csv` | Held-out pairs for evaluation |

Each row contains:
- `input_text` — original biomedical abstract sentence
- `target_text` — corresponding plain-language simplification

---

## Project Architecture

```
biomedical-text-simplification/
│
├── biogpt_final.py              # Full fine-tuning pipeline (BioGPT + HuggingFace Trainer)
├── biogptner_and_scores.py      # NER analysis, fast fine-tuning variant + evaluation metrics
├── data/
│   ├── train.csv
│   └── test.csv
├── outputs/
│   ├── PLABA_outputs_alternative.csv   # Generated simplifications
│   └── generated_texts.csv
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Lowercasing and removal of non-alphanumeric characters
- Tokenization using NLTK `word_tokenize`
- Stopword removal (`nltk.corpus.stopwords`)
- Lemmatization using `WordNetLemmatizer`
- Stemming using `PorterStemmer` (alternate preprocessing variant)

### 2. Linguistic Analysis
- **Named Entity Recognition (NER)** using spaCy (`en_core_web_sm`) to identify medical entities, drug names, and biomedical concepts
- **Dependency Parsing** to analyse syntactic structure — long noun phrases, passive constructions, and domain-specific jargon were found to influence simplification quality

### 3. Model Fine-Tuning
- **Model:** [`microsoft/BioGPT`](https://huggingface.co/microsoft/BioGPT) — a GPT-2-style causal LM pre-trained on 15M+ PubMed abstracts
- **Tokenizer:** `BioGptTokenizer` with max sequence length of 512
- **Training Framework:** HuggingFace `Trainer` API
- **Key Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Epochs | 5 (full run) / 1 (fast variant) |
| Batch size | 4 (full) / 2 (fast) |
| Max length | 512 tokens |
| Beam search | 5 beams |
| Temperature | 0.7 |
| Max new tokens | 100–128 |

### 4. Text Generation
- Beam search decoding (`num_beams=5`) with sampling (`do_sample=True`)
- `no_repeat_ngram_size=2` to reduce repetition
- GPU-accelerated inference via PyTorch (`cuda` if available)

---

## Evaluation Metrics

The generated simplifications were evaluated against target plain-language texts using the following metrics:

| Metric | What it measures |
|--------|-----------------|
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | N-gram overlap between generated and reference text |
| **BERTScore (P / R / F1)** | Semantic similarity using contextual embeddings |
| **SARI** | Quality of word additions, deletions, and kept words relative to source & reference |
| **FKGL** (Flesch-Kincaid Grade Level) | Readability level of the generated text |

> SARI was implemented from scratch using token-level precision/recall for add, delete, and keep operations, as a fallback to the `sari` library.

---

## Tech Stack

| Category | Tools / Libraries |
|----------|------------------|
| Language | Python 3 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP Preprocessing | NLTK, spaCy |
| Model | microsoft/BioGPT |
| Evaluation | rouge-score, bert-score, textstat, SARI |
| Data Handling | Pandas, HuggingFace Datasets |
| Environment | Google Colab (GPU) |

---

## Getting Started

### Prerequisites
```bash
pip install transformers torch datasets pandas nltk spacy sacremoses
pip install rouge_score bert_score textstat sari
python -m spacy download en_core_web_sm
```

### Run Fine-Tuning
```bash
# Full training pipeline
python biogpt_final.py

# NER analysis + fast training variant
python biogptner_and_scores.py
```

> **Note:** Both scripts were originally developed in Google Colab. Google Drive mount paths (`/content/drive/...`) should be updated to your local paths before running locally.

---

##  Key Findings

- Domain-specific jargon, long noun phrases, and passive constructions are primary complexity drivers in biomedical text
- Controlled use of these linguistic features in target text can **enhance readability** across model types
- NER and dependency parsing help surface entities that need simplification, improving downstream generation quality
- BioGPT, when fine-tuned on PLABA, produces fluent simplifications with measurable gains in ROUGE and BERTScore over zero-shot baselines

---

##  Author

**Bhargava Sai Matcha**
B.Tech Information Technology — JNTUH (2026)
[LinkedIn](https://linkedin.com/in/bhargava-sai-matcha) | bhargavasai.intern@gmail.com

---

## 📄 License

This project is for academic and research purposes. The PLABA dataset is subject to its own terms at [OSF](https://osf.io/rnpmf/).

---

## Acknowledgements

- Microsoft Research for the [BioGPT model](https://github.com/microsoft/BioGPT)
- PLABA Dataset authors for the biomedical simplification benchmark
- HuggingFace for the Transformers and Datasets libraries
