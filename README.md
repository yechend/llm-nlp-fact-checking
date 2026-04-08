# 🧠 LLM-NLP Fact-Checking Pipeline

### Multi-Phase Retrieval & Classification for Climate Claims

---

## 📌 Overview

This project implements a **multi-stage fact-checking system** for climate-related claims using:

* Hybrid information retrieval (BM25 + dense embeddings)
* Transformer-based re-ranking
* Prompt-based classification with lightweight LLMs

The system processes **1.2M+ evidence sentences** and classifies claims into:

* `SUPPORTS`
* `REFUTES`
* `NOT_ENOUGH_INFO`
* `DISPUTED`

Developed as part of **COMP90042 (NLP, University of Melbourne)**, this project demonstrates a **modular, production-style NLP pipeline**.

📄 Full report:
👉 [Project Report](docs/NLP.pdf)

---

## 🏗️ System Architecture

### 🔥 Main Pipeline (Production)

Implemented in:

```
notebooks/main_info_retrieval_pipeline.ipynb
```

### Pipeline Flow

```
Claims → Hybrid Retrieval → Re-ranking → LLM Classification → Final Predictions
```

---

## ⚙️ Core Components

### 1. Preprocessing

* Stage-specific text processing
* Optimised for retrieval and classification

---

### 2. Hybrid Retrieval (Key Contribution)

* **BM25 (Pyserini)** → lexical matching
* **MiniLM bi-encoder (DPR)** → semantic retrieval
* Merge candidates into hybrid pool

✅ Improves recall significantly

---

### 3. Cross-Encoder Re-ranking

* Model: `MiniLM cross-encoder`
* Curriculum learning:

  * Phase 1 → random negatives
  * Phase 2 → hard negatives

✅ Produces top-5 high-quality evidence

---

### 4. LLM-Based Classification

* Models:

  * TinyLlama (best)
  * Qwen1.5
  * Phi-1.5

* Zero-shot prompt-based classification

Handles:

* Conflicting evidence → `DISPUTED`
* Missing evidence → `NOT_ENOUGH_INFO`

---

## ⚠️ Experimental Pipelines (Exploration Only)

Not used in final system due to lower performance.

### Perplexity Filtering

```
notebooks/experiment_perplexity.ipynb
```

### FAISS Retrieval

```
notebooks/experiment_faiss.ipynb
```

### SimCSE Pipeline

```
notebooks/simcse_and_classification.ipynb
```

🔎 Findings:

* SimCSE retrieves semantically relevant but low gold-match evidence
* FAISS filtering lacks robustness
* Hybrid pipeline outperforms alternatives

---

## 📊 Results

* **F-A Harmonic Mean:** ~0.283
* Significant improvement over baseline (~0.08)

### Key Insights

* Hybrid retrieval boosts recall
* Curriculum learning improves stability
* TinyLlama outperforms larger models
* Strong pipeline > large model alone

---

## 📂 Project Structure

```
.
├── notebooks/
│   ├── main_info_retrieval_pipeline.ipynb
│   ├── experiment_faiss.ipynb
│   ├── experiment_perplexity.ipynb
│   ├── simcse_and_classification.ipynb
│
├── data/
│   ├── train-claims.json
│   ├── dev-claims.json
│   └── test-claims-unlabelled.json
│
├── generate_train_dataset/
├── Sim_tools/
├── saved_model/
├── local_data/
├── log/
│
├── eval.py
├── README.md
├── LICENSE
```

---

## ⚙️ Setup & Running

### Install Dependencies

```bash
pip install pyserini
pip install transformers sentence-transformers faiss-cpu
pip install nltk
```

---

### Run Pipeline

Open:

```
notebooks/main_info_retrieval_pipeline.ipynb
```

Run in order:

* Preprocessing
* Retrieval
* Re-ranking
* Classification

---

### Evaluate

```bash
python eval.py \
  --predictions dev-claims-predictions.json \
  --groundtruth dev-claims.json
```

---

## 📈 Evaluation Metrics

* Evidence Retrieval F1
* Classification Accuracy
* F-A Harmonic Mean

---

## 👥 Contributors

* **Yechen Deng** — Retrieval, re-ranking, main pipepline
* **Zhenyuan He** — SimCSE, FAISS, LLM classification
* **Wen Zhou** — BERTopic, analysis, experiments

---

## 📄 License

MIT License