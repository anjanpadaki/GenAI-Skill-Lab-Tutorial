# GenAI Skill Lab 🚀

> **Comprehensive laboratory implementations for the Multimodal Generative AI Skill Lab (7th Semester, MSRIT).**  
> Covers Deep Learning, Pretrained Word Embeddings, Seq2Seq Transformers, Recurrent Neural Networks (LSTMs), and Domain-Specific NLP Pipelines.

---

## 📌 Repository Overview

This repository contains end-to-end Python implementations, model architectures, evaluation metrics, and execution outputs for the GenAI Skill Lab exercises:

| Lab Question | Script / Asset | Primary Focus & Architecture | Key Technologies |
| :--- | :--- | :--- | :--- |
| **Q1** | [`Q1_Multimodal_Generative_AI_Lab (2).odt`](./Q1_Multimodal_Generative_AI_Lab%20(2).odt) | Multimodal Generative AI Lab Document & Report | Documentation |
| **Q2** | [`Q2.py`](./Q2.py) | Semantic Neighbor Retrieval & Prompt Enrichment | `gensim`, GloVe 50D |
| **Q3** | [`Q3.py`](./Q3.py) | Student Academic Performance Prediction (Pass/Fail) using a Deep Neural Network | `TensorFlow`, `Keras`, `scikit-learn` |
| **Q4** | [`Q4.py`](./Q4.py)<br>↳ [`Q4-output`](./Q4-output) | Academic Research Report Summarization using Hugging Face Pretrained Seq2Seq Transformers | `transformers`, `torch`, `Flan-T5` |
| **Q5** | [`Q5.py`](./Q5.py)<br>↳ [`technology_word_embeddings_pca.png`](./technology_word_embeddings_pca.png) | Word Vector Arithmetic, 2D PCA Dimensionality Reduction, Domain Clustering & Prompt Expansion | `gensim`, `PCA`, `matplotlib` |
| **Q6** | [`Q6.py`](./Q6.py) | Custom Word2Vec Embedding Training on a Domain-Specific Medical Corpus | `gensim`, Word2Vec (Skip-gram/CBOW) |
| **Q8** | [`sentiment_analysis.py`](./sentiment_analysis.py) | E-Commerce Customer Review Sentiment Classification Pipeline | `transformers`, `DistilBERT` |
| **Q12** | [`Q12.py`](./Q12.py)<br>↳ [`Q12-output`](./Q12-output) | Sequence Modeling & Character-Level Text Generation with Hyperparameter Tuning on Shakespeare Corpus | `PyTorch`, `LSTM` |

---

## 🛠️ Environment Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/anjanpadaki/GenAI-Skill-Lab-Tutorial.git
cd GenAI-Skill-Lab-Tutorial
```

### 2. Install Required Dependencies
Ensure you have **Python 3.10+** (tested on Python 3.12). Install the required packages via `pip`:

```bash
pip install tensorflow torch transformers gensim scikit-learn matplotlib pandas numpy
```

---

## 🔬 Lab Questions & Detailed Descriptions

### 🎓 Question 2: Semantic Prompt Enrichment
- **File**: [`Q2.py`](./Q2.py)
- **Objective**: Use pretrained word vectors to identify top-5 semantic neighbors for a target word (`technology`) and automatically enrich downstream generative prompt contexts.
- **Run Command**:
  ```bash
  python Q2.py
  ```

---

### 🎓 Question 3: Academic Performance Prediction using TensorFlow DNN
- **File**: [`Q3.py`](./Q3.py)
- **Objective**: Develop a Deep Neural Network (DNN) in TensorFlow to predict whether a student will **PASS** or **FAIL** based on their `Study_Hours` and `Attendance_Percentage`.
- **Architecture**:
  - Input Layer (2 features)
  - `Dense(32, relu)` + `Dropout(0.2)`
  - `Dense(16, relu)` $\rightarrow$ `Dense(8, relu)`
  - `Dense(1, sigmoid)` for binary classification
- **Results**: Achieved **89.00% test accuracy** with clear classification reports and inference confidence metrics across multiple student profiles.
- **Run Command**:
  ```bash
  python Q3.py
  ```

---

### 🎓 Question 4: Academic Report Summarization Tool
- **File**: [`Q4.py`](./Q4.py) | **Output File**: [`Q4-output`](./Q4-output)
- **Objective**: Summarize long technical academic reports into concise executive summaries for faster faculty review.
- **Architecture**:
  - Uses the pretrained instruction-tuned Hugging Face model `google/flan-t5-small`.
  - Employs beam search generation (`num_beams=4`, `length_penalty=1.5`, `no_repeat_ngram_size=2`).
- **Results**: Reduced 146 words down to 31 words (**78.77% compression**) while preserving core research methodologies and quantitative findings.
- **Run Command**:
  ```bash
  python Q4.py
  ```

---

### 🎓 Question 5: Pretrained Word Vectors, Vector Arithmetic & PCA Visualization
- **File**: [`Q5.py`](./Q5.py) | **Visualization**: [`technology_word_embeddings_pca.png`](./technology_word_embeddings_pca.png)
- **Objective**:
  1. **Vector Arithmetic**: Validates semantic linearity and offsets:
     - $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} = \vec{v}_{\text{queen}}$ ($0.8524$)
     - $\vec{v}_{\text{paris}} - \vec{v}_{\text{france}} + \vec{v}_{\text{germany}} = \vec{v}_{\text{berlin}}$ ($0.9204$)
  2. **Dimensionality Reduction**: Projects 10 technology domain word vectors from 50D $\rightarrow$ 2D using Principal Component Analysis (PCA) ($54.92\%$ variance explained).
  3. **Cluster Analysis**: Analyzes hardware, web infrastructure, and software logic clusters.
  4. **Contextual Expansion**: Generates semantically enriched prompts using vector neighbors.
- **Run Command**:
  ```bash
  python Q5.py
  ```

---

### 🎓 Question 6: Custom Word2Vec on Domain-Specific Medical Corpus
- **File**: [`Q6.py`](./Q6.py)
- **Objective**: Train a custom Word2Vec model on domain-specific healthcare texts.
- **Highlights**:
  - Demonstrates clinical co-occurrence (`doctor` $\leftrightarrow$ `physician`, `antibiotic` $\leftrightarrow$ `medication`).
  - Implements semantic category outlier detection (e.g., identifying `hospital` as an institutional outlier among pharmacological terms).
- **Run Command**:
  ```bash
  python Q6.py
  ```

---

### 🎓 Question 8: Real-Time Customer Sentiment Analysis
- **File**: [`sentiment_analysis.py`](./sentiment_analysis.py)
- **Objective**: Classify customer feedback reviews into **POSITIVE** or **NEGATIVE** sentiment to drive automated customer support escalation.
- **Model**: Hugging Face pipeline using `distilbert-base-uncased-finetuned-sst-2-english`.
- **Run Command**:
  ```bash
  python sentiment_analysis.py
  ```

---

### 🎓 Question 12: LSTM Text Generation Model (Shakespeare Dataset)
- **File**: [`Q12.py`](./Q12.py) | **Output File**: [`Q12-output`](./Q12-output)
- **Objective**: Build and train an LSTM sequence model in PyTorch to predict next characters and generate continuous text from seed prompts.
- **Hyperparameter Tuning Comparison**:
  - **Baseline (Config 1)**: `seq_len=10`, `hidden_dim=64`, `epochs=15` $\rightarrow$ Loss: `0.8271`
  - **Enhanced (Config 2)**: `seq_len=20`, `hidden_dim=128`, `epochs=30` $\rightarrow$ Loss: `0.0134` (**98.4% improvement**)
- **Generation Examples**: Generates coherent Shakespearean continuation from seed phrases such as `"the slings and arrows"` and `"to die, to sleep"`.
- **Run Command**:
  ```bash
  python Q12.py
  ```

---

## 📊 Summary of Output Files & Artifacts

| File | Type | Description |
| :--- | :--- | :--- |
| [`Q4-output`](./Q4-output) | Text Log | Execution log & summarization metrics for Question 4 |
| [`Q12-output`](./Q12-output) | Text Log | Training progression, comparison table, and generated text for Question 12 |
| [`technology_word_embeddings_pca.png`](./technology_word_embeddings_pca.png) | Image (PNG) | 2D PCA scatter plot showing technology word clusters for Question 5 |

---

## 👥 Authors

| Name | USN |
| :--- | :--- |
| **Aditya Karan** | `1MS23CS012` |
| **Shivansh** | `1MS23CS016` |
| **Amay Ranjan** | `1MS23CS019` |
| **Anjan Padaki** | `1MS23CS024` |

*Department of Computer Science & Engineering*  
*Ramaiah Institute of Technology (MSRIT), Bangalore*

