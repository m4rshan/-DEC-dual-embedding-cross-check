# 📘 Dual Embedding Cross-Check (DEC)
*A simple, low-cost method for LLM hallucination and drift detection using dual semantic spaces*

The goal was to reduce compute while simultaneously increasing accuracy. To make a 20B parameter model as reliable as a 70B one. And give GPUs and RAM back to gamers. 

---

## ✨ Overview

Large language models (LLMs) are powerful but prone to **hallucination**, **semantic drift**, and **overconfident errors**. Current mitigation strategies rely on:

- Larger model sizes  
- Reinforcement learning (RLHF / RLAIF)  
- Retrieval augmentation  
- Ensembles & self-consistency checks  

These approaches **increase compute cost**, reduce interpretability, or require retraining the model.

**Dual Embedding Cross-Check (DEC)** proposes a simple alternative:

> Compare a transformer’s dynamic output embeddings to a stable, independent word-vector space (Word2Vec, GloVe, FastText).  
> Divergence between the two semantic trajectories may indicate hallucination or drift.

DEC acts as an **external semantic stabilizer**, requiring:

- no model changes  
- no retraining  
- no GPU resources  
- minimal computation  
- full interpretability  

---

## 🧠 Concept Summary

Transformers produce **token embeddings** that shift depending on context.  
Word embeddings provide **stable semantic coordinates**.

DEC compares:

1. **Transformer Semantic Path** (dynamic)  
2. **Word-Vector Semantic Path** (stable)

If they diverge beyond a threshold → possible hallucination or semantic instability.

---

## 🧭 Analogy — *Compass + GPS*

Transformers behave like a **GPS**:  
high-resolution but sometimes unstable.

Word embeddings act like a **compass**:  
simple, fixed, and stable.

If the GPS says you're heading north but the compass says you're going east —  
**something is off-course**.

DEC detects that moment.

---

## 🔧 Architecture Diagram

### **High-Level DEC Flow**

```text
TRANSFORMER SPACE (dynamic) ----\
                                  >--- DEC ---> Drift Signal
STATIC WORD SPACE (stable) ------/
```

### **Conceptual Flow**

```text
Transformer Embeddings → LLM Semantic Path
          |                     |
          v                     v
Word Embeddings ← Static Semantic Path
          |_____________________|
                    |
                    v
              DEC Cross-Check
                    |
                    v
          Drift / Hallucination Score
```

---

## 🧩 How DEC Works

### 1. Extract token embeddings from LLM output  
Forms the **dynamic semantic trajectory**.

### 2. Map tokens to nearest word vectors  
Using cosine similarity.

### 3. Construct parallel semantic paths  
- Path A: transformer embedding path  
- Path B: static word-vector path  

### 4. Compute divergence metrics  
- Local step-wise drift  
- Global path deviation  
- Anomaly scoring  

### 5. Output: Drift / Hallucination signal  

---

## 🧪 Intended Use Cases

✔ **Hallucination Detection**  
✔ **Semantic Drift Detection**  
✔ **Safety & Alignment Tooling**  
✔ **Small-Model Stabilization**

---

## 📂 Repository Structure

```text
DEC/
 ├── README.md
 ├── LICENSE
 ├── diagrams/
 │     ├── dec_flowchart.svg
 │     └── dec_tikz.tex
 ├── examples/
 │     └── dec_pseudocode.py
 ├── src/
 │     └── (future implementation)
 └── .gitignore (optional)
```

---

## 🧪 Pseudocode Example

```python
def dec_compare(llm_embeddings, word_vectors, threshold=0.25):

    mapped = []
    for token_embed in llm_embeddings:
        nearest = find_closest_word_vector(token_embed, word_vectors)
        mapped.append(nearest)

    llm_path = compute_path(llm_embeddings)
    static_path = compute_path(mapped)

    drift = cosine_distance(llm_path, static_path)

    if drift > threshold:
        return "Potential Hallucination", drift

    return "Stable", drift
```

---

## ⚠️ Limitations

- Word embeddings are coarse  
- DEC does **not** eliminate hallucinations  
- DEC is **not** a truth oracle  
- Requires empirical validation  
- Should complement, not replace, other safety methods  

---

## 📄 Citation (Template)

```bibtex
@article{m4rshan2025dec,
  title={Dual Embedding Cross-Check (DEC): A Lightweight Method for Detecting LLM Drift and Hallucination},
  author={Raghvendran Kumar / Raghav Kumar},
  year={2025},
  location={India},
  affiliation={Independent Explorer}
  email={raghavk.azp@gmail.com / raghavk.azp@outlook.com}
  note={Preprint},
}
```

---

## ⭐ Status

**This repository contains the conceptual foundation for DEC.  
Implementation and evaluation coming soon.**

---

🧭 *DEC is not a perfect solution — but a simple, interpretable step toward safer, more stable AI systems.*
