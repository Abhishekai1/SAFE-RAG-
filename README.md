# SAFE-RAG++: A Causal, Uncertainty-Aware, Knowledge-Grounded Hallucination Firewall

**SAFE-RAG++** is a research-grade pipeline to **diagnose, quantify, and reduce hallucinations** in retrieval-augmented generation (RAG). It blends **causal interventions** with a **multi-signal firewall** (NLI entailment · self-consistency · token-level uncertainty) and ships with an evaluation harness and plots so you can study what actually drives factual failure—**retrieval** or **generation**.

> Why I built this
> RAG is increasingly used for safety-critical decisions. Yet, when answers drift from evidence, we rarely know *why*. SAFE-RAG++ turns hallucination reduction into a **measurement problem**: identify failure modes, attach signals with predictive power, and verify via controlled, causal perturbations.

---

## ✨ Key ideas

* **Causal probes for RAG**
  Drop the top document, randomize contexts, or shuffle order to test if factuality is **retriever-limited** or **generator-limited**.
* **Multi-signal firewall**

  * **Entailment**: RoBERTa-large-MNLI checks if the answer is supported by retrieved passages.
  * **Self-consistency**: sample multiple generations; vote to stabilize outputs.
  * **Uncertainty**: token-level entropy proxy from decoder logits.
* **Evaluation + viz**
  Small SQuAD subset for quick iteration; CSV artifacts + histograms/scatters to analyze relationships between **entailment↓**, **uncertainty↑**, and firewall **flags**.

---

## 🧱 Architecture (at a glance)

```
Question ─┐
          │      ┌───────────────┐       ┌──────────────────────────┐
          └──►   │   Retriever   │ ───►  │  Top-k Evidence Passages │
                 │ (MiniLM + IP) │       └──────────────────────────┘
                 └───────┬───────┘
                         │
                [Causal Interventions: drop-top, random, shuffle]
                         │
                 ┌───────▼───────┐
                 │   Generator   │  (FLAN-T5)
                 └───────┬───────┘
                         │ Answer
      ┌──────────────────┼────────────────────────────┐
      │                  │                            │
 ┌────▼────┐        ┌────▼────────┐             ┌─────▼─────┐
 │  NLI    │        │ Self-Consis │             │ Uncertainty│
 │(MNLI)   │        │ (vote)      │             │ (entropy)  │
 └────┬────┘        └────┬────────┘             └─────┬─────┘
      └───────────────►  Firewall  ◄──────────────────┘
                     (flag/abstain/allow)
```

---

## 📦 Setup

### Option A — pip (CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# If you see NumPy/numba conflicts, uninstall numba/opencv or keep NumPy==1.26.4.
```

### Option B — conda (CPU or GPU)

```bash
conda env create -f environment.yml
conda activate saferagpp
# For GPU builds, remove 'cpuonly' and install pytorch per your CUDA version.
```

---

## 🚀 Run (Notebook)

Open `notebooks/SAFE-RAG++.ipynb` and run top-to-bottom. For CPU:

* `KB_MAX_DOCS = 2000–5000`
* `GEN_MODEL = "google/flan-t5-base"`
* `N = 50` in the eval harness (scale later)

Artifacts land in `results/`:

* `safe_rag_eval.csv` — per-question signals (entailment, entropy, flags, self-consistency, EM).
* `safe_rag_ablation.csv` — variant-wise results for causal interventions.

---

## 🔬 Experiments you can replicate

1. **Retrieval sensitivity**
   Drop top-1 doc vs. shuffle vs. random docs; measure ∆entailment and flag rates.
2. **Signal reliability**
   Plot histograms of entailment and entropy; scatter entailment vs. entropy; quantify correlation with flags.
3. **Consistency helps**
   Compare single-shot vs. self-consistency voting on EM and flag rates.

**What to look for**

* If **dropping the top doc** degrades entailment and raises flags, the system is **retriever-limited**.
* If **random docs** spike flags across the board, your firewall is doing useful work (not just gating by length).
* If **entropy↑** aligns with **entailment↓**, uncertainty is informative (even with a cheap proxy).

---

## 🧭 Extending to paper-ready

* Replace entropy with **mutual information** via MC-Dropout or small **deep ensembles**.
* Train a **domain-tuned verifier** (biomed/legal) and compare to MNLI.
* Formalize an **SCM** for (retrieval quality → context utility → answer factuality) and estimate **ATE**.
* Add **active learning**: fine-tune retriever/generator using firewall-flagged hard cases.
* Evaluate on **TruthfulQA / HotpotQA / BioASQ** and report in-domain vs. OOD.

---

## 🧪 Reproducibility

* `SEED=42` across NumPy/PyTorch.
* FAISS Flat index for deterministic search (HNSW optional for scale).
* Pinned versions (see `requirements.txt` / `environment.yml`).
* Include `pip freeze > results/requirements.lock.txt` with each run.

---

## 🧯 Troubleshooting

* **Transformers not found**: ensure no local `transformers/` folder/file shadows the package.
* **NumPy dtype size changed**: align all packages to **NumPy 1.26.4** (or fully to 2.x), then **restart kernel**.
* **Slow on CPU**: reduce KB size; set `BATCH=128`; use `paraphrase-MiniLM-L3-v2`; **cache** embeddings (`.npy`).
* **No GPU**: it still works; just keep the smaller settings above.

---

## ⚖️ Ethics

The goal is **reliability**: abstain or flag when evidence is weak; surface uncertainty; never “force confidence.”

---

## 📜 License

MIT (suggested). Add your `LICENSE` file.

---

## 📚 Citation

```bibtex
@software{safe-rag-plus-plus_2025,
  author  = {Abhishek Yadav},
  title   = {SAFE-RAG++: A Causal, Uncertainty-Aware, Knowledge-Grounded Hallucination Firewall},
  year    = {2025},
  url     = {https://github.com/<your-username>/<your-repo>},
}
```

---

## 📁 Repo layout

```
.
├── notebooks/
│   └── SAFE-RAG++.ipynb
├── results/
│   ├── safe_rag_eval.csv
│   └── safe_rag_ablation.csv
├── requirements.txt
├── environment.yml
└── README.md
```

