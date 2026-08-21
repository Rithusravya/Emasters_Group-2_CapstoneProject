# CodeGen: Small Code-LM Fine-Tuning & Hybrid RAG Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A capstone project (eMasters CS: AI & ML, Group 2) that fine-tunes a small open-source code language model — **Qwen2.5-Coder-0.5B-Instruct** — with **LoRA**, builds a **hybrid semantic + AST/structural retrieval (RAG)** layer on top of it, and evaluates the result against the un-adapted base model and an independently invoked LLM baseline, across text-to-SQL, documentation-generation, program-synthesis, and commit-message-generation tasks.

---

## Table of Contents

- [Executive Summary](#-executive-summary)
- [Key Capabilities](#-key-capabilities)
- [System Architecture](#-system-architecture)
- [Pipeline Stages](#-pipeline-stages)
- [Benchmark Results](#-benchmark-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Repository Layout](#-repository-layout)
- [Configuration](#-configuration)
- [Requirements](#-requirements)
- [License](#-license)

---

## Executive Summary

CodeGen combines **Retrieval-Augmented Generation (RAG)** and **parameter-efficient LoRA fine-tuning** on top of a compact instruction-tuned code model to support three developer-facing capabilities: natural-language-to-SQL translation, automatic source-code documentation generation, and program/commit-message generation. The primary evaluation dataset is the **Spider** cross-domain text-to-SQL benchmark (1,034 dev-split examples), supplemented by small hand-curated example sets for documentation, program-synthesis, and commit-message generation.

Retrieval embeddings are produced by the small code LM itself (no second embedding model needs to be downloaded), and are combined with an AST/structural fingerprint index so retrieval can be reranked on structure, not just vector similarity.

---

## Key Capabilities

| Capability | Description |
|---|---|
| **Multi-Task Generation** | Program synthesis, docstring generation, Text-to-SQL translation, commit-message synthesis |
| **Code-LM Self-Embeddings** | Retrieval embeddings mean-pooled from the base model's own hidden states — no separate embedding model required (an external `BAAI/bge-small-en-v1.5` path is also available and config-switchable) |
| **Hybrid Retrieval** | FAISS (`IndexFlatIP`) dense semantic search fused with an AST/structural fingerprint reranker (α = 0.7 dense / 0.3 structural) |
| **LoRA Fine-Tuning** | PEFT LoRA adapter (r=16, α=32) on the attention projections, with a custom training loop (gradient checkpointing, prompt-token label masking, OOM-safe stepping) |
| **Small-LM vs. LLM Harness** | Pluggable large-LLM baseline generator supporting local, OpenAI-compatible, and Gemini backends |
| **Comprehensive Evaluation** | BLEU, ROUGE-1/L, token-F1, CodeBERTScore, SQL Exact Match, SQL Execution Accuracy, RAG gain-over-baseline, top-K sweep |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Configuration & Data                                         │
│  • Load config.yaml (paths, model, LoRA, retrieval, eval)       │
│  • Convert & normalize Spider (dev.json + tables.json + SQLite) │
│  • Build retrieval corpus (1,034 question/SQL pairs)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 2. Model Loading & LoRA Fine-Tuning                              │
│  • Load Qwen2.5-Coder-0.5B-Instruct (tokenizer + base weights)  │
│  • Attach LoRA adapter, train on illustrative task examples      │
│  • Save / reload adapter checkpoint                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 3. Generation Tasks                                              │
│  • Text-to-SQL   • Documentation   • Program Synthesis           │
│  • Commit Messages, evaluated for base model vs. LoRA model      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 4. Hybrid Indexing & RAG                                         │
│  • CodeLMEmbedder → FAISS semantic index (IndexFlatIP)           │
│  • ASTIndexer → structural fingerprints (Python AST / SQL regex) │
│  • HybridRetriever → dense + structural blended top-K search     │
│  • RAGPipeline → context injection + generation                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 5. Evaluation & Comparison                                       │
│  • Base vs. LoRA vs. RAG vs. LLM-baseline (ModelComparator)      │
│  • RAG gain-over-baseline & top-K sweep (RAGEvaluator)           │
│  • Charts (matplotlib) + JSON metrics persisted to output/       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

The pipeline is run through a set of Jupyter notebooks rather than a single CLI entry point:

| Notebook | Stage |
|---|---|
| `baseline_model.ipynb` | Loads the **un-adapted base model**, runs text-to-SQL and documentation generation, saves metrics to `outputs/generated/baseline_model_results.json` |
| `finetuned_model.ipynb` | Loads/trains the **LoRA adapter**, re-runs the same two tasks, saves metrics to `outputs/generated/finetuned_model_results.json` |
| `spider_eval.ipynb` | Spider-specific SQL execution evaluation |
| `eval.ipynb` | RAG-gain measurement, top-K sweep, small-LM-vs-LLM comparison |
| `main.ipynb` | Complete End-to-End pipeline |

Supporting production code lives in `src/` (see [Repository Layout](#-repository-layout)) and is imported by the notebooks — it is not currently exposed as a standalone CLI script.

### Task-Specific Generators (`src/generators/`)

| Task | Module | Benchmark used | Primary Metrics |
|---|---|---|---|
| **Text-to-SQL** | `text_to_sql.py`, `sql_generator.py` | Spider (100-sample subset of the 1,034-question dev split) | Exact Match, Execution Accuracy, BLEU, CodeBERTScore |
| **Documentation Generation** | `doc_gen.py` | 3 hand-authored illustrative examples | BLEU, ROUGE, CodeBERTScore, F1 |
| **Program Synthesis** | `program_generator.py` | 3 hand-authored illustrative examples | Qualitative + shared metric suite |
| **Commit Message Generation** | `commit_gen.py` | 3 hand-authored illustrative examples | Qualitative + shared metric suite |
| **LLM Baseline** | `llm_baseline.py` | Same prompts as program synthesis | Comparative scoring against the small LM |

### Hybrid Indexing & Retrieval (`src/embeddings/`)

| Component | Technology | Purpose |
|---|---|---|
| `code_lm_embedder.py` | Mean-pooled hidden states of the base model itself | Retrieval embeddings without a second model download |
| `embedding.py` | `BAAI/bge-small-en-v1.5` (external, config-switchable) | Alternative embedding backend |
| `indexing.py` | FAISS `IndexFlatIP` | Dense semantic vector store (cosine similarity via normalized inner product) |
| `ast_indexing.py` | Python `ast` module (Python code) / regex-based structural extractor (SQL) | Structural fingerprints — function/class names, or referenced tables/aggregates/clauses |
| `hybrid_retriever.py` | Blended reranker | Over-fetches `k × 4` from FAISS, reranks with `α * dense + (1-α) * structural` |

### Evaluation & Persistence (`src/evaluation/`)

| Component | Function |
|---|---|
| `metrics.py` | BLEU, ROUGE, token-F1, CodeBERTScore, execution accuracy primitives |
| `evaluation.py` | Task-level evaluation engine |
| `sql_eval.py` | SQL normalization and Spider-style execution comparison |
| `comparator.py` | Base vs. LoRA vs. RAG side-by-side comparison |
| `rag_evaluation.py` | RAG gain-over-baseline measurement and top-K sweep |
| `visualization.py` | Matplotlib bar-chart generation, saved to `output/plots/` |

---

## Benchmark Results

Numbers below are taken directly from `outputs/generated/baseline_model_results.json` and `finetuned_model_results.json` (100-sample Spider text-to-SQL subset, 3-sample documentation-generation set).

### Baseline (un-adapted) vs. LoRA Fine-Tuned

| Task | Metric | Base Model | LoRA Fine-Tuned |
|---|---|:---:|:---:|
| **Text-to-SQL** | BLEU | 0.2329 | **0.2467** |
| | CodeBERTScore | 0.9740 | **0.9759** |
| | F1 | 0.4642 | **0.4879** |
| | ROUGE-1 | 0.4378 | **0.4602** |
| | ROUGE-L | 0.4274 | **0.4503** |
| | Exact Match | 7% | **8%** |
| | Execution Accuracy | 11% | **14%** |
| **Documentation Generation** | BLEU | 0.0241 | **0.0528** |
| | CodeBERTScore | 0.9250 | **0.9447** |
| | F1 | 0.1181 | **0.2022** |
| | ROUGE-1 | 0.0831 | **0.1615** |
| | ROUGE-L | 0.0719 | **0.1494** |

LoRA fine-tuning (trained on only 9 illustrative prompt/completion examples — 2.16M trainable params, 0.44% of the model) improves every metric on both tasks. Gains are proportionally larger on documentation generation.

### RAG, Top-K Sweep, and LLM Baseline (5-question SQL slice)

- On the small 5-question evaluation slice used for this stage, the **RAG-augmented pipeline underperformed the no-RAG LoRA baseline** on every metric — a genuine, reproducible negative result, not a placeholder. See [Known Limitations](#-known-limitations--future-work) for why.
- A **top-K sweep** over K ∈ {1, 2, 3, 5, 8} shows the gain is least negative around **K = 2–3** and degrades further at K = 8.
- The **LLM-baseline comparison** fell back to the same local Qwen2.5-Coder-0.5B model in this run because no external API credential was resolved at runtime — see the security note below and Known Limitations.
- For reference, the rule-based Spider baseline scores **12.28% exact match / 33.66% execution accuracy** over the full 1,034-question dev set.

Plots for all of the above are saved under `output/plots/` (`scores_baseline_text_to_sql_vs_doc_generation.png`, `scores_finetuned_text_to_sql_vs_doc_generation.png`, `scores_model_comparison_checkpoint2.png`, `scores_model_comparison_full.png`, `topk_gain_sweep.png`).


## Installation

### Prerequisites

```bash
Python 3.10 or higher
A CUDA-capable GPU is optional — the code auto-detects CUDA > MPS (Apple Silicon) > CPU
Recommended: 16 GB RAM, 8 GB+ VRAM if using GPU
```

### Step 1: Clone Repository

```bash
git clone https://github.com/Rithusravya/Emasters_Group-2_CapstoneProject.git
cd Emasters_Group-2_CapstoneProject
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# OR
venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Quick Start

This project is driven through notebooks, not a CLI. Typical order:

1. **Prepare data** — `src/scripts/convert_spider_to_jsonl.py` converts raw Spider files (`data/raw/Spider/`) into `data/processed/spider_eval.jsonl` and a retrieval corpus; `src/scripts/prepare_processed_data.py` builds the MongoDB-style variant.
2. **Baseline evaluation** — open `baseline_model.ipynb` and run all cells to evaluate the un-adapted base model on text-to-SQL and documentation generation.
3. **LoRA fine-tuning + evaluation** — open `finetuned_model.ipynb` to train (or reload) the LoRA adapter at `models/checkpoints/lora_finetuned/` and re-run the same evaluations.
4. **RAG / top-K / LLM comparison** — open `eval.ipynb` to build the FAISS + AST indices, measure RAG gain, sweep top-K, and compare against the LLM baseline.
5. **Spider execution evaluation** — open `spider_eval.ipynb` for SQL-execution-based scoring against the SQLite databases.

Programmatic usage of the core pipeline, e.g. for text-to-SQL generation:

```python
from src.config import load_config
from src.models.load_model import ModelLoader
from src.generators.program_generator import GenerationPipeline
from src.generators.text_to_sql import TextToSQLGenerator

config = load_config()  # reads configs/config.yaml
loader = ModelLoader(config)
models, tokenizer = loader.load_models(lora_path=config.lora.output_dir)

pipeline = GenerationPipeline(models["lora"] or models["base"], tokenizer, config.generation)
sql_gen = TextToSQLGenerator(pipeline)

result = sql_gen.generate_queries(
    question="How many singers are there?",
    schema="CREATE TABLE singer (singer_id INT, name TEXT, age INT)",
)
print(result["sql_query"])
```

---

## Repository Layout

```
Emasters_Group-2_CapstoneProject/
├── configs/
│   └── config.yaml                    # Project, model, LoRA, retrieval, eval & output paths
├── data/
│   ├── raw/                           # Raw Spider files (dev.json, tables.json, SQLite DBs)
│   ├── processed/                     # Normalized JSONL (spider_eval, retrieval_corpus, etc.)
│   └── indices/                       # Saved FAISS index, AST store, FAISS metadata
├── models/
│   └── checkpoints/lora_finetuned/    # Saved LoRA adapter + tokenizer files
├── src/
│   ├── config.py                      # YAML → dataclass config loader
│   ├── data/
│   │   └── data_loader.py             # Generic JSON/JSONL dataset loader (Spider, etc.)
│   ├── models/
│   │   └── load_model.py              # Base model + LoRA loading, training loop, base-vs-LoRA generation
│   ├── generators/
│   │   ├── program_generator.py       # Core GenerationPipeline + prompt builder
│   │   ├── text_to_sql.py             # Text-to-SQL generator
│   │   ├── sql_generator.py           # SQL prompt/parsing helpers
│   │   ├── doc_gen.py                 # Documentation generator
│   │   ├── commit_gen.py              # Commit-message generator
│   │   └── llm_baseline.py            # Pluggable local/OpenAI/Gemini LLM baseline
│   ├── embeddings/
│   │   ├── code_lm_embedder.py        # Embeddings from the small code LM's own hidden states
│   │   ├── embedding.py               # External embedding-model backend (BGE)
│   │   ├── indexing.py                # FAISS semantic index manager
│   │   ├── ast_indexing.py            # AST / SQL structural fingerprint indexer
│   │   └── hybrid_retriever.py        # Dense + structural blended retrieval
│   ├── rag/
│   │   └── rag_pipeline.py            # Context retrieval, prompt building, RAG generation
│   ├── evaluation/
│   │   ├── metrics.py                 # BLEU, ROUGE, F1, CodeBERTScore, execution accuracy
│   │   ├── evaluation.py              # Task-level evaluation engine
│   │   ├── sql_eval.py                # Spider SQL normalization & execution comparison
│   │   ├── comparator.py              # Base vs. LoRA vs. RAG comparison
│   │   ├── rag_evaluation.py          # RAG gain-over-baseline + top-K sweep
│   │   └── visualization.py           # Chart generation
│   └── scripts/
│       ├── convert_spider_to_jsonl.py # Raw Spider → normalized JSONL
│       └── prepare_processed_data.py  # Build derived/MongoDB-style datasets
├── outputs/generated/                 # Per-task generated samples & timestamped run results
├── output/plots/                      # Saved comparison charts
├── backup/                            # Earlier snapshot of evaluation/rag/embeddings modules
├── baseline_model.ipynb               # Base model evaluation
├── finetuned_model.ipynb              # LoRA fine-tuning + evaluation
├── spider_eval.ipynb                  # Spider execution-based SQL evaluation
├── eval.ipynb                         # RAG gain / top-K sweep / LLM comparison
├── main.ipynb                         # Colab entry point
├── requirements.txt
└── README.md
```

---

## Configuration (`configs/config.yaml`)

Key fields actually read by `src/config.py`:

```yaml
model_name: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
lora_adapter_path: "models/checkpoints/lora_finetuned"
embedding_model: "BAAI/bge-small-en-v1.5"   # used only when embedding.source == "external"

embedding:
  source: "code_lm"           # "code_lm" (default, self-embeddings) or "external"

retrieval:
  hybrid: true
  structural_weight_alpha: 0.7
  structural_language: "sql"
  topk_sweep: [1, 2, 3, 5, 8]

llm_baseline:
  backend: "local"            # "local" or an external provider ("openai" / "gemini")
  local_model: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  api_model: "gpt-5.4-mini"
  max_new_tokens: 256
  temperature: 0.2

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  batch_size: 4
  learning_rate: 0.0003
  epochs: 5
  output_dir: "models/checkpoints/lora_finetuned"

indexing:
  top_k: 3
  vector_dim: 1024            # must match the embedding model's output dimension

outputs:
  logs_dir: "output/logs"
  metrics_path: "output/metrics/evaluation_metrics.json"
  plots_dir: "output/plots"
```

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.5.1 | Core deep learning framework |
| `transformers` | 4.46.3 | Pretrained model + tokenizer loading (Qwen2.5-Coder) |
| `peft` | 0.13.2 | LoRA fine-tuning |
| `datasets` | 3.1.0 | Dataset utilities |
| `faiss-cpu` | 1.9.0 | Dense vector similarity search |
| `tree-sitter` / `tree-sitter-python` | 0.23.2 / 0.23.6 | (Available for AST-based parsing) |
| `codebleu` | 0.7.0 | CodeBLEU metric |
| `codebert-score` | 0.3.13 | CodeBERTScore semantic similarity metric |
| `evaluate` | 0.4.3 | Hugging Face evaluation metrics |
| `rouge-score` | 0.1.2 | ROUGE metrics |
| `nltk` | 3.9.1 | Text preprocessing / BLEU support |
| `PyYAML` | 6.0.2 | Config parsing |
| `matplotlib` / `seaborn` | 3.10.0 / 0.13.2 | Plotting |
| `numpy` | 2.1.3 | Numerical computing |

See `requirements.txt` for the full pinned list.

---

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## Citation

```bibtex
@software{codegen_capstone_2026,
  title  = {CodeGen: Small Code-LM Fine-Tuning and Hybrid RAG Pipeline},
  author = {Emasters Group 2},
  year   = {2026},
  url    = {https://github.com/Rithusravya/Emasters_Group-2_CapstoneProject}
}
```

---

<div align="center">

**Built by IIIT-H eMasters Group 2**

[⬆ Back to Top](#-codegen-small-code-lm-fine-tuning--hybrid-rag-pipeline)

</div>
